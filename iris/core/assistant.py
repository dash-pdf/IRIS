"""
iris/core/assistant.py
----------------------
Decoupled IRIS Voice & Intelligence Orchestrator.

Coordinates independent components into a rock-solid, half-duplex conversational system:
- State Machine (IDLE -> LISTENING -> THINKING -> SPEAKING -> COOLDOWN -> IDLE)
- Audio Capture with software mute gate
- Energy-based Voice Activity Detection (VAD)
- Transcript Self-Echo Filter (eliminates speaker-to-mic feedback loops)
- Decoupled Speech-to-Text (STT) Adapter
- Decoupled Intelligence (LLM) Adapter
- Decoupled Text-to-Speech (TTS) Adapter
- Audio Player with instant abort
- Durable Markdown Conversation Archive
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from iris.adapters.model.base import ModelAdapter
from iris.adapters.stt import create_stt_adapter
from iris.adapters.stt.base import STTAdapter
from iris.adapters.tts import create_tts_adapter, shape_speech_text
from iris.adapters.tts.base import TTSAdapter
from iris.audio.capture import AudioCapture
from iris.audio.echo_filter import EchoFilter
from iris.audio.player import AudioPlayer
from iris.audio.vad import VoiceActivityDetector
from iris.core.config import IRISConfig, load_config
from iris.core.state import AudioState, AudioStateMachine
from iris.services.archive import ConversationArchive
from iris.services.conversation import (
    ConversationService,
    IRIS_LOCAL_SYSTEM_PROMPT,
    IRIS_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class IRISAssistant:
    """
    Main IRIS runtime coordinator.
    """

    def __init__(
        self,
        config: IRISConfig | None = None,
        model_adapter: ModelAdapter | None = None,
        tts_adapter: TTSAdapter | None = None,
        stt_adapter: STTAdapter | None = None,
        on_state_change: Callable[[AudioState, AudioState, str], None] | None = None,
        on_user_transcript: Callable[[str], None] | None = None,
        on_assistant_response: Callable[[str], None] | None = None,
        on_echo_detected: Callable[[str, str], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_silence_warning: Callable[[], None] | None = None,
    ) -> None:
        self.config = config or load_config(require_voice=False)

        # Adapters
        if model_adapter is not None:
            self.model = model_adapter
        else:
            from iris.adapters.model import create_model_adapter

            self.model = create_model_adapter(self.config)
        self.tts = tts_adapter or create_tts_adapter(self.config)
        self.stt = stt_adapter or create_stt_adapter(self.config)

        # Callbacks
        self.on_state_change = on_state_change
        self.on_user_transcript = on_user_transcript
        self.on_assistant_response = on_assistant_response
        self.on_echo_detected = on_echo_detected
        self.on_error = on_error
        self.on_silence_warning = on_silence_warning

        # State Machine
        self.state_machine = AudioStateMachine(on_change=self._handle_state_change)

        # Audio Components
        self.player = getattr(self.tts, "_player", AudioPlayer())
        self.capture = AudioCapture(
            on_frame=self._on_audio_frame,
            device=self.config.audio_input_device,
            on_silence_warning=self._handle_silence_warning,
        )
        self.echo_filter = EchoFilter(similarity_threshold=self.config.echo_similarity_threshold)
        self.vad = VoiceActivityDetector(
            sample_rate=self.config.sample_rate,
            energy_threshold=self.config.vad_energy_threshold,
            silence_timeout_ms=self.config.silence_timeout_ms,
            on_speech_start=self._on_speech_start,
        )

        # Services
        system_prompt = (
            IRIS_LOCAL_SYSTEM_PROMPT
            if self.config.llm_provider.lower() == "ollama"
            else IRIS_SYSTEM_PROMPT
        )
        self.conversation = ConversationService(system_prompt=system_prompt)
        self.archive = ConversationArchive(
            archive_dir=self.config.archive_dir,
            persona_name=self.config.persona_name,
        )

        self._is_running = False
        self._turn_lock = threading.Lock()
        self.last_turn_timings: dict[str, Any] = {}

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_state(self) -> AudioState:
        return self.state_machine.state

    def _handle_state_change(self, old_state: AudioState, new_state: AudioState, turn_id: str) -> None:
        logger.info("State changed: %s -> %s [turn: %s]", old_state.value, new_state.value, turn_id[:8])
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state, turn_id)
            except Exception as exc:
                logger.error("Error in on_state_change callback: %s", exc)

    def _handle_silence_warning(self) -> None:
        if self.on_silence_warning:
            try:
                self.on_silence_warning()
            except Exception as exc:
                logger.error("Error in on_silence_warning callback: %s", exc)

    def restore_conversation_history(self, max_turns: int = 10) -> int:
        """
        Restore recent turns from the archive into the in-memory conversation.
        Returns the number of turns restored.
        """
        turns = self.archive.load_recent_turns(max_turns=max_turns)
        if turns:
            self.conversation.restore_turns(turns)
            logger.info("Restored %d turns from previous session into memory context", len(turns))
            return len(turns)
        return 0

    def start(self) -> None:
        """Start the assistant and audio capture."""
        if self._is_running:
            return

        self._is_running = True
        if self.conversation.turn_count == 0:
            self.restore_conversation_history(max_turns=10)

        self.archive.open_session()
        self.state_machine.reset()
        self.capture.unmute()
        self.capture.start(on_frame=self._on_audio_frame)
        logger.info(
            "IRIS assistant started (Model: %s, TTS: %s, STT: %s)",
            self.model.__class__.__name__,
            self.tts.voice_name,
            self.stt.provider_name,
        )

    def stop(self) -> None:
        """Stop audio processing and cleanly shut down."""
        if not self._is_running:
            return

        self._is_running = False
        self.capture.stop()
        self.player.stop()
        if hasattr(self.tts, "stop"):
            self.tts.stop()
        self.state_machine.cancel()

        try:
            self.archive.close_session()
        except Exception as exc:
            logger.debug("Archive session close notice: %s", exc)

        logger.info("IRIS assistant stopped")

    def _on_speech_start(self) -> None:
        """Fired by VAD when user begins speaking."""
        if self.state_machine.state == AudioState.IDLE:
            self.state_machine.transition_to(AudioState.LISTENING)

    def _on_audio_frame(self, frame_bytes: bytes) -> None:
        """Microphone audio callback (16kHz 16-bit mono chunks)."""
        if not self._is_running:
            return

        # Half-duplex guard: Drop frame if muted or assistant is active
        if self.capture.is_muted or not self.state_machine.is_mic_active:
            return

        utterance = self.vad.process_frame(frame_bytes)
        if utterance:
            # End of user utterance detected!
            speech_end_time = time.perf_counter()
            turn_id = self.state_machine.new_turn()
            if self.state_machine.transition_to(AudioState.THINKING, expected_turn_id=turn_id):
                # Immediately mute microphone to eliminate any chance of feedback
                self.capture.mute()
                threading.Thread(
                    target=self._process_utterance,
                    args=(utterance, turn_id, speech_end_time),
                    name=f"iris-turn-{turn_id[:8]}",
                    daemon=True,
                ).start()

    def _process_utterance(
        self,
        utterance_bytes: bytes,
        turn_id: str,
        speech_end_time: float | None = None,
    ) -> None:
        """Process transcribed speech through echo-filter, LLM, and TTS."""
        with self._turn_lock:
            try:
                t_utterance_start = time.perf_counter()
                if speech_end_time is None:
                    speech_end_time = t_utterance_start

                # Stage 1: Microphone capture duration
                capture_duration_s = len(utterance_bytes) / (2.0 * self.config.sample_rate)

                # Stage 2: Speech-to-text processing time
                t_stt_start = time.perf_counter()
                transcript = self.stt.transcribe(utterance_bytes, sample_rate=self.config.sample_rate).strip()
                t_stt_end = time.perf_counter()
                stt_duration_s = t_stt_end - t_stt_start

                if not transcript:
                    logger.debug("Empty transcript, resetting to IDLE")
                    self._reset_to_idle()
                    return

                logger.info("Transcribed [turn %s]: '%s' (STT took %.3fs)", turn_id[:8], transcript, stt_duration_s)

                # Check for Acoustic Self-Echo
                is_echo, echo_reason = self.echo_filter.is_echo(transcript)
                if is_echo:
                    logger.warning(
                        "DISCARDED ACOUSTIC SELF-ECHO [turn %s]: '%s' (Reason: %s)",
                        turn_id[:8],
                        transcript,
                        echo_reason,
                    )
                    if self.on_echo_detected:
                        self.on_echo_detected(transcript, echo_reason)
                    self._reset_to_idle()
                    return

                # Valid user turn
                if self.on_user_transcript:
                    self.on_user_transcript(transcript)
                self.conversation.add_user_turn(transcript)
                self.archive.write_turn("user", transcript)

                # Stage 3: Time waiting for the LLM response, including the HTTP request
                # Stage 4: Text generation before response is complete (non-streaming: full response received atomically in Stage 3)
                messages = self.conversation.get_messages()
                t_llm_start = time.perf_counter()
                response_text = self.model.generate_response(messages).strip()
                t_llm_end = time.perf_counter()
                llm_duration_s = t_llm_end - t_llm_start

                if not response_text:
                    response_text = "Understood."

                logger.info("Generated response [turn %s]: '%s' (LLM took %.3fs)", turn_id[:8], response_text, llm_duration_s)
                if self.on_assistant_response:
                    self.on_assistant_response(response_text)

                speech_text = shape_speech_text(response_text) or response_text
                # Register with echo filter BEFORE speaking aloud
                self.echo_filter.register_assistant_response(speech_text)
                self.conversation.add_assistant_turn(response_text)
                self.archive.write_turn("assistant", response_text)

                # Speech Output (SPEAKING -> COOLDOWN -> IDLE)
                if not self.state_machine.transition_to(AudioState.SPEAKING, expected_turn_id=turn_id):
                    logger.warning("Turn cancelled before speech playback")
                    self._reset_to_idle()
                    return

                def on_done() -> None:
                    self._on_playback_completed(turn_id)

                # Stage 5: Text-to-speech generation time
                # Stage 6: Total time from end of user speech to start of audible playback
                t_tts_start = time.perf_counter()
                if hasattr(self.tts, "speak"):
                    self.tts.speak(speech_text, on_done=on_done)
                else:
                    samples, sample_rate = self.tts.synthesize(speech_text)
                    self.player.play(samples, sample_rate=sample_rate, on_done=on_done)
                t_playback_start = time.perf_counter()
                tts_duration_s = t_playback_start - t_tts_start

                total_latency_s = t_playback_start - speech_end_time

                # Record turn timings and intelligence routing diagnostics
                selected_model = getattr(self.model, "last_selected_model", None) or getattr(self.model, "model_name", None)
                provider_name = getattr(self.model, "last_provider_name", None) or type(self.model).__name__
                route = getattr(self.model, "last_route", None)
                routing_reason = getattr(self.model, "last_routing_reason", None)
                fallback_decision = getattr(self.model, "last_fallback_decision", None)
                model_metrics = getattr(self.model, "last_metrics", None)

                self.last_turn_timings = {
                    "turn_id": turn_id,
                    "capture_duration_s": capture_duration_s,
                    "stt_duration_s": stt_duration_s,
                    "llm_duration_s": llm_duration_s,
                    "llm_streaming": False,
                    "tts_duration_s": tts_duration_s,
                    "total_latency_s": total_latency_s,
                    "selected_model": selected_model,
                    "provider_name": provider_name,
                    "route": route,
                    "routing_reason": routing_reason,
                    "fallback_decision": fallback_decision,
                }
                if model_metrics:
                    self.last_turn_timings["model_metrics"] = model_metrics

                routing_telemetry_line = ""
                if route or routing_reason:
                    routing_telemetry_line = f"\n   -> Intelligence Routing: [{route or 'direct'}] {routing_reason or ''}"
                    if fallback_decision:
                        routing_telemetry_line += f" | Notice: {fallback_decision}"

                if model_metrics:
                    load_s = model_metrics.get("load_duration_s")
                    if isinstance(load_s, (int, float)):
                        prompt_tokens = model_metrics.get("prompt_eval_count")
                        prompt_s = model_metrics.get("prompt_eval_duration_s")
                        eval_tokens = model_metrics.get("eval_count")
                        eval_s = model_metrics.get("eval_duration_s")
                        tps = model_metrics.get("tokens_per_second", 0.0)
                        load_str = f"{load_s:.3f}s"
                        prompt_dur_str = f"{prompt_s:.3f}s" if isinstance(prompt_s, (int, float)) else "N/A"
                        eval_dur_str = f"{eval_s:.3f}s" if isinstance(eval_s, (int, float)) else "N/A"
                        routing_telemetry_line += (
                            f"\n   -> Model Telemetry: load={load_str} | "
                            f"prompt_eval={prompt_tokens or 0} tok ({prompt_dur_str}) | "
                            f"eval={eval_tokens or 0} tok ({eval_dur_str}, {tps:.1f} tok/s)"
                        )
                    elif "http_duration_s" in model_metrics:
                        http_s = model_metrics.get("http_duration_s", 0.0)
                        routing_telemetry_line += f" (Provider HTTP: {http_s:.3f}s)"

                logger.info(
                    "\n============================================================\n"
                    "[TURN TIMING BREAKDOWN] Turn: %s\n"
                    "1. Microphone Capture Duration:       %.3fs (%dms)\n"
                    "2. Speech-to-Text (STT) Processing:   %.3fs (%dms) [%s]\n"
                    "3. Intelligence (LLM / Tools):        %.3fs (%dms) [%s / %s]%s\n"
                    "4. Text Generation (Streaming):       Not implemented (non-streaming; full response received in Stage 3)\n"
                    "5. Text-to-Speech (TTS) Generation:   %.3fs (%dms) [%s]\n"
                    "6. Speech-End to Playback Total:      %.3fs (%dms)\n"
                    "============================================================",
                    turn_id[:8],
                    capture_duration_s,
                    int(capture_duration_s * 1000),
                    stt_duration_s,
                    int(stt_duration_s * 1000),
                    getattr(self.stt, "provider_name", type(self.stt).__name__),
                    llm_duration_s,
                    int(llm_duration_s * 1000),
                    provider_name,
                    selected_model or type(self.model).__name__,
                    routing_telemetry_line,
                    tts_duration_s,
                    int(tts_duration_s * 1000),
                    getattr(self.tts, "voice_name", type(self.tts).__name__),
                    total_latency_s,
                    int(total_latency_s * 1000),
                )

            except Exception as exc:
                logger.error("Error during utterance processing [turn %s]: %s", turn_id[:8], exc)
                try:
                    self.archive.write_turn("error", str(exc))
                except Exception:
                    pass
                if self.on_error:
                    self.on_error(exc)
                self._reset_to_idle()

    def _on_playback_completed(self, turn_id: str) -> None:
        """Called when audio player finishes speaking."""
        # Transition to COOLDOWN
        self.state_machine.transition_to(AudioState.COOLDOWN, expected_turn_id=turn_id)
        logger.debug("Playback complete. Cooling down for %d ms", self.config.cooldown_ms)

        # Allow room reverberation and speaker acoustic decay to dissipate
        time.sleep(self.config.cooldown_ms / 1000.0)

        # Reset VAD buffer and transition to IDLE
        self.vad.reset()
        self.state_machine.transition_to(AudioState.IDLE, expected_turn_id=turn_id)

        # Unmute microphone only after cooldown has elapsed
        self.capture.unmute()
        logger.debug("Microphone unmuted. IRIS is now IDLE and listening.")

    def _reset_to_idle(self) -> None:
        """Safely reset state to IDLE and unmute microphone."""
        self.vad.reset()
        self.state_machine.transition_to(AudioState.IDLE)
        self.capture.unmute()

    def process_text_turn(self, text: str, wait_for_playback: bool = True) -> str:
        """
        Process a text message directly without microphone input.
        Useful for console fallback, testing, or keyboard chat.
        """
        # If currently speaking or in cooldown, wait for previous utterance to finish
        if self.state_machine.state in {AudioState.SPEAKING, AudioState.COOLDOWN}:
            self.player.wait()
            while self.state_machine.state != AudioState.IDLE:
                time.sleep(0.05)

        turn_id = self.state_machine.new_turn()
        text_turn_start = time.perf_counter()
        self.state_machine.transition_to(AudioState.THINKING, expected_turn_id=turn_id)

        transcript = text.strip()
        if self.on_user_transcript:
            self.on_user_transcript(transcript)
        self.conversation.add_user_turn(transcript)
        self.archive.write_turn("user", transcript)

        try:
            messages = self.conversation.get_messages()
            t_llm_start = time.perf_counter()
            response_text = self.model.generate_response(messages).strip()
            t_llm_end = time.perf_counter()
            llm_duration_s = t_llm_end - t_llm_start

            if not response_text:
                response_text = "Understood."

            speech_text = shape_speech_text(response_text) or response_text
            self.echo_filter.register_assistant_response(speech_text)
            if self.on_assistant_response:
                self.on_assistant_response(response_text)
            self.conversation.add_assistant_turn(response_text)
            self.archive.write_turn("assistant", response_text)

            # Synthesize and play
            self.state_machine.transition_to(AudioState.SPEAKING, expected_turn_id=turn_id)

            def on_done() -> None:
                self._on_playback_completed(turn_id)

            t_tts_start = time.perf_counter()
            if hasattr(self.tts, "speak"):
                self.tts.speak(speech_text, blocking=False, on_done=on_done)
            else:
                samples, sample_rate = self.tts.synthesize(speech_text)
                self.player.play(samples, sample_rate=sample_rate, on_done=on_done)
            t_playback_start = time.perf_counter()
            tts_duration_s = t_playback_start - t_tts_start
            total_latency_s = t_playback_start - text_turn_start

            selected_model = getattr(self.model, "last_selected_model", None) or getattr(self.model, "model_name", None)
            provider_name = getattr(self.model, "last_provider_name", None) or type(self.model).__name__
            route = getattr(self.model, "last_route", None)
            routing_reason = getattr(self.model, "last_routing_reason", None)
            fallback_decision = getattr(self.model, "last_fallback_decision", None)
            model_metrics = getattr(self.model, "last_metrics", None)

            self.last_turn_timings = {
                "turn_id": turn_id,
                "capture_duration_s": 0.0,
                "stt_duration_s": 0.0,
                "llm_duration_s": llm_duration_s,
                "llm_streaming": False,
                "tts_duration_s": tts_duration_s,
                "total_latency_s": total_latency_s,
                "selected_model": selected_model,
                "provider_name": provider_name,
                "route": route,
                "routing_reason": routing_reason,
                "fallback_decision": fallback_decision,
            }
            if model_metrics:
                self.last_turn_timings["model_metrics"] = model_metrics

            routing_telemetry_line = ""
            if route or routing_reason:
                routing_telemetry_line = f"\n   -> Intelligence Routing: [{route or 'direct'}] {routing_reason or ''}"
                if fallback_decision:
                    routing_telemetry_line += f" | Notice: {fallback_decision}"

            if model_metrics:
                load_s = model_metrics.get("load_duration_s")
                if isinstance(load_s, (int, float)):
                    prompt_tokens = model_metrics.get("prompt_eval_count")
                    prompt_s = model_metrics.get("prompt_eval_duration_s")
                    eval_tokens = model_metrics.get("eval_count")
                    eval_s = model_metrics.get("eval_duration_s")
                    tps = model_metrics.get("tokens_per_second", 0.0)
                    load_str = f"{load_s:.3f}s"
                    prompt_dur_str = f"{prompt_s:.3f}s" if isinstance(prompt_s, (int, float)) else "N/A"
                    eval_dur_str = f"{eval_s:.3f}s" if isinstance(eval_s, (int, float)) else "N/A"
                    routing_telemetry_line += (
                        f"\n   -> Model Telemetry: load={load_str} | "
                        f"prompt_eval={prompt_tokens or 0} tok ({prompt_dur_str}) | "
                        f"eval={eval_tokens or 0} tok ({eval_dur_str}, {tps:.1f} tok/s)"
                    )
                elif "http_duration_s" in model_metrics:
                    http_s = model_metrics.get("http_duration_s", 0.0)
                    routing_telemetry_line += f" (Provider HTTP: {http_s:.3f}s)"

            logger.info(
                "\n============================================================\n"
                "[TEXT TURN TIMING BREAKDOWN] Turn: %s\n"
                "1. Microphone Capture Duration:       0.000s (N/A - text turn)\n"
                "2. Speech-to-Text (STT) Processing:   0.000s (N/A - text turn)\n"
                "3. Intelligence (LLM / Tools):        %.3fs (%dms) [%s / %s]%s\n"
                "4. Text Generation (Streaming):       Not implemented (non-streaming; full response received in Stage 3)\n"
                "5. Text-to-Speech (TTS) Generation:   %.3fs (%dms) [%s]\n"
                "6. Processing Latency to Audio:       %.3fs (%dms)\n"
                "============================================================",
                turn_id[:8],
                llm_duration_s,
                int(llm_duration_s * 1000),
                provider_name,
                selected_model or type(self.model).__name__,
                routing_telemetry_line,
                tts_duration_s,
                int(tts_duration_s * 1000),
                getattr(self.tts, "voice_name", type(self.tts).__name__),
                total_latency_s,
                int(total_latency_s * 1000),
            )

            if wait_for_playback:
                self.player.wait()

            return response_text
        except Exception as exc:
            logger.error("Error during text turn processing [turn %s]: %s", turn_id[:8], exc)
            self._reset_to_idle()
            raise

