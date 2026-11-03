"""
iris/adapters/voice_elevenlabs.py
----------------------------------
Voice adapter: ElevenLabs Conversational AI (ElevenAgents).

Responsibility:
- Connect to ElevenLabs ConvAI via WebSocket.
- Stream microphone audio to ElevenLabs STT.
- Receive transcript events (what the user said).
- Send response text back to ElevenLabs TTS.
- Receive and play synthesised audio.
- Surface conversation events (user spoke, IRIS replied) to the caller
  via callbacks, so the archive and conversation service can be updated.

What this adapter does NOT do:
- It does not call the LLM. That is the model adapter's job.
- It does not write to the Markdown archive. That is the archive's job.
- It does not manage conversation history. That is the conversation service.

All ElevenLabs-specific SDK code is isolated here. If we ever switch to
Azure Speech or a local Whisper + TTS stack, only this file changes.

Architecture note (Config B):
ElevenLabs ConvAI calls our custom LLM server (poc_voice_loop.py acts
as that server for Phase 0). The ElevenLabs SDK's Conversation class
handles the WebSocket, STT, turn-taking, and TTS automatically. We
provide callbacks to intercept events.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from elevenlabs.conversational_ai.conversation import AudioInterface

logger = logging.getLogger(__name__)


# ── SoundDevice Audio Interface (macOS & Windows cross-platform) ──────────────

class SoundDeviceAudioInterface(AudioInterface):
    """
    Cross-platform audio interface using sounddevice.

    Does not require system-level C libraries or compilers.
    Streams 16-bit PCM mono audio at 16kHz for microphone input and
    speaker output.
    """

    INPUT_FRAMES_PER_BUFFER = 4000   # 250ms @ 16kHz
    OUTPUT_FRAMES_PER_BUFFER = 1000  # 62.5ms @ 16kHz
    SAMPLE_RATE = 16000

    def __init__(self) -> None:
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError as exc:
            raise ImportError(
                "sounddevice is required for audio input/output. "
                "Run: pip install sounddevice numpy"
            ) from exc

        self._input_callback: Callable[[bytes], None] | None = None
        self._output_queue: queue.Queue[bytes] = queue.Queue()
        self._should_stop = threading.Event()
        self._output_thread: threading.Thread | None = None
        self._in_stream: Any = None
        self._out_stream: Any = None

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        self._input_callback = input_callback
        self._should_stop.clear()

        # Start output worker thread
        self._output_thread = threading.Thread(
            target=self._run_output_worker,
            name="iris-audio-out",
            daemon=True,
        )

        # 16-bit PCM, mono, 16kHz input stream
        self._in_stream = self._sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.INPUT_FRAMES_PER_BUFFER,
            callback=self._in_callback,
        )

        # 16-bit PCM, mono, 16kHz output stream
        self._out_stream = self._sd.RawOutputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.OUTPUT_FRAMES_PER_BUFFER,
        )

        self._in_stream.start()
        self._out_stream.start()
        self._output_thread.start()

    def stop(self) -> None:
        self._should_stop.set()
        if self._output_thread and self._output_thread.is_alive():
            self._output_thread.join(timeout=1.0)

        if self._in_stream:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass

        if self._out_stream:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass

    def output(self, audio: bytes) -> None:
        """Enqueue audio chunk to be played."""
        self._output_queue.put(audio)

    def interrupt(self) -> None:
        """Clear any buffered audio when user interrupts the assistant."""
        try:
            while True:
                self._output_queue.get_nowait()
        except queue.Empty:
            pass

    def _in_callback(self, in_data: Any, frames: int, time_info: Any, status: Any) -> None:
        if self._input_callback and in_data:
            self._input_callback(bytes(in_data))

    def _run_output_worker(self) -> None:
        while not self._should_stop.is_set():
            try:
                audio = self._output_queue.get(timeout=0.1)
                if self._out_stream and not self._out_stream.closed:
                    self._out_stream.write(audio)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.debug("Audio output stream write error: %s", exc)
                break


# ── Type aliases for callback signatures ──────────────────────────────────────

# Called when the user's speech is transcribed.
# Arg: transcript string (what the user said).
OnUserTranscript = Callable[[str], None]

# Called when IRIS's response has been generated and is being spoken.
# Arg: response string (what IRIS is saying).
OnAssistantResponse = Callable[[str], None]

# Called when a session-level error occurs.
# Arg: error message string (must not contain secrets).
OnError = Callable[[str], None]


class ElevenLabsVoiceAdapter:
    """
    Manages a single ElevenLabs ConvAI conversation session.

    Usage (Phase 0 PoC):
        adapter = ElevenLabsVoiceAdapter(
            api_key="...",
            agent_id="...",
            on_user_transcript=my_transcript_handler,
            on_assistant_response=my_response_handler,
            on_error=my_error_handler,
        )
        adapter.start()    # Blocks until conversation ends or stop() is called.
        adapter.stop()

    The ElevenLabs SDK's `Conversation` class handles:
        - WebSocket connection lifecycle.
        - Microphone capture (via sounddevice on macOS).
        - STT (Scribe ASR).
        - Turn-taking model (when the user has finished speaking).
        - TTS playback (speaks IRIS's response through the speakers).

    Our custom LLM is configured in the ElevenLabs agent dashboard as a
    "Custom LLM" endpoint. ElevenLabs calls that endpoint; this adapter
    does not need to make LLM calls itself in Config B.

    For Phase 0 PoC specifically, we configure the agent to use an echo
    response (or a simple static prompt) so we can test the voice loop
    without a real LLM call.
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        on_user_transcript: OnUserTranscript | None = None,
        on_assistant_response: OnAssistantResponse | None = None,
        on_error: OnError | None = None,
    ) -> None:
        self._api_key = api_key
        self._agent_id = agent_id
        self._on_user_transcript = on_user_transcript or _noop_str
        self._on_assistant_response = on_assistant_response or _noop_str
        self._on_error = on_error or _noop_str
        self._conversation: Any = None  # elevenlabs.Conversation instance
        self._should_stop = threading.Event()

    def start(self) -> None:
        """
        Start a ConvAI session and block until it ends.

        Connects to ElevenLabs, opens the microphone, and enters the
        conversation loop. Returns when the session ends (either naturally
        or because stop() was called from another thread).

        Raises:
            ImportError: If the elevenlabs package is not installed.
            RuntimeError: If the ElevenLabs connection fails.
        """
        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs.conversational_ai.conversation import Conversation
        except ImportError as exc:
            raise ImportError(
                "The 'elevenlabs' package is not installed. "
                "Run: pip install elevenlabs"
            ) from exc

        logger.info("Connecting to ElevenLabs ConvAI (agent_id=%s)…", self._agent_id)

        client = ElevenLabs(api_key=self._api_key)

        # Determine if signed URL is supported by the key; otherwise connect directly
        requires_auth = False
        if self._api_key:
            try:
                client.conversational_ai.get_signed_url(agent_id=self._agent_id)
                requires_auth = True
            except Exception:
                requires_auth = False

        self._conversation = Conversation(
            client=client,
            agent_id=self._agent_id,
            requires_auth=requires_auth,
            # SoundDeviceAudioInterface uses sounddevice directly for
            # cross-platform mic input and speaker playback (macOS CoreAudio, Windows WASAPI).
            audio_interface=SoundDeviceAudioInterface(),
            # Callback: ElevenLabs has recognised what the user said.
            callback_agent_response=self._handle_assistant_response,
            # Callback: partial transcript while user is still speaking.
            callback_agent_response_correction=self._handle_response_correction,
            # Callback: final transcript of what the user said.
            callback_user_transcript=self._handle_user_transcript,
            # Callback: latency information (for monitoring).
            callback_latency_measurement=self._handle_latency,
        )

        try:
            self._should_stop.clear()
            self._conversation.start_session()
            # Keep main thread alive while session is running
            while not self._should_stop.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Session interrupted by user.")
        except Exception as exc:  # noqa: BLE001
            safe_msg = f"ElevenLabs session error: {type(exc).__name__}"
            logger.exception("ElevenLabs session failed")
            self._on_error(safe_msg)
        finally:
            self._end_session_safely()

    def stop(self) -> None:
        """Request the current session to end gracefully."""
        self._should_stop.set()
        if self._conversation is not None:
            try:
                self._conversation.end_session()
            except Exception:  # noqa: BLE001
                logger.debug("Error ending ElevenLabs session (ignored on stop).")

    # ── Internal callbacks ────────────────────────────────────────────────

    def _handle_user_transcript(self, transcript: str) -> None:
        """Called by ElevenLabs SDK when the user's speech is fully transcribed."""
        logger.debug("User transcript: %r", transcript)
        if transcript.strip():
            self._on_user_transcript(transcript.strip())

    def _handle_assistant_response(self, response: str) -> None:
        """Called by ElevenLabs SDK when IRIS's response text is available."""
        logger.debug("Assistant response: %r", response)
        if response.strip():
            self._on_assistant_response(response.strip())

    def _handle_response_correction(self, original: str, corrected: str) -> None:
        """Called when ElevenLabs corrects a previously streamed response."""
        # For Phase 0, we log this but don't surface it to the archive.
        logger.debug("Response correction: %r → %r", original, corrected)

    def _handle_latency(self, latency_ms: int) -> None:
        """Called by ElevenLabs SDK with round-trip latency in milliseconds."""
        logger.info("ElevenLabs latency: %d ms", latency_ms)

    def _end_session_safely(self) -> None:
        """Attempt a clean session end; swallow errors."""
        if self._conversation is not None:
            try:
                self._conversation.end_session()
            except Exception:  # noqa: BLE001
                pass
            self._conversation = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _noop_str(text: str) -> None:  # noqa: ARG001
    """No-op callback for when no handler is provided."""
    pass

