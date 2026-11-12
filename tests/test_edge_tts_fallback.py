"""
tests/test_edge_tts_fallback.py
--------------------------------
Comprehensive automated tests for Edge TTS upgrade, resilience, and native macOS fallback.

Verifies:
1. Successful Edge TTS generation.
2. Automatic fallback to macOS TTS on 403 or network failure.
3. Clean utterance isolation (no partial chunks mixed or played).
4. Edge TTS retried on subsequent turns (provider recovery).
5. State machine reset to IDLE if both providers fail.
6. Microphone unmuted and VAD reset on total failure.
7. Consecutive failure counting and recovery.
8. Five sequential synthesis requests without state degradation.
9. Token and query parameter redaction in error logging.
10. macOS voice resolution and British English voice selection.
"""

from __future__ import annotations

import asyncio
import io
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from iris.adapters.tts.base import TTSAdapter
from iris.adapters.tts.edge_tts import EdgeTTS
from iris.adapters.tts.macos_tts import MacOSSayTTS, MacOSTTS, resolve_macos_voice
from iris.adapters.tts.mock_tts import MockTTS
from iris.audio.capture import AudioCapture
from iris.audio.player import AudioPlayer
from iris.audio.vad import VoiceActivityDetector
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig
from iris.core.state import AudioState, AudioStateMachine


class DummyFallbackTTS(TTSAdapter):
    """Predictable fallback TTS adapter for deterministic unit testing."""

    def __init__(self, voice_name: str = "Flo (English (UK))", sample_rate: int = 22050) -> None:
        self._voice_name = voice_name
        self.sample_rate = sample_rate
        self.synthesize_call_count = 0
        self.last_synthesized_text = ""
        self.should_fail = False

    @property
    def voice_name(self) -> str:
        return self._voice_name

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.synthesize_call_count += 1
        self.last_synthesized_text = text
        if self.should_fail:
            raise RuntimeError("Dummy fallback synthesis failed")
        samples = np.ones(int(self.sample_rate * 0.5), dtype=np.float32) * 0.25
        return samples, self.sample_rate


def _generate_valid_wav_bytes(duration_s: float = 0.5, sample_rate: int = 24000) -> bytes:
    """Helper to generate valid WAV bytes in memory."""
    buf = io.BytesIO()
    samples = np.ones(int(sample_rate * duration_s), dtype=np.float32) * 0.1
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


class TestEdgeTTSFallback:
    def test_successful_edge_tts_generation(self) -> None:
        """Scenario 1: Successful Edge TTS generation with reset failure count."""
        dummy_fallback = DummyFallbackTTS()
        tts = EdgeTTS(fallback_tts=dummy_fallback)

        valid_wav = _generate_valid_wav_bytes(0.5, 24000)

        async def mock_stream(*args, **kwargs):
            yield {"type": "audio", "data": valid_wav}

        with patch("edge_tts.Communicate.stream", side_effect=mock_stream):
            samples, sr = tts.generate_audio("IRIS systems online.")
            assert len(samples) > 0
            assert sr == 24000
            assert tts.consecutive_failures == 0
            assert tts.using_fallback is False
            assert dummy_fallback.synthesize_call_count == 0

    def test_edge_tts_403_triggers_macos_fallback(self) -> None:
        """Scenario 2: Automatic fallback to macOS TTS when Edge TTS raises 403."""
        dummy_fallback = DummyFallbackTTS(voice_name="Flo (English (UK))")
        tts = EdgeTTS(fallback_tts=dummy_fallback)

        async def mock_stream_403(*args, **kwargs):
            raise RuntimeError("WSServerHandshakeError: 403, message='Invalid response status'")
            yield {"type": "audio", "data": b""}

        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_403):
            samples, sr = tts.generate_audio("Fallback test.")
            assert len(samples) > 0
            assert sr == 22050  # Fallback sample rate
            assert tts.consecutive_failures == 1
            assert tts.using_fallback is True
            assert dummy_fallback.synthesize_call_count == 1
            assert dummy_fallback.last_synthesized_text == "Fallback test."

    def test_edge_tts_network_error_triggers_fallback(self) -> None:
        """Scenario 2b: Fallback on connection/network drop or timeout."""
        dummy_fallback = DummyFallbackTTS()
        tts = EdgeTTS(fallback_tts=dummy_fallback)

        async def mock_stream_network_error(*args, **kwargs):
            raise ConnectionResetError("Connection closed by peer")
            yield {"type": "audio", "data": b""}

        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_network_error):
            samples, sr = tts.generate_audio("Network error test.")
            assert len(samples) > 0
            assert tts.consecutive_failures == 1
            assert tts.using_fallback is True
            assert dummy_fallback.synthesize_call_count == 1

    def test_complete_utterance_resynthesis_no_partial_mixed(self) -> None:
        """Scenario 3: Discard incomplete buffer if Edge TTS fails midway; full utterance via fallback."""
        dummy_fallback = DummyFallbackTTS()
        mock_player = MagicMock(spec=AudioPlayer)
        tts = EdgeTTS(player=mock_player, fallback_tts=dummy_fallback)

        # Yield one chunk, then raise an exception midway
        async def mock_stream_fail_midway(*args, **kwargs):
            yield {"type": "audio", "data": b"PARTIAL_BROKEN_BYTES"}
            raise RuntimeError("Connection dropped midway through synthesis")

        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_fail_midway):
            tts.speak("Important instruction for user.", blocking=False)

            # Assert fallback synthesized the COMPLETE text
            assert dummy_fallback.synthesize_call_count == 1
            assert dummy_fallback.last_synthesized_text == "Important instruction for user."

            # Assert mock player received the fallback samples, never partial bytes
            assert mock_player.play.called
            played_samples, played_sr = mock_player.play.call_args[0][0], mock_player.play.call_args[1]["sample_rate"]
            assert played_sr == 22050
            assert len(played_samples) == int(22050 * 0.5)

    def test_edge_tts_retried_on_subsequent_turn(self) -> None:
        """Scenario 4: Edge TTS is attempted first on each turn (provider recovery)."""
        dummy_fallback = DummyFallbackTTS()
        tts = EdgeTTS(fallback_tts=dummy_fallback)

        valid_wav = _generate_valid_wav_bytes(0.5, 24000)

        # Turn 1: Edge TTS fails -> Fallback used
        async def mock_stream_fail(*args, **kwargs):
            raise RuntimeError("403 Forbidden")
            yield {"type": "audio", "data": b""}

        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_fail):
            tts.generate_audio("Turn 1 text")
            assert tts.consecutive_failures == 1
            assert tts.using_fallback is True
            assert dummy_fallback.synthesize_call_count == 1

        # Turn 2: Edge TTS recovers and succeeds
        async def mock_stream_ok(*args, **kwargs):
            yield {"type": "audio", "data": valid_wav}

        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_ok):
            tts.generate_audio("Turn 2 text")
            assert tts.consecutive_failures == 0
            assert tts.using_fallback is False
            # Fallback was NOT called on turn 2
            assert dummy_fallback.synthesize_call_count == 1

        # Turn 3: Edge TTS succeeds again
        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_ok):
            tts.generate_audio("Turn 3 text")
            assert tts.consecutive_failures == 0
            assert tts.using_fallback is False
            assert dummy_fallback.synthesize_call_count == 1

    def test_consecutive_failure_counting_and_recovery(self) -> None:
        """Scenario 7: Consecutive failure counter tracks repeated failures and resets upon recovery."""
        dummy_fallback = DummyFallbackTTS()
        tts = EdgeTTS(fallback_tts=dummy_fallback)
        valid_wav = _generate_valid_wav_bytes(0.5, 24000)

        async def mock_stream_fail(*args, **kwargs):
            raise RuntimeError("503 Service Unavailable")
            yield {"type": "audio", "data": b""}

        async def mock_stream_ok(*args, **kwargs):
            yield {"type": "audio", "data": valid_wav}

        # 3 failures in a row
        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_fail):
            for i in range(1, 4):
                tts.generate_audio(f"Failure iteration {i}")
                assert tts.consecutive_failures == i
                assert tts.using_fallback is True

        assert dummy_fallback.synthesize_call_count == 3

        # Recovery on 4th call
        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_ok):
            tts.generate_audio("Recovery iteration")
            assert tts.consecutive_failures == 0
            assert tts.using_fallback is False
            # Fallback was not called on recovery
            assert dummy_fallback.synthesize_call_count == 3

    def test_both_providers_fail_raises_runtime_error(self) -> None:
        """Scenario 5a: If both Edge TTS and fallback fail, raises RuntimeError."""
        dummy_fallback = DummyFallbackTTS()
        dummy_fallback.should_fail = True
        tts = EdgeTTS(fallback_tts=dummy_fallback)

        async def mock_stream_fail(*args, **kwargs):
            raise RuntimeError("403 Forbidden")
            yield {"type": "audio", "data": b""}

        with patch("edge_tts.Communicate.stream", side_effect=mock_stream_fail):
            with pytest.raises(RuntimeError, match="Both Edge TTS and fallback TTS failed"):
                tts.generate_audio("Doomed utterance")

            assert tts.consecutive_failures == 1

    def test_state_machine_reset_to_idle_on_total_tts_failure(self) -> None:
        """Scenario 5b & 6: On total TTS failure, state machine resets to IDLE, mic unmuted, VAD reset."""
        cfg = IRISConfig(
            tts_provider="edge",
            cooldown_ms=10,
        )

        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_muted = True
        mock_player = MagicMock(spec=AudioPlayer)
        mock_vad = MagicMock(spec=VoiceActivityDetector)

        # Create assistant with a failing TTS
        failing_tts = EdgeTTS(
            fallback_tts=DummyFallbackTTS(),
            player=mock_player,
        )
        # Force generate_audio to raise RuntimeError
        failing_tts.generate_audio = MagicMock(side_effect=RuntimeError("Both providers failed completely"))

        from iris.adapters.model.mock_adapter import MockModelAdapter
        from iris.adapters.stt.mock_stt import MockSTT

        assistant = IRISAssistant(
            config=cfg,
            model_adapter=MockModelAdapter(responses=["Understood."]),
            tts_adapter=failing_tts,
            stt_adapter=MockSTT(default_response="Hello assistant"),
        )
        assistant.capture = mock_capture
        assistant.player = mock_player
        assistant.vad = mock_vad

        # Trigger turn processing
        turn_id = assistant.state_machine.new_turn()
        assistant.state_machine.transition_to(AudioState.THINKING)

        assistant._process_utterance(
            utterance_bytes=b"\x00" * 3200,
            turn_id=turn_id,
            speech_end_time=time.perf_counter(),
        )

        # Verification:
        # 1. State machine reset to IDLE (never stuck in SPEAKING or ERROR)
        assert assistant.state_machine.state == AudioState.IDLE
        # 2. Microphone unmuted
        assert mock_capture.unmute.called
        # 3. VAD reset
        assert mock_vad.reset.called

    def test_five_sequential_synthesis_requests(self) -> None:
        """Scenario 8: Five sequential synthesis requests without state degradation."""
        dummy_fallback = DummyFallbackTTS()
        tts = EdgeTTS(fallback_tts=dummy_fallback)
        valid_wav = _generate_valid_wav_bytes(0.3, 24000)

        async def mock_stream(*args, **kwargs):
            yield {"type": "audio", "data": valid_wav}

        with patch("edge_tts.Communicate.stream", side_effect=mock_stream):
            for i in range(1, 6):
                samples, sr = tts.generate_audio(f"Sequential request number {i}.")
                assert len(samples) > 0
                assert sr == 24000
                assert tts.consecutive_failures == 0
                assert tts.using_fallback is False

        assert dummy_fallback.synthesize_call_count == 0

    def test_secure_logging_sanitizes_tokens_and_urls(self) -> None:
        """Scenario 9: Tokens, query params, and credentials are redacted from logs."""
        tts = EdgeTTS()
        dirty_error = (
            "403 Forbidden connecting to "
            "wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1"
            "?Sec-MS-GEC=ABC123XYZ&Sec-MS-GEC-Version=1-130.0.2849.68&token=super_secret_auth"
        )
        sanitized = tts._sanitize_error(dirty_error)
        assert "ABC123XYZ" not in sanitized
        assert "super_secret_auth" not in sanitized
        assert "?[REDACTED]" in sanitized

    def test_macos_voice_resolution(self) -> None:
        """Scenario 10: macOS voice resolution prioritizes Flo and British English."""
        # 1. Exact Flo resolution
        mock_voices = {
            "Flo (English (UK))": "en_GB",
            "Daniel (English (UK))": "en_GB",
            "Samantha": "en_US",
        }
        with patch("iris.adapters.tts.macos_tts.get_installed_macos_voices", return_value=mock_voices):
            assert resolve_macos_voice("Flo (English (UK))") == "Flo (English (UK))"
            assert resolve_macos_voice("Flo") == "Flo"

            # 2. When preferred voice is missing, select Flo or British fallback
            fallback_choice = resolve_macos_voice("MissingVoice")
            assert "Flo" in fallback_choice or "Daniel" in fallback_choice

        # 3. MacOSTTS class alias check
        assert MacOSTTS is MacOSSayTTS
