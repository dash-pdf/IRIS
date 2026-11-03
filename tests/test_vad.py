"""
tests/test_vad.py
-----------------
Unit tests for iris/audio/vad.py (Voice Activity Detection & Utterance Segmentation).
"""

from __future__ import annotations

import struct
import pytest

from iris.audio.vad import VoiceActivityDetector, calculate_rms


def _generate_frame(amplitude: int, num_samples: int = 800) -> bytes:
    """Generate 16-bit PCM mono frame with given constant amplitude."""
    return struct.pack(f"{num_samples}h", *([amplitude] * num_samples))


class TestVoiceActivityDetector:
    def test_calculate_rms(self) -> None:
        frame_silent = _generate_frame(0)
        assert calculate_rms(frame_silent) == 0.0

        frame_loud = _generate_frame(1000)
        assert abs(calculate_rms(frame_loud) - 1000.0) < 1.0

    def test_speech_onset_and_silence_timeout(self) -> None:
        start_events = []
        end_events = []

        vad = VoiceActivityDetector(
            sample_rate=16000,
            energy_threshold=400.0,
            silence_timeout_ms=100,  # 100ms silence timeout
            min_speech_ms=50,
            on_speech_start=lambda: start_events.append(True),
            on_speech_end=lambda data: end_events.append(data),
        )

        # 1. Feed 100ms of silence (2 x 50ms frames)
        frame_silent = _generate_frame(50, 800)
        assert vad.process_frame(frame_silent) is None
        assert vad.process_frame(frame_silent) is None
        assert vad.is_speaking is False
        assert len(start_events) == 0

        # 2. Feed 150ms of speech (3 x 50ms frames)
        frame_voice = _generate_frame(1200, 800)
        assert vad.process_frame(frame_voice) is None
        assert vad.is_speaking is True
        assert len(start_events) == 1

        assert vad.process_frame(frame_voice) is None
        assert vad.process_frame(frame_voice) is None

        # 3. Feed silence until timeout (2 x 50ms = 100ms silence)
        assert vad.process_frame(frame_silent) is None
        utterance = vad.process_frame(frame_silent)

        # Should return complete segmented utterance
        assert utterance is not None
        assert len(end_events) == 1
        assert vad.is_speaking is False

    def test_discard_short_glitch_below_min_speech(self) -> None:
        vad = VoiceActivityDetector(
            energy_threshold=400.0,
            silence_timeout_ms=100,
            min_speech_ms=200,  # Min 200ms
        )

        # 50ms voice pop
        frame_voice = _generate_frame(1500, 800)
        frame_silent = _generate_frame(0, 800)

        vad.process_frame(frame_voice)
        vad.process_frame(frame_silent)
        utterance = vad.process_frame(frame_silent)

        # Discarded because total speech (50ms) < min_speech_ms (200ms)
        assert utterance is None
        assert vad.is_speaking is False

