"""
iris/audio/vad.py
-----------------
Voice Activity Detector (VAD) and Utterance Segmenter.

Analyzes raw 16kHz 16-bit PCM mono audio stream to identify:
1. When user begins speaking (voice activity onset).
2. When user stops speaking (end-of-speech silence timeout).
3. Segments audio into clean, bounded utterance byte chunks.
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def calculate_rms(frame_bytes: bytes) -> float:
    """Calculate Root Mean Square (RMS) energy level of 16-bit PCM audio."""
    count = len(frame_bytes) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"{count}h", frame_bytes)
    sum_squares = sum(s * s for s in shorts)
    return math.sqrt(sum_squares / count)


class VoiceActivityDetector:
    """
    Energy-based Voice Activity Detector with silence timeout.

    Configurable parameters:
    - energy_threshold: Minimum RMS to classify as voice (default: 450).
    - silence_timeout_ms: Silence duration before marking end-of-speech (default: 750ms).
    - min_speech_ms: Discard clicks/pops shorter than this (default: 250ms).
    - max_speech_ms: Hard cutoff if user speaks continuously (default: 15000ms).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 450.0,
        silence_timeout_ms: int = 750,
        min_speech_ms: int = 250,
        max_speech_ms: int = 15000,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[bytes], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms

        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._is_speaking = False
        self._speech_buffer: bytearray = bytearray()
        self._silence_duration_ms: float = 0.0
        self._speech_duration_ms: float = 0.0

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        """Clear all buffers and reset state."""
        self._is_speaking = False
        self._speech_buffer.clear()
        self._silence_duration_ms = 0.0
        self._speech_duration_ms = 0.0

    def process_frame(self, frame_bytes: bytes) -> bytes | None:
        """
        Process a chunk of 16-bit PCM mono audio.

        Returns:
            bytes of complete utterance if speech just concluded, or None.
        """
        if not frame_bytes:
            return None

        bytes_per_sample = 2  # 16-bit
        frame_ms = (len(frame_bytes) / (bytes_per_sample * self.sample_rate)) * 1000.0

        rms = calculate_rms(frame_bytes)
        is_speech_frame = rms >= self.energy_threshold

        if not self._is_speaking:
            if is_speech_frame:
                self._is_speaking = True
                self._speech_duration_ms = frame_ms
                self._silence_duration_ms = 0.0
                self._speech_buffer.extend(frame_bytes)
                if self.on_speech_start:
                    self.on_speech_start()
            return None

        # Already in speaking state
        self._speech_buffer.extend(frame_bytes)
        self._speech_duration_ms += frame_ms

        if is_speech_frame:
            self._silence_duration_ms = 0.0
        else:
            self._silence_duration_ms += frame_ms

        # Check termination conditions:
        # 1. Silence exceeded timeout, OR 2. Max utterance length exceeded
        timeout_reached = self._silence_duration_ms >= self.silence_timeout_ms
        max_reached = self._speech_duration_ms >= self.max_speech_ms

        if timeout_reached or max_reached:
            utterance_data = bytes(self._speech_buffer)
            total_duration_ms = self._speech_duration_ms
            self.reset()

            if total_duration_ms >= self.min_speech_ms:
                if self.on_speech_end:
                    self.on_speech_end(utterance_data)
                return utterance_data
            else:
                logger.debug("Utterance too short (%d ms < %d ms), discarded", total_duration_ms, self.min_speech_ms)
                return None

        return None

