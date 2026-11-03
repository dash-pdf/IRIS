"""
iris/adapters/tts/mock_tts.py
-----------------------------
Mock Text-to-Speech Adapter for headless unit and integration testing.
"""

from __future__ import annotations

import numpy as np

from iris.adapters.tts.base import TTSAdapter


class MockTTS(TTSAdapter):
    """
    Produces deterministic synthetic audio for tests without external calls.
    """

    def __init__(self, sample_rate: int = 16000, tone_freq: float = 440.0) -> None:
        self._sample_rate = sample_rate
        self._tone_freq = tone_freq
        self.last_synthesized_text: str = ""

    @property
    def voice_name(self) -> str:
        return "MockTTS"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.last_synthesized_text = text.strip()
        if not self.last_synthesized_text:
            return np.zeros(0, dtype=np.int16), self._sample_rate

        # Duration proportional to word count (~100ms per word, minimum 200ms)
        word_count = max(1, len(text.split()))
        duration_s = min(2.0, max(0.2, word_count * 0.1))
        num_samples = int(self._sample_rate * duration_s)

        # Generate a gentle low-amplitude tone
        t = np.linspace(0, duration_s, num_samples, False)
        tone = (np.sin(2 * np.pi * self._tone_freq * t) * 1000).astype(np.int16)
        return tone, self._sample_rate

