"""
iris/adapters/stt/base.py
-------------------------
Abstract interface for Speech-to-Text (STT) Adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class STTAdapter(ABC):
    """
    Abstract Speech-to-Text Adapter interface.
    """

    @abstractmethod
    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe raw PCM audio bytes to text.

        Args:
            pcm_bytes: Raw 16-bit signed PCM mono audio bytes.
            sample_rate: Audio sampling frequency (default: 16000).

        Returns:
            Transcribed text string.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the STT provider."""
        pass

