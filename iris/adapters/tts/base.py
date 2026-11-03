"""
iris/adapters/tts/base.py
-------------------------
Abstract interface for Text-To-Speech (TTS) Adapters.

Decouples speech synthesis from intelligence runtime, agent frameworks,
and specific vendors.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np


def shape_speech_text(text: str) -> str:
    """
    Clean model-generated text before speech synthesis (TTS).

    Removes markdown formatting, code fences, links, and raw URLs while
    preserving spoken words, conversational punctuation, and numbers.

    Args:
        text: Raw model response text.

    Returns:
        Spoken-ready text for TTS.
    """
    if not text or not text.strip():
        return ""

    s = text

    # Remove code fence delimiters (```python or ```)
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"```", "", s)

    # Convert markdown links [text](url) -> text
    # Strip image links ![alt](url) -> ""
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # Strip raw URLs, preserving trailing punctuation if attached
    s = re.sub(r"https?://\S+?([.,;:!?])(?=\s|$)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)

    # Strip markdown headings (# Title, ## Subtitle, etc.)
    s = re.sub(r"^\s*#{1,6}\s+", "", s, flags=re.MULTILINE)

    # Strip list bullets (-, *, +, •) and blockquotes (>) at start of line
    s = re.sub(r"^\s*[-*+•]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*>\s+", "", s, flags=re.MULTILINE)

    # Inline code: `code` -> code
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"`", "", s)

    # Strikethrough: ~~text~~ -> text
    s = re.sub(r"~~([^~]+)~~", r"\1", s)

    # Bold and bold-italics: ***text***, **text** -> text
    s = re.sub(r"\*{2,3}([^*]+)\*{2,3}", r"\1", s)
    # Italics: *text* -> text
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"\*", "", s)

    # Underscores for emphasis: only match when bounded by non-word chars or start/end
    # (preserve snake_case identifiers)
    s = re.sub(r"(^|[\s(])_{1,3}([^_]+)_{1,3}(?=[\s).,;:!?]|$)", r"\1\2", s)

    # Remove spaces preceding punctuation marks
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)

    # Normalize whitespace: newlines and multiple spaces into single space
    s = re.sub(r"\s+", " ", s).strip()

    return s


class TTSAdapter(ABC):
    """
    Abstract Text-to-Speech Adapter interface.
    """

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize text into raw audio samples.

        Args:
            text: Text to synthesize.

        Returns:
            Tuple of (samples, sample_rate), where samples is a 1D numpy array
            (int16 or float32).
        """
        pass

    @property
    @abstractmethod
    def voice_name(self) -> str:
        """Name or identifier of the active voice."""
        pass

