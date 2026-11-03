"""
iris/adapters/stt/console_stt.py
--------------------------------
Console-based Text Input Adapter.

Fallback when audio input is unavailable, muted, or for silent developer interaction.
"""

from __future__ import annotations

import sys
from iris.adapters.stt.base import STTAdapter


class ConsoleSTT(STTAdapter):
    """
    Console input adapter allowing keyboard typing in place of speech.
    """

    def __init__(self, prompt: str = "You: ") -> None:
        self._prompt = prompt

    @property
    def provider_name(self) -> str:
        return "Console (Keyboard)"

    def transcribe(self, pcm_bytes: bytes = b"", sample_rate: int = 16000) -> str:
        try:
            print()
            sys.stdout.write(self._prompt)
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return ""

