"""
iris/adapters/stt/mock_stt.py
-----------------------------
Mock Speech-to-Text Adapter for deterministic unit and integration tests.
"""

from __future__ import annotations

import collections
from iris.adapters.stt.base import STTAdapter


class MockSTT(STTAdapter):
    """
    Mock STT adapter that returns pre-queued transcripts.
    """

    def __init__(self, default_response: str = "Hello IRIS") -> None:
        self._queue: collections.deque[str] = collections.deque()
        self._default_response = default_response
        self.call_count: int = 0

    @property
    def provider_name(self) -> str:
        return "MockSTT"

    def enqueue_transcript(self, text: str) -> None:
        """Enqueue a canned transcript for subsequent transcribe calls."""
        self._queue.append(text)

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        self.call_count += 1
        if self._queue:
            return self._queue.popleft()
        return self._default_response

