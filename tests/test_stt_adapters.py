"""
tests/test_stt_adapters.py
--------------------------
Unit tests for iris/adapters/stt/ (Speech-To-Text Adapters).
"""

from __future__ import annotations

import pytest

from iris.adapters.stt import (
    ConsoleSTT,
    ElevenLabsSTT,
    MockSTT,
    create_stt_adapter,
)
from iris.core.config import IRISConfig


class TestSTTAdapters:
    def test_mock_stt_queue(self) -> None:
        stt = MockSTT(default_response="Default utterance")
        assert stt.provider_name == "MockSTT"

        stt.enqueue_transcript("First message")
        stt.enqueue_transcript("Second message")

        assert stt.transcribe(b"dummy") == "First message"
        assert stt.transcribe(b"dummy") == "Second message"
        assert stt.transcribe(b"dummy") == "Default utterance"
        assert stt.call_count == 3

    def test_console_stt_initialization(self) -> None:
        stt = ConsoleSTT(prompt="Test: ")
        assert "Console" in stt.provider_name

    def test_elevenlabs_stt_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            ElevenLabsSTT(api_key="")

    def test_elevenlabs_stt_empty_bytes(self) -> None:
        stt = ElevenLabsSTT(api_key="test-key")
        assert stt.transcribe(b"") == ""
        assert stt.transcribe(b"\x00" * 10) == ""

    def test_create_stt_adapter_factory(self) -> None:
        cfg_mock = IRISConfig(stt_provider="mock")
        stt_mock = create_stt_adapter(cfg_mock)
        assert isinstance(stt_mock, MockSTT)

        cfg_console = IRISConfig(stt_provider="console")
        stt_console = create_stt_adapter(cfg_console)
        assert isinstance(stt_console, ConsoleSTT)

    def test_elevenlabs_stt_transcribe_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Testing speech recognition"
        mock_client.speech_to_text.convert.return_value = mock_result

        stt = ElevenLabsSTT(api_key="valid-test-key")
        stt._client = mock_client

        # 1600 bytes = 800 samples of 16-bit PCM = 50ms at 16kHz
        pcm_bytes = b"\x01\x00" * 800
        transcript = stt.transcribe(pcm_bytes, sample_rate=16000)

        assert transcript == "Testing speech recognition"
        assert mock_client.speech_to_text.convert.call_count == 1

