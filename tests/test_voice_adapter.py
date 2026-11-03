"""
tests/test_voice_adapter.py
---------------------------
Unit tests for iris/adapters/voice_elevenlabs.py.

Tests that the adapter and its SoundDeviceAudioInterface can be
instantiated, have correct method signatures, and handle interrupts
and callbacks cleanly.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from iris.adapters.voice_elevenlabs import (
    ElevenLabsVoiceAdapter,
    SoundDeviceAudioInterface,
)


class TestSoundDeviceAudioInterface:
    def test_instantiation(self) -> None:
        """SoundDeviceAudioInterface can be instantiated."""
        audio = SoundDeviceAudioInterface()
        assert audio.INPUT_FRAMES_PER_BUFFER == 4000
        assert audio.OUTPUT_FRAMES_PER_BUFFER == 1000
        assert audio.SAMPLE_RATE == 16000

    def test_output_queues_audio(self) -> None:
        """Calling output() puts bytes into the internal queue."""
        audio = SoundDeviceAudioInterface()
        chunk = b"\x00\x01" * 100
        audio.output(chunk)
        assert audio._output_queue.qsize() == 1
        assert audio._output_queue.get() == chunk

    def test_interrupt_clears_queue(self) -> None:
        """Calling interrupt() empties the output queue immediately."""
        audio = SoundDeviceAudioInterface()
        audio.output(b"chunk1")
        audio.output(b"chunk2")
        audio.output(b"chunk3")
        assert audio._output_queue.qsize() == 3

        audio.interrupt()
        assert audio._output_queue.qsize() == 0

    def test_input_callback_triggers(self) -> None:
        """When input data arrives, the registered callback is invoked with bytes."""
        audio = SoundDeviceAudioInterface()
        received = []
        audio._input_callback = lambda b: received.append(b)

        sample_bytes = b"\x12\x34\x56\x78"
        audio._in_callback(sample_bytes, 2, None, None)

        assert len(received) == 1
        assert received[0] == sample_bytes


class TestElevenLabsVoiceAdapterCallbacks:
    def test_adapter_instantiation(self) -> None:
        """ElevenLabsVoiceAdapter initializes with callbacks."""
        user_cb = MagicMock()
        asst_cb = MagicMock()
        err_cb = MagicMock()

        adapter = ElevenLabsVoiceAdapter(
            api_key="test-key",
            agent_id="test-agent",
            on_user_transcript=user_cb,
            on_assistant_response=asst_cb,
            on_error=err_cb,
        )
        assert adapter._api_key == "test-key"
        assert adapter._agent_id == "test-agent"

    def test_user_transcript_callback(self) -> None:
        """_handle_user_transcript strips text and calls user callback."""
        user_cb = MagicMock()
        adapter = ElevenLabsVoiceAdapter(
            api_key="test-key",
            agent_id="test-agent",
            on_user_transcript=user_cb,
        )
        adapter._handle_user_transcript("  Hello world!  ")
        user_cb.assert_called_once_with("Hello world!")

    def test_empty_user_transcript_ignored(self) -> None:
        """Whitespace-only transcripts are not forwarded."""
        user_cb = MagicMock()
        adapter = ElevenLabsVoiceAdapter(
            api_key="test-key",
            agent_id="test-agent",
            on_user_transcript=user_cb,
        )
        adapter._handle_user_transcript("   ")
        user_cb.assert_not_called()

    def test_assistant_response_callback(self) -> None:
        """_handle_assistant_response calls assistant callback."""
        asst_cb = MagicMock()
        adapter = ElevenLabsVoiceAdapter(
            api_key="test-key",
            agent_id="test-agent",
            on_assistant_response=asst_cb,
        )
        adapter._handle_assistant_response("I hear you loud and clear.")
        asst_cb.assert_called_once_with("I hear you loud and clear.")

    def test_stop_when_no_session_is_safe(self) -> None:
        """stop() when not running is safe and does not crash."""
        adapter = ElevenLabsVoiceAdapter(
            api_key="test-key",
            agent_id="test-agent",
        )
        adapter.stop()  # Should not raise

