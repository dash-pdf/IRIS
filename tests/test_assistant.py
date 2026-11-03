"""
tests/test_assistant.py
-----------------------
Integration tests for iris/core/assistant.py (Decoupled IRIS Orchestrator).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iris.adapters.model.mock_adapter import MockModelAdapter
from iris.adapters.stt.mock_stt import MockSTT
from iris.adapters.tts.mock_tts import MockTTS
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig
from iris.core.state import AudioState


@pytest.fixture
def temp_archive_dir(tmp_path: Path) -> Path:
    return tmp_path / "conversations"


@pytest.fixture
def mock_assistant(temp_archive_dir: Path) -> IRISAssistant:
    cfg = IRISConfig(
        llm_provider="mock",
        tts_provider="mock",
        stt_provider="mock",
        archive_dir=temp_archive_dir,
        cooldown_ms=50,  # Fast cooldown for testing
    )
    model = MockModelAdapter(
        responses=["I am IRIS, all systems nominal.", "Understood, proceeding now."]
    )
    tts = MockTTS()
    stt = MockSTT()

    assistant = IRISAssistant(
        config=cfg,
        model_adapter=model,
        tts_adapter=tts,
        stt_adapter=stt,
    )
    return assistant


class TestIRISAssistant:
    def test_start_and_stop_lifecycle(self, mock_assistant: IRISAssistant) -> None:
        assert mock_assistant.is_running is False
        assert mock_assistant.current_state == AudioState.IDLE

        mock_assistant.start()
        assert mock_assistant.is_running is True
        assert mock_assistant.capture.is_running is True
        assert mock_assistant.capture.is_muted is False

        mock_assistant.stop()
        assert mock_assistant.is_running is False
        assert mock_assistant.capture.is_running is False

    def test_text_turn_execution_and_archiving(
        self, mock_assistant: IRISAssistant, temp_archive_dir: Path
    ) -> None:
        mock_assistant.start()

        resp = mock_assistant.process_text_turn("Status check please.")
        assert resp == "I am IRIS, all systems nominal."

        # Wait for brief cooldown to return to IDLE
        time.sleep(0.1)
        assert mock_assistant.current_state == AudioState.IDLE

        mock_assistant.stop()

        # Verify Markdown file was written to disk
        md_files = list(temp_archive_dir.glob("**/*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "Status check please." in content
        assert "I am IRIS, all systems nominal." in content

    def test_echo_rejection_prevents_loopback(self, mock_assistant: IRISAssistant) -> None:
        echo_events = []
        mock_assistant.on_echo_detected = lambda text, reason: echo_events.append((text, reason))

        mock_assistant.start()

        # Turn 1: Assistant speaks
        resp1 = mock_assistant.process_text_turn("What is your designation?")
        assert resp1 == "I am IRIS, all systems nominal."
        time.sleep(0.1)

        # Now simulate microphone hearing the speaker output
        echo_transcript = "I am IRIS all systems nominal"
        stt: MockSTT = mock_assistant.stt  # type: ignore
        stt.enqueue_transcript(echo_transcript)

        # Process the simulated utterance
        mock_assistant._process_utterance(b"\x00" * 1600, turn_id="test-turn-echo")

        # Echo filter must have detected and dropped the echo
        assert len(echo_events) == 1
        assert echo_events[0][0] == echo_transcript
        assert "match" in echo_events[0][1] or "fuzzy" in echo_events[0][1]

        # Crucial check: History must NOT have the echoed utterance as a user turn
        history = mock_assistant.conversation.get_messages()
        # Should only have system prompt, original user turn, original assistant turn
        user_turns = [m["content"] for m in history if m["role"] == "user"]
        assert len(user_turns) == 1
        assert user_turns[0] == "What is your designation?"
        assert echo_transcript not in user_turns

        mock_assistant.stop()

    def test_multi_turn_dialogue_continuity(self, mock_assistant: IRISAssistant) -> None:
        mock_assistant.start()

        resp1 = mock_assistant.process_text_turn("First question.")
        assert resp1 == "I am IRIS, all systems nominal."
        time.sleep(0.1)

        resp2 = mock_assistant.process_text_turn("Second question.")
        assert resp2 == "Understood, proceeding now."
        time.sleep(0.1)

        history = mock_assistant.conversation.get_messages()
        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        asst_msgs = [m["content"] for m in history if m["role"] == "assistant"]

        assert user_msgs == ["First question.", "Second question."]
        assert asst_msgs == ["I am IRIS, all systems nominal.", "Understood, proceeding now."]

        mock_assistant.stop()

    def test_text_turn_timing_measurements_recorded(self, mock_assistant: IRISAssistant) -> None:
        mock_assistant.start()
        resp = mock_assistant.process_text_turn("Give me a timing test.")
        assert resp == "I am IRIS, all systems nominal."
        time.sleep(0.1)

        timings = mock_assistant.last_turn_timings
        assert timings is not None
        assert "llm_duration_s" in timings
        assert "tts_duration_s" in timings
        assert "total_latency_s" in timings
        assert timings["llm_streaming"] is False
        assert timings["llm_duration_s"] >= 0.0
        assert timings["tts_duration_s"] >= 0.0
        assert timings["total_latency_s"] >= 0.0

        mock_assistant.stop()

    def test_audio_utterance_timing_measurements(self, mock_assistant: IRISAssistant) -> None:
        mock_assistant.start()
        stt: MockSTT = mock_assistant.stt  # type: ignore
        stt.enqueue_transcript("Audio timing test.")

        # Simulate 1.0 second of 16kHz 16-bit mono audio (32,000 bytes)
        one_sec_audio = b"\x00" * 32000
        speech_end_time = time.perf_counter()
        turn_id = mock_assistant.state_machine.new_turn()
        mock_assistant.state_machine.transition_to(AudioState.THINKING, expected_turn_id=turn_id)
        mock_assistant._process_utterance(one_sec_audio, turn_id=turn_id, speech_end_time=speech_end_time)
        time.sleep(0.1)

        timings = mock_assistant.last_turn_timings
        assert timings["turn_id"] == turn_id
        assert abs(timings["capture_duration_s"] - 1.0) < 0.05
        assert timings["stt_duration_s"] >= 0.0
        assert timings["llm_duration_s"] >= 0.0
        assert timings["tts_duration_s"] >= 0.0
        assert timings["total_latency_s"] >= 0.0
        assert timings["llm_streaming"] is False

        mock_assistant.stop()

    def test_speech_text_shaping_and_archive_preservation(
        self, temp_archive_dir: Path
    ) -> None:
        cfg = IRISConfig(
            llm_provider="mock",
            tts_provider="mock",
            stt_provider="mock",
            archive_dir=temp_archive_dir,
            cooldown_ms=50,
        )
        model = MockModelAdapter(
            responses=[
                "Here is the status: **Online** at 100%. Visit https://example.com for info."
            ]
        )
        tts = MockTTS()
        stt = MockSTT()

        assistant = IRISAssistant(
            config=cfg,
            model_adapter=model,
            tts_adapter=tts,
            stt_adapter=stt,
        )
        assistant.start()

        resp = assistant.process_text_turn("Status please.")
        assert resp == "Here is the status: **Online** at 100%. Visit https://example.com for info."

        # Verify TTS received the shaped speech text
        assert tts.last_synthesized_text == "Here is the status: Online at 100%. Visit for info."

        # Verify echo filter registered the shaped speech text
        assert assistant.echo_filter._last_assistant_text == "Here is the status: Online at 100%. Visit for info."

        # Verify Markdown archive retained unshaped response text
        md_files = list(temp_archive_dir.glob("**/*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "**Online**" in content
        assert "https://example.com" in content

        assistant.stop()



