"""
tests/test_session_restore.py
-----------------------------
Comprehensive tests for Phase 1 acceptance criterion 8:
Restoring recent conversation context from the Markdown archive on startup.

Test coverage:
- Successful restoration of recent turns.
- Correct chronological ordering.
- Empty archive.
- Malformed Markdown.
- Incomplete final turn.
- No duplicated turns after restart.
- Confirmation that the system prompt is retained.
- Read errors handled gracefully.
- Unrelated sessions not merged.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from iris.adapters.model.mock_adapter import MockModelAdapter
from iris.adapters.stt.mock_stt import MockSTT
from iris.adapters.tts.mock_tts import MockTTS
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig
from iris.services.archive import ConversationArchive, parse_session_markdown
from iris.services.conversation import ConversationService, IRIS_SYSTEM_PROMPT, Turn


@pytest.fixture
def temp_archive_dir(tmp_path: Path) -> Path:
    d = tmp_path / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestParseSessionMarkdown:
    def test_parse_valid_dialogue_turns(self) -> None:
        content = """# IRIS Conversation — 2026-10-11

**Session file:** `session-001.md`  
**Started:** 2026-10-11 12:00:00

---

### [2026-10-11 12:00:05] User
Hello IRIS, are you ready?

### [2026-10-11 12:00:07] IRIS
Always ready. How can I assist you today?

### [2026-10-11 12:00:15] User
Give me a quick status update.

### [2026-10-11 12:00:18] IRIS
All systems operating within normal parameters.

---

**Session ended:** 2026-10-11 12:01:00
"""
        turns = parse_session_markdown(content, persona_name="IRIS")
        assert len(turns) == 4
        assert turns[0] == Turn(role="user", content="Hello IRIS, are you ready?")
        assert turns[1] == Turn(role="assistant", content="Always ready. How can I assist you today?")
        assert turns[2] == Turn(role="user", content="Give me a quick status update.")
        assert turns[3] == Turn(role="assistant", content="All systems operating within normal parameters.")

    def test_metadata_headings_and_warnings_not_loaded(self) -> None:
        """Headings, timestamps, warnings (⚠️), and error entries must not become dialogue turns."""
        content = """# IRIS Conversation — 2026-10-11

**Session file:** `session-002.md`
**Started:** 2026-10-11 13:00:00

---

### [2026-10-11 13:00:05] User
Can you connect to the server?

### [2026-10-11 13:00:08] Error
⚠️ Connection timed out after 5000ms

### [2026-10-11 13:00:10] IRIS
I encountered a temporary connection issue, but I am back online now.

---

**Session ended:** 2026-10-11 13:01:00
"""
        turns = parse_session_markdown(content, persona_name="IRIS")
        assert len(turns) == 2
        assert turns[0] == Turn(role="user", content="Can you connect to the server?")
        assert turns[1] == Turn(
            role="assistant",
            content="I encountered a temporary connection issue, but I am back online now.",
        )
        # Ensure error warning didn't leak into turns
        for t in turns:
            assert "⚠️" not in t.content
            assert "timed out" not in t.content

    def test_incomplete_final_turn_dropped(self) -> None:
        """An unanswered user turn at the end of a session must be dropped."""
        content = """### [2026-10-11 14:00:01] User
Hello.

### [2026-10-11 14:00:03] IRIS
Hello there.

### [2026-10-11 14:00:10] User
What is the weather like right now?
"""
        turns = parse_session_markdown(content, persona_name="IRIS")
        assert len(turns) == 2
        assert turns[-1].role == "assistant"
        assert turns[-1].content == "Hello there."

    def test_single_unanswered_user_turn_returns_empty(self) -> None:
        """A session with only an unanswered user turn yields no complete turns."""
        content = """### [2026-10-11 14:00:01] User
Hello? Anyone there?
"""
        turns = parse_session_markdown(content, persona_name="IRIS")
        assert turns == []

    def test_malformed_markdown_handled_gracefully(self) -> None:
        """Malformed markdown with invalid headers or garbage content returns empty or valid turns."""
        malformed = """
This is random unstructured text.
No markdown turn headers anywhere.
---
**Some metadata**
### Broken Header Without Bracket
Random text here.
"""
        turns = parse_session_markdown(malformed, persona_name="IRIS")
        assert turns == []


class TestArchiveSessionRestoration:
    def test_empty_archive_returns_empty_list(self, temp_archive_dir: Path) -> None:
        """Empty archive directory returns empty list."""
        archive = ConversationArchive(archive_dir=temp_archive_dir)
        turns = archive.load_recent_turns()
        assert turns == []

    def test_nonexistent_archive_dir_returns_empty_list(self, tmp_path: Path) -> None:
        """Nonexistent archive directory returns empty list without error."""
        nonexistent = tmp_path / "does_not_exist"
        archive = ConversationArchive(archive_dir=nonexistent)
        turns = archive.load_recent_turns()
        assert turns == []

    def test_successful_restoration_from_previous_session(self, temp_archive_dir: Path) -> None:
        """Load turns from the most recent session file."""
        archive1 = ConversationArchive(archive_dir=temp_archive_dir)
        archive1.open_session()
        archive1.write_turn("user", "What is the date today?")
        archive1.write_turn("assistant", "It is the 11th of October.")
        archive1.close_session()

        archive2 = ConversationArchive(archive_dir=temp_archive_dir)
        turns = archive2.load_recent_turns(max_turns=10)
        assert len(turns) == 2
        assert turns[0] == Turn(role="user", content="What is the date today?")
        assert turns[1] == Turn(role="assistant", content="It is the 11th of October.")

    def test_correct_chronological_ordering_and_slicing(self, temp_archive_dir: Path) -> None:
        """Restore max 10 turns in strict chronological order from a larger session."""
        archive = ConversationArchive(archive_dir=temp_archive_dir)
        archive.open_session()
        # Write 8 complete exchanges = 16 turns
        for i in range(1, 9):
            archive.write_turn("user", f"Question {i}")
            archive.write_turn("assistant", f"Answer {i}")
        archive.close_session()

        # Request last 10 turns (exchanges 4 to 8: Question 4 .. Answer 8)
        turns = archive.load_recent_turns(max_turns=10)
        assert len(turns) == 10
        assert turns[0] == Turn(role="user", content="Question 4")
        assert turns[1] == Turn(role="assistant", content="Answer 4")
        assert turns[-2] == Turn(role="user", content="Question 8")
        assert turns[-1] == Turn(role="assistant", content="Answer 8")

    def test_scans_across_date_directories_for_latest_session(self, temp_archive_dir: Path) -> None:
        """Selects the latest session across multi-day folder hierarchy."""
        day1_dir = temp_archive_dir / "2026" / "10" / "09"
        day1_dir.mkdir(parents=True)
        (day1_dir / "session-001.md").write_text(
            "### [2026-10-09 10:00:00] User\nDay 9 turn\n\n### [2026-10-09 10:00:01] IRIS\nDay 9 reply\n",
            encoding="utf-8",
        )

        day2_dir = temp_archive_dir / "2026" / "10" / "11"
        day2_dir.mkdir(parents=True)
        (day2_dir / "session-001.md").write_text(
            "### [2026-10-11 10:00:00] User\nDay 11 turn\n\n### [2026-10-11 10:00:01] IRIS\nDay 11 reply\n",
            encoding="utf-8",
        )

        archive = ConversationArchive(archive_dir=temp_archive_dir)
        turns = archive.load_recent_turns(max_turns=10)
        assert len(turns) == 2
        assert turns[0].content == "Day 11 turn"
        assert turns[1].content == "Day 11 reply"

    def test_unrelated_sessions_not_concatenated(self, temp_archive_dir: Path) -> None:
        """History is restored ONLY from the single most recent relevant session."""
        day_dir = temp_archive_dir / "2026" / "10" / "11"
        day_dir.mkdir(parents=True)
        (day_dir / "session-001.md").write_text(
            "### [2026-10-11 09:00:00] User\nSession 1\n\n### [2026-10-11 09:00:01] IRIS\nSession 1 reply\n",
            encoding="utf-8",
        )
        (day_dir / "session-002.md").write_text(
            "### [2026-10-11 10:00:00] User\nSession 2\n\n### [2026-10-11 10:00:01] IRIS\nSession 2 reply\n",
            encoding="utf-8",
        )

        archive = ConversationArchive(archive_dir=temp_archive_dir)
        turns = archive.load_recent_turns(max_turns=10)
        assert len(turns) == 2
        assert turns[0].content == "Session 2"
        assert turns[1].content == "Session 2 reply"

    def test_read_errors_handled_gracefully(self, temp_archive_dir: Path) -> None:
        """Unreadable files log a warning and fall back gracefully."""
        day_dir = temp_archive_dir / "2026" / "10" / "11"
        day_dir.mkdir(parents=True)
        bad_file = day_dir / "session-002.md"
        bad_file.write_text("some content", encoding="utf-8")
        good_file = day_dir / "session-001.md"
        good_file.write_text(
            "### [2026-10-11 09:00:00] User\nFallback turn\n\n### [2026-10-11 09:00:01] IRIS\nFallback reply\n",
            encoding="utf-8",
        )

        archive = ConversationArchive(archive_dir=temp_archive_dir)
        with patch.object(Path, "read_text", side_effect=[IOError("Disk error"), "### [2026-10-11 09:00:00] User\nFallback turn\n\n### [2026-10-11 09:00:01] IRIS\nFallback reply\n"]):
            turns = archive.load_recent_turns()
            assert len(turns) == 2
            assert turns[0].content == "Fallback turn"


class TestAssistantSessionRestartIntegration:
    def test_assistant_restores_context_and_prevents_duplicate_file_turns(
        self, temp_archive_dir: Path
    ) -> None:
        """
        End-to-end restart verification:
        1. Run 1: Assistant converses and saves Session 1.
        2. Run 2: Assistant starts, loads Session 1 context into memory.
        3. Verify Session 2 file on disk does NOT duplicate Session 1 turns.
        4. Assistant processes a new turn in Session 2.
        5. Verify Session 2 file contains ONLY Session 2 turns.
        6. Verify in-memory context retains both Session 1 and Session 2 turns.
        7. Verify system prompt is preserved in messages sent to model.
        """
        cfg = IRISConfig(
            llm_provider="mock",
            tts_provider="mock",
            stt_provider="mock",
            archive_dir=temp_archive_dir,
            cooldown_ms=10,
        )

        # ── RUN 1 ─────────────────────────────────────────────────────────────
        model1 = MockModelAdapter(responses=["First assistant response."])
        assistant1 = IRISAssistant(
            config=cfg,
            model_adapter=model1,
            tts_adapter=MockTTS(),
            stt_adapter=MockSTT(),
        )
        assistant1.start()
        reply1 = assistant1.process_text_turn("First user message.")
        assert reply1 == "First assistant response."
        session_file_1 = assistant1.archive.session_file
        assert session_file_1 is not None and session_file_1.exists()
        assistant1.stop()

        content_1 = session_file_1.read_text(encoding="utf-8")
        assert "First user message." in content_1
        assert "First assistant response." in content_1

        # ── RUN 2 (RESTART) ───────────────────────────────────────────────────
        model2 = MockModelAdapter(responses=["Second assistant response."])
        assistant2 = IRISAssistant(
            config=cfg,
            model_adapter=model2,
            tts_adapter=MockTTS(),
            stt_adapter=MockSTT(),
        )

        # Before start: turn count is 0
        assert assistant2.conversation.turn_count == 0

        # Start assistant: context should be restored from Session 1
        assistant2.start()
        assert assistant2.conversation.turn_count == 2
        assert assistant2.conversation.get_last_user_text() == "First user message."
        assert assistant2.conversation.get_last_assistant_text() == "First assistant response."

        session_file_2 = assistant2.archive.session_file
        assert session_file_2 is not None and session_file_2.exists()
        assert session_file_2 != session_file_1

        # Verify Session 2 file on disk does NOT duplicate Session 1 turns!
        content_2_initial = session_file_2.read_text(encoding="utf-8")
        assert "First user message." not in content_2_initial
        assert "First assistant response." not in content_2_initial

        # Verify system prompt is preserved as first message
        messages = assistant2.conversation.get_messages()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == IRIS_SYSTEM_PROMPT
        assert messages[1] == {"role": "user", "content": "First user message."}
        assert messages[2] == {"role": "assistant", "content": "First assistant response."}

        # Process new turn in Session 2
        reply2 = assistant2.process_text_turn("Second user message.")
        assert reply2 == "Second assistant response."

        # Model should have received the full conversation context (system + 2 restored + 1 new user)
        last_call_messages = model2.received_messages[-1]
        assert len(last_call_messages) == 4
        assert last_call_messages[0]["role"] == "system"
        assert last_call_messages[1] == {"role": "user", "content": "First user message."}
        assert last_call_messages[2] == {"role": "assistant", "content": "First assistant response."}
        assert last_call_messages[3] == {"role": "user", "content": "Second user message."}

        assistant2.stop()

        # Verify Session 2 file on disk contains ONLY the new turn
        content_2_final = session_file_2.read_text(encoding="utf-8")
        assert "First user message." not in content_2_final
        assert "Second user message." in content_2_final
        assert "Second assistant response." in content_2_final
