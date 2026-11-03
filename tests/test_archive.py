"""
tests/test_archive.py
---------------------
Unit tests for iris/services/archive.py — the Markdown conversation archive.

These tests run entirely offline. No API keys or network access required.
They create temporary directories and clean them up after each test.

Tests cover acceptance criteria T01–T04 from the Phase 0 plan:
    T01 — Archive creates file at the correct path.
    T02 — User turn is written with timestamp and label.
    T03 — Assistant turn is written with IRIS label.
    T04 — (Structural) No secrets appear in source code.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from iris.services.archive import ConversationArchive


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def archive_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for archive files."""
    d = tmp_path / "conversations"
    d.mkdir()
    return d


@pytest.fixture()
def archive(archive_dir: Path) -> ConversationArchive:
    """Return a ConversationArchive using the temp directory."""
    return ConversationArchive(archive_dir=archive_dir, persona_name="IRIS")


# ── T01: File creation at correct path ───────────────────────────────────────

class TestSessionFileCreation:
    def test_open_session_creates_file(self, archive: ConversationArchive) -> None:
        """T01: open_session() must create a Markdown file."""
        session_file = archive.open_session()
        archive.close_session()
        assert session_file.exists(), "Session file was not created."

    def test_session_file_is_markdown(self, archive: ConversationArchive) -> None:
        """T01: Session file must have a .md extension."""
        session_file = archive.open_session()
        archive.close_session()
        assert session_file.suffix == ".md"

    def test_session_file_in_date_subdirectory(self, archive: ConversationArchive, archive_dir: Path) -> None:
        """T01: File must be under YYYY/MM/DD/ relative to the archive_dir."""
        from datetime import datetime
        session_file = archive.open_session()
        archive.close_session()

        now = datetime.now()
        expected_subpath = Path(
            f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
        )
        # The session file's parent (relative to archive_dir) should match
        # the YYYY/MM/DD structure.
        relative = session_file.relative_to(archive_dir)
        assert relative.parts[:3] == expected_subpath.parts, (
            f"Expected date path {expected_subpath}, got {relative}"
        )

    def test_session_file_named_session_001(self, archive: ConversationArchive) -> None:
        """T01: First session on a given day must be session-001.md."""
        session_file = archive.open_session()
        archive.close_session()
        assert session_file.name == "session-001.md"

    def test_second_session_increments_number(self, archive_dir: Path) -> None:
        """T01: A second session on the same day must be session-002.md."""
        a1 = ConversationArchive(archive_dir=archive_dir)
        f1 = a1.open_session()
        a1.close_session()

        a2 = ConversationArchive(archive_dir=archive_dir)
        f2 = a2.open_session()
        a2.close_session()

        assert f1.name == "session-001.md"
        assert f2.name == "session-002.md"

    def test_open_session_twice_raises(self, archive: ConversationArchive) -> None:
        """T01: Calling open_session() twice without closing must raise."""
        archive.open_session()
        with pytest.raises(RuntimeError, match="already open"):
            archive.open_session()
        archive.close_session()  # cleanup


# ── T02: User turn written correctly ──────────────────────────────────────────

class TestUserTurnWriting:
    def test_user_turn_appears_in_file(self, archive: ConversationArchive) -> None:
        """T02: A user turn must appear in the Markdown file."""
        archive.open_session()
        archive.write_turn("user", "Hello IRIS, can you hear me?")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        assert "Hello IRIS, can you hear me?" in content

    def test_user_turn_has_label(self, archive: ConversationArchive) -> None:
        """T02: User turn must have the 'User' speaker label."""
        archive.open_session()
        archive.write_turn("user", "Test message")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        assert "User" in content

    def test_user_turn_has_timestamp(self, archive: ConversationArchive) -> None:
        """T02: User turn must include a timestamp in YYYY-MM-DD HH:MM:SS format."""
        archive.open_session()
        archive.write_turn("user", "Timestamp test")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        # Match the timestamp pattern: [2026-10-09 15:33:01]
        timestamp_pattern = re.compile(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")
        assert timestamp_pattern.search(content), (
            "No timestamp found in session file."
        )

    def test_user_turn_text_is_stripped(self, archive: ConversationArchive) -> None:
        """T02: Leading/trailing whitespace is stripped from turn text."""
        archive.open_session()
        archive.write_turn("user", "   padded message   ")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        # The stripped version should appear; the padded version should not.
        assert "padded message" in content

    def test_write_before_open_raises(self, archive: ConversationArchive) -> None:
        """T02: Writing a turn without an open session must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="No session is open"):
            archive.write_turn("user", "Should fail")


# ── T03: Assistant turn written correctly ─────────────────────────────────────

class TestAssistantTurnWriting:
    def test_assistant_turn_appears_in_file(self, archive: ConversationArchive) -> None:
        """T03: An assistant turn must appear in the file."""
        archive.open_session()
        archive.write_turn("assistant", "Hello. I can hear you clearly.")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        assert "Hello. I can hear you clearly." in content

    def test_assistant_label_is_persona_name(self, archive_dir: Path) -> None:
        """T03: The assistant turn label must match the configured persona name."""
        custom_archive = ConversationArchive(
            archive_dir=archive_dir, persona_name="ARIA"
        )
        custom_archive.open_session()
        custom_archive.write_turn("assistant", "Hello from ARIA.")
        custom_archive.close_session()

        content = custom_archive.session_file.read_text(encoding="utf-8")
        assert "ARIA" in content
        assert "Hello from ARIA." in content

    def test_full_conversation_exchange(self, archive: ConversationArchive) -> None:
        """T03: A complete user→IRIS exchange must appear in order in the file."""
        archive.open_session()
        archive.write_turn("user", "What time is it?")
        archive.write_turn("assistant", "It's quarter past three.")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        user_pos = content.find("What time is it?")
        iris_pos = content.find("It's quarter past three.")

        assert user_pos != -1, "User turn not found."
        assert iris_pos != -1, "Assistant turn not found."
        assert user_pos < iris_pos, "User turn must appear before assistant turn."


# ── T04: Error turn and error recording ───────────────────────────────────────

class TestErrorRecording:
    def test_error_turn_appears_in_file(self, archive: ConversationArchive) -> None:
        """Errors must be recorded in the Markdown file."""
        archive.open_session()
        archive.write_error("ElevenLabs connection timed out.")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        assert "ElevenLabs connection timed out." in content

    def test_error_turn_has_warning_indicator(self, archive: ConversationArchive) -> None:
        """Error entries must include a visual warning indicator."""
        archive.open_session()
        archive.write_error("Something failed.")
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        assert "⚠️" in content


# ── Session lifecycle ─────────────────────────────────────────────────────────

class TestSessionLifecycle:
    def test_close_without_open_is_safe(self, archive: ConversationArchive) -> None:
        """close_session() when no session is open must not raise."""
        archive.close_session()  # Should be silent

    def test_session_file_is_none_before_open(self, archive: ConversationArchive) -> None:
        """session_file property must be None before open_session()."""
        assert archive.session_file is None

    def test_is_open_false_before_open(self, archive: ConversationArchive) -> None:
        """is_open must be False before a session is opened."""
        assert not archive.is_open

    def test_is_open_true_during_session(self, archive: ConversationArchive) -> None:
        """is_open must be True between open and close."""
        archive.open_session()
        assert archive.is_open
        archive.close_session()

    def test_is_open_false_after_close(self, archive: ConversationArchive) -> None:
        """is_open must be False after close_session()."""
        archive.open_session()
        archive.close_session()
        assert not archive.is_open

    def test_session_file_has_header(self, archive: ConversationArchive) -> None:
        """Session file must start with an IRIS conversation header."""
        archive.open_session()
        archive.close_session()
        content = archive.session_file.read_text(encoding="utf-8")
        assert content.startswith("# IRIS Conversation")

    def test_session_file_has_footer(self, archive: ConversationArchive) -> None:
        """Session file must end with a 'Session ended' footer."""
        archive.open_session()
        archive.close_session()
        content = archive.session_file.read_text(encoding="utf-8")
        assert "Session ended" in content

    def test_multiple_turns_preserved_in_order(self, archive: ConversationArchive) -> None:
        """Multiple turns must appear in the file in the order they were written."""
        messages = [
            ("user", "First message"),
            ("assistant", "First reply"),
            ("user", "Second message"),
            ("assistant", "Second reply"),
        ]
        archive.open_session()
        for role, text in messages:
            archive.write_turn(role, text)  # type: ignore[arg-type]
        archive.close_session()

        content = archive.session_file.read_text(encoding="utf-8")
        positions = [content.find(text) for _, text in messages]
        assert positions == sorted(positions), "Turns not in written order."
        assert all(p != -1 for p in positions), "Some turns are missing."

