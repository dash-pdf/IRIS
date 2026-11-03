"""
iris/services/archive.py
------------------------
Durable Markdown conversation archive.

Responsibility: Write every conversation turn to a timestamped Markdown
file. This is the canonical human-readable record of all IRIS conversations.

Key design decisions:
- One file per session. Sessions map to a date-based folder hierarchy.
- Files are append-only during a session. Past sessions are never modified.
- Errors are recorded (without secrets) rather than silently discarded.
- Thread-safe writes via an internal lock (single-process safety).
- Archive and long-term memory (Honcho) are separate concerns. This module
  knows nothing about memory retrieval.

File layout:
    memory/conversations/
        YYYY/
            MM/
                DD/
                    session-001.md
                    session-002.md
                    ...

Markdown format per turn:
    ### [2026-10-09 15:33:01 IST] User
    Hello IRIS, can you hear me?

    ### [2026-10-09 15:33:03 IST] IRIS
    Yes — loud and clear. How can I help?
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from iris.services.conversation import Turn

logger = logging.getLogger(__name__)


# Speaker labels used in the Markdown file.
SpeakerRole = Literal["user", "assistant", "system", "error"]

_SPEAKER_LABEL: dict[SpeakerRole, str] = {
    "user": "User",
    "assistant": "IRIS",
    "system": "System",
    "error": "Error",
}

_SESSION_FILE_TEMPLATE = "session-{n:03d}.md"


class ConversationArchive:
    """
    Writes conversation turns to a Markdown file for the current session.

    Usage:
        archive = ConversationArchive(archive_dir=Path("memory/conversations"))
        archive.open_session()
        archive.write_turn("user", "Hello IRIS.")
        archive.write_turn("assistant", "Hello. How can I help?")
        archive.close_session()

    The archive is safe to use from a single thread (the main conversation
    loop). The internal lock guards against accidental concurrent writes.
    """

    def __init__(self, archive_dir: Path, persona_name: str = "IRIS") -> None:
        self._archive_dir = archive_dir
        self._persona_name = persona_name
        self._session_file: Path | None = None
        self._lock = threading.Lock()
        self._session_open = False

        # Override the assistant label with the configured persona name.
        self._speaker_labels: dict[SpeakerRole, str] = {
            **_SPEAKER_LABEL,
            "assistant": persona_name,
        }

    # ── Session lifecycle ─────────────────────────────────────────────────

    def open_session(self) -> Path:
        """
        Open a new session file for today's date.

        Creates the directory tree if needed. Finds the next available
        session number (session-001, session-002, …) so previous sessions
        are never overwritten.

        Returns the path to the newly created session file.
        """
        with self._lock:
            if self._session_open:
                raise RuntimeError(
                    "A session is already open. Call close_session() first."
                )

            session_dir = self._session_dir_for_today()
            session_dir.mkdir(parents=True, exist_ok=True)

            session_file = self._next_session_file(session_dir)
            self._session_file = session_file
            self._session_open = True

            # Write the session header.
            now = _now_str()
            header = (
                f"# IRIS Conversation — {_today_str()}\n\n"
                f"**Session file:** `{session_file.name}`  \n"
                f"**Started:** {now}\n\n"
                f"---\n\n"
            )
            session_file.write_text(header, encoding="utf-8")
            return session_file

    def close_session(self) -> None:
        """Append a closing footer and mark the session as closed."""
        with self._lock:
            if not self._session_open or self._session_file is None:
                return
            footer = f"\n---\n\n**Session ended:** {_now_str()}\n"
            self._append(footer)
            self._session_open = False

    # ── Turn writing ──────────────────────────────────────────────────────

    def write_turn(self, role: SpeakerRole, text: str) -> None:
        """
        Append a single conversation turn to the session file.

        Args:
            role:  "user", "assistant", "system", or "error".
            text:  The spoken or generated text. Must not contain secrets.

        Raises:
            RuntimeError: If called before open_session().
        """
        with self._lock:
            self._require_open()
            label = self._speaker_labels.get(role, role.capitalize())
            timestamp = _now_str()
            entry = f"### [{timestamp}] {label}\n{text.strip()}\n\n"
            self._append(entry)

    def write_error(self, message: str) -> None:
        """
        Record an error turn. Use this when a downstream service fails.

        The message must NOT contain API keys, tokens, or any secret.
        The caller is responsible for sanitising the message before passing
        it here.
        """
        self.write_turn("error", f"⚠️ {message}")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def session_file(self) -> Path | None:
        """Path to the current session file, or None if no session is open."""
        return self._session_file

    @property
    def is_open(self) -> bool:
        """True while a session is active."""
        return self._session_open

    # ── Internal helpers ──────────────────────────────────────────────────

    def _append(self, text: str) -> None:
        """Append text to the current session file (must hold _lock)."""
        assert self._session_file is not None
        with self._session_file.open("a", encoding="utf-8") as fh:
            fh.write(text)

    def _require_open(self) -> None:
        if not self._session_open or self._session_file is None:
            raise RuntimeError(
                "No session is open. Call open_session() before writing turns."
            )

    def _session_dir_for_today(self) -> Path:
        today = datetime.now()
        return self._archive_dir / f"{today.year:04d}" / f"{today.month:02d}" / f"{today.day:02d}"

    @staticmethod
    def _next_session_file(session_dir: Path) -> Path:
        """Find the next session-NNN.md filename that does not yet exist."""
        n = 1
        while True:
            candidate = session_dir / _SESSION_FILE_TEMPLATE.format(n=n)
            if not candidate.exists():
                return candidate
            n += 1

    # ── History Recovery & Inspection ─────────────────────────────────────

    @staticmethod
    def _session_sort_key(path: Path) -> tuple[int, int, int, int, str]:
        """Extract (year, month, day, session_number, str_path) for chronological ordering."""
        try:
            parts = path.parts
            year = int(parts[-4])
            month = int(parts[-3])
            day = int(parts[-2])
            m = re.search(r"session-(\d+)", path.stem)
            num = int(m.group(1)) if m else 0
            return (year, month, day, num, str(path))
        except Exception:
            return (0, 0, 0, 0, str(path))

    def get_session_files(self, exclude_file: Path | None = None) -> list[Path]:
        """
        Return all existing session files in chronological order (oldest to newest).
        """
        if not self._archive_dir.exists():
            return []

        target_exclude = exclude_file if exclude_file is not None else (
            self._session_file if self._session_open else None
        )
        excluded_resolved = target_exclude.resolve() if target_exclude is not None else None

        files: list[Path] = []
        for f in self._archive_dir.rglob("session-*.md"):
            if not f.is_file():
                continue
            if excluded_resolved is not None and f.resolve() == excluded_resolved:
                continue
            files.append(f)

        files.sort(key=self._session_sort_key)
        return files

    def get_latest_session_file(self, exclude_file: Path | None = None) -> Path | None:
        """Return the path to the most recent session file on disk, or None."""
        files = self.get_session_files(exclude_file=exclude_file)
        return files[-1] if files else None

    def load_recent_turns(
        self,
        max_turns: int = 10,
        exclude_file: Path | None = None,
    ) -> list[Turn]:
        """
        Load the most recent relevant session and restore up to max_turns complete
        conversational turns.

        Scans backwards through session files (newest first), finds the latest
        session that contains valid complete turns, and returns up to max_turns in
        chronological order.
        """
        session_files = self.get_session_files(exclude_file=exclude_file)
        if not session_files:
            return []

        for session_path in reversed(session_files):
            try:
                content = session_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Failed to read session file %s: %s", session_path, exc)
                continue

            turns = parse_session_markdown(content, persona_name=self._persona_name)
            if not turns:
                continue

            # Slice up to max_turns
            if max_turns > 0 and len(turns) > max_turns:
                turns = turns[-max_turns:]
                # Ensure the restored slice begins with a user turn if possible
                while turns and turns[0].role != "user":
                    turns.pop(0)

            return turns

        return []


# ── Markdown Parser ───────────────────────────────────────────────────────────

def parse_session_markdown(content: str, persona_name: str = "IRIS") -> list[Turn]:
    """
    Parse conversation turns from a session Markdown file.

    Extracts genuine User and Assistant turns in chronological order.
    Filters out system headers, footers, error entries (⚠️), and session metadata.
    Discards any trailing incomplete user turn to guarantee the restored context
    ends cleanly on an assistant reply.
    """
    header_pattern = re.compile(r"^###\s+\[(?P<timestamp>[^\]]+)\]\s+(?P<label>.+)$")
    assistant_aliases = {"iris", "assistant", persona_name.strip().lower()}

    raw_turns: list[Turn] = []
    current_role: Literal["user", "assistant"] | None = None
    current_lines: list[str] = []

    def commit_turn() -> None:
        nonlocal current_role, current_lines
        if current_role is not None and current_lines:
            text = "\n".join(current_lines).strip()
            # Ignore warning/error text or empty messages
            if text and not text.startswith("⚠️"):
                raw_turns.append(Turn(role=current_role, content=text))
        current_role = None
        current_lines = []

    for line in content.splitlines():
        m = header_pattern.match(line.strip())
        if m:
            commit_turn()
            label = m.group("label").strip()
            label_lower = label.lower()
            if label_lower == "user":
                current_role = "user"
            elif label_lower in assistant_aliases:
                current_role = "assistant"
            else:
                current_role = None
            continue

        # Horizontal rule divider (marks header/footer)
        if line.strip() == "---":
            commit_turn()
            continue

        # Document title
        if line.startswith("#"):
            commit_turn()
            continue

        if current_role is not None:
            # Metadata lines
            if line.strip().startswith("**Session"):
                commit_turn()
                continue
            current_lines.append(line)

    commit_turn()

    # Drop any trailing incomplete user turn (unanswered prompt)
    while raw_turns and raw_turns[-1].role == "user":
        raw_turns.pop()

    return raw_turns



# ── Formatting helpers ────────────────────────────────────────────────────────

def _now_str() -> str:
    """Current local time formatted for Markdown, e.g. '2026-10-09 15:33:01'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today_str() -> str:
    """Today's date, e.g. '2026-10-09'."""
    return datetime.now().strftime("%Y-%m-%d")

