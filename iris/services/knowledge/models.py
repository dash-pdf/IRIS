"""
iris/services/knowledge/models.py
---------------------------------
Data models and constants for the IRIS Markdown Knowledge Vault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VAULT_FOLDERS: tuple[str, ...] = (
    "Inbox",
    "Notes",
    "Projects",
    "Research",
    "Archive",
)

VALID_NOTE_TYPES: tuple[str, ...] = (
    "note",
    "inbox",
    "project",
    "research",
    "reference",
    "archive",
)

DEFAULT_FOLDER_BY_TYPE: dict[str, str] = {
    "inbox": "Inbox",
    "note": "Notes",
    "project": "Projects",
    "research": "Research",
    "reference": "Notes",
    "archive": "Archive",
}


@dataclass
class NoteMetadata:
    """YAML frontmatter metadata for a Knowledge Vault Markdown note."""

    note_id: str
    title: str
    created_at: str
    updated_at: str
    note_type: str = "note"
    folder: str = "Notes"
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    source_url: str = ""


@dataclass
class NoteDocument:
    """In-memory representation of a persisted Knowledge Vault note."""

    metadata: NoteMetadata
    body: str
    path: Path
    relative_path: str = ""


@dataclass(frozen=True)
class SearchResult:
    """Ranked search match returned by the SQLite FTS5 knowledge index."""

    note_id: str
    title: str
    path: Path
    relative_path: str
    folder: str
    note_type: str
    tags: list[str]
    updated_at: str
    excerpt: str
    score: float
