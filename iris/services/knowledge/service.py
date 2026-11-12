"""
iris/services/knowledge/service.py
----------------------------------
Unified Knowledge Service coordinating Markdown Vault storage, Inbox capture &
recovery, SQLite FTS5 indexing, and explicit-intent memory integration.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path

from iris.adapters.memory_stub import MemoryAdapter
from iris.services.knowledge.frontmatter import normalize_tags, utc_now_iso
from iris.services.knowledge.index import KnowledgeIndex
from iris.services.knowledge.models import (
    DEFAULT_FOLDER_BY_TYPE,
    VALID_NOTE_TYPES,
    NoteDocument,
    NoteMetadata,
    SearchResult,
)
from iris.services.knowledge.storage import VaultStorage

logger = logging.getLogger(__name__)

# Future hook signature for Phase 2+ AI/rule classification of Inbox items:
# fn(NoteDocument) -> (target_folder, new_title, new_tags, new_note_type)
InboxClassifierFn = Callable[[NoteDocument], tuple[str, str | None, list[str] | None, str | None]]


class KnowledgeService:
    """
    High-level Knowledge Vault API for voice, desktop UI, research browser, and routing tools.

    Markdown files are the permanent source of truth; SQLite FTS5 provides fast,
    disposable full-text and metadata indexing.
    """

    def __init__(
        self,
        vault_dir: Path,
        index_path: Path | None = None,
        sync_on_init: bool = True,
    ) -> None:
        self._storage = VaultStorage(vault_dir)
        resolved_index = (
            Path(index_path)
            if index_path is not None
            else self._storage.vault_dir.parent / "memory" / "index" / "knowledge.db"
        )
        self._index = KnowledgeIndex(resolved_index, self._storage.vault_dir)
        if sync_on_init:
            self.rebuild_index()

    @property
    def vault_dir(self) -> Path:
        return self._storage.vault_dir

    @property
    def index(self) -> KnowledgeIndex:
        return self._index

    @property
    def storage(self) -> VaultStorage:
        return self._storage

    # ── Note Lifecycle CRUD ───────────────────────────────────────────────────

    def create_note(
        self,
        title: str,
        content: str,
        folder: str | None = None,
        note_type: str = "note",
        tags: list[str] | None = None,
        source_url: str = "",
        status: str = "active",
    ) -> NoteDocument:
        """Create and index a permanent Markdown note with YAML frontmatter."""
        clean_title = title.strip() or "Untitled Note"
        clean_type = note_type.strip().lower() or "note"
        if clean_type not in VALID_NOTE_TYPES:
            clean_type = "note"

        target_folder = folder or DEFAULT_FOLDER_BY_TYPE.get(clean_type, "Notes")
        canonical_folder = self._storage.canonical_folder_name(target_folder)

        now_iso = utc_now_iso()
        metadata = NoteMetadata(
            note_id=uuid.uuid4().hex,
            title=clean_title,
            created_at=now_iso,
            updated_at=now_iso,
            note_type=clean_type,
            folder=canonical_folder,
            tags=normalize_tags(tags),
            status=status,
            source_url=source_url.strip(),
        )
        doc = self._storage.write_note(metadata, content)
        self._index.upsert_note(doc)
        return doc

    def get_note(self, identifier_or_path: str | Path) -> NoteDocument | None:
        """Retrieve a note by relative/absolute file path or `note_id`."""
        candidate_str = str(identifier_or_path).strip()
        if not candidate_str:
            return None

        # 1. Check indexed note_id
        rel_path = self._index.find_path_by_id(candidate_str)
        if rel_path:
            try:
                return self._storage.read_note(rel_path)
            except FileNotFoundError:
                self._index.remove_by_path(rel_path)
                return None

        # 2. Check path directly
        try:
            return self._storage.read_note(identifier_or_path)
        except (FileNotFoundError, ValueError):
            return None

    def update_note(
        self,
        identifier_or_path: str | Path,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        note_type: str | None = None,
        status: str | None = None,
        source_url: str | None = None,
    ) -> NoteDocument:
        """Update an existing note atomically and refresh its search index record."""
        existing = self.get_note(identifier_or_path)
        if existing is None:
            raise FileNotFoundError(f"Cannot update missing note: {identifier_or_path}")

        meta = existing.metadata
        if title is not None and title.strip():
            meta.title = title.strip()
        if tags is not None:
            meta.tags = normalize_tags(tags)
        if note_type is not None and note_type.strip():
            meta.note_type = note_type.strip().lower()
        if status is not None and status.strip():
            meta.status = status.strip()
        if source_url is not None:
            meta.source_url = source_url.strip()

        meta.updated_at = utc_now_iso()
        new_body = content if content is not None else existing.body
        updated_doc = self._storage.write_note(meta, new_body, target_path=existing.path)
        self._index.upsert_note(updated_doc)
        return updated_doc

    def move_note(self, identifier_or_path: str | Path, target_folder: str) -> NoteDocument:
        """Move a note cleanly between vault folders and update its search index entry."""
        existing = self.get_note(identifier_or_path)
        if existing is None:
            raise FileNotFoundError(f"Cannot move missing note: {identifier_or_path}")

        old_rel = existing.relative_path
        meta = existing.metadata
        meta.updated_at = utc_now_iso()

        moved_doc = self._storage.move_note(existing.path, target_folder, updated_metadata=meta)
        if old_rel != moved_doc.relative_path:
            self._index.remove_by_path(old_rel)
        self._index.upsert_note(moved_doc)
        return moved_doc

    def archive_note(self, identifier_or_path: str | Path) -> NoteDocument:
        """Move a note into `Archive/` and mark its status `archived`."""
        existing = self.get_note(identifier_or_path)
        if existing is None:
            raise FileNotFoundError(f"Cannot archive missing note: {identifier_or_path}")

        old_rel = existing.relative_path
        meta = existing.metadata
        meta.status = "archived"
        meta.updated_at = utc_now_iso()

        archived_doc = self._storage.move_note(existing.path, "Archive", updated_metadata=meta)
        if old_rel != archived_doc.relative_path:
            self._index.remove_by_path(old_rel)
        self._index.upsert_note(archived_doc)
        return archived_doc

    def delete_note(self, identifier_or_path: str | Path) -> bool:
        """Delete a note from disk and remove it from the search index."""
        existing = self.get_note(identifier_or_path)
        if existing is None:
            return False

        deleted = self._storage.delete_note(existing.path)
        if deleted:
            self._index.remove_by_path(existing.relative_path)
            self._index.remove_by_id(existing.metadata.note_id)
        return deleted

    # ── Capture Inbox & Crash-Safe Organisation ───────────────────────────────

    def capture_to_inbox(
        self,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
        source_url: str = "",
    ) -> NoteDocument:
        """
        Save explicit captured text safely into `vault/Inbox/` preserving the
        original meaning verbatim.
        """
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Cannot capture empty content into Inbox.")

        derived_title = title.strip() or self._derive_title(clean_content)
        return self.create_note(
            title=derived_title,
            content=clean_content,
            folder="Inbox",
            note_type="inbox",
            tags=tags,
            source_url=source_url,
            status="unprocessed",
        )

    def list_inbox(self) -> list[NoteDocument]:
        """List all notes currently waiting in `vault/Inbox/`."""
        docs: list[NoteDocument] = []
        for path in self._storage.list_note_files(folder="Inbox"):
            try:
                docs.append(self._storage.read_note(path))
            except Exception as exc:
                logger.warning("Failed reading Inbox item %s: %s", path.name, exc)
        return docs

    def process_inbox_item(
        self,
        identifier_or_path: str | Path,
        target_folder: str = "Notes",
        new_title: str | None = None,
        new_tags: list[str] | None = None,
        new_note_type: str | None = None,
        classifier_fn: InboxClassifierFn | None = None,
    ) -> NoteDocument:
        """
        Promote an Inbox item into a permanent vault folder (`Notes`, `Projects`, `Research`, etc.).

        Safety guarantee: if `classifier_fn`, validation, or destination write fails, the
        original Inbox note remains intact on disk and indexed under `Inbox/`.
        """
        existing = self.get_note(identifier_or_path)
        if existing is None:
            raise FileNotFoundError(f"Inbox note not found: {identifier_or_path}")

        chosen_folder = target_folder
        chosen_title = new_title
        chosen_tags = new_tags
        chosen_type = new_note_type

        # Optional Phase 2+ classifier hook; failure leaves Inbox item completely untouched
        if classifier_fn is not None:
            c_folder, c_title, c_tags, c_type = classifier_fn(existing)
            chosen_folder = c_folder or chosen_folder
            chosen_title = c_title or chosen_title
            chosen_tags = c_tags if c_tags is not None else chosen_tags
            chosen_type = c_type or chosen_type

        old_rel = existing.relative_path
        meta = existing.metadata
        if chosen_title and chosen_title.strip():
            meta.title = chosen_title.strip()
        if chosen_tags is not None:
            meta.tags = normalize_tags(chosen_tags)
        if chosen_type and chosen_type.strip():
            meta.note_type = chosen_type.strip().lower()
        elif meta.note_type == "inbox":
            canonical = self._storage.canonical_folder_name(chosen_folder)
            meta.note_type = "research" if canonical == "Research" else ("project" if canonical == "Projects" else "note")

        meta.status = "active"
        meta.updated_at = utc_now_iso()

        # Write destination first, then unlink old Inbox file (atomic recovery on crash)
        moved_doc = self._storage.move_note(existing.path, chosen_folder, updated_metadata=meta)
        if old_rel != moved_doc.relative_path:
            self._index.remove_by_path(old_rel)
        self._index.upsert_note(moved_doc)
        return moved_doc

    # ── Search & Rebuild ──────────────────────────────────────────────────────

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        note_type: str | None = None,
        folder: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Full-text and metadata search across the permanent Knowledge Vault."""
        return self._index.search(
            query=query,
            tags=tags,
            note_type=note_type,
            folder=folder,
            limit=limit,
        )

    def rebuild_index(self) -> int:
        """Rebuild the disposable SQLite FTS5 index completely from vault Markdown files."""
        return self._index.rebuild_from_vault(self._storage)

    @staticmethod
    def _derive_title(content: str, max_words: int = 8) -> str:
        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "Captured Note")
        first_line = first_line.lstrip("# ").strip()
        words = first_line.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]).rstrip(".,;:") + "..."
        return first_line[:80] or "Captured Note"


class VaultMemoryAdapter(MemoryAdapter):
    """
    Knowledge-backed `MemoryAdapter` implementation.

    Retrieves relevant permanent Markdown notes when queried while explicitly
    refusing automatic permanent-note creation during ordinary conversation turns.
    """

    def __init__(self, knowledge_service: KnowledgeService, max_results: int = 3) -> None:
        self._knowledge = knowledge_service
        self._max_results = max_results

    def get_relevant_context(self, user_message: str) -> str:
        results = self._knowledge.search(query=user_message, limit=self._max_results)
        if not results:
            return ""

        lines = ["Relevant permanent knowledge notes:"]
        for item in results:
            lines.append(f"- {item.title} ({item.relative_path}): {item.excerpt}")
        return "\n".join(lines)

    def store_turn(self, user_text: str, assistant_text: str) -> None:  # noqa: ARG002
        # Phase 1 invariant: ordinary dialogue turns are archived in memory/conversations/,
        # NEVER automatically promoted into permanent Knowledge Vault notes.
        return
