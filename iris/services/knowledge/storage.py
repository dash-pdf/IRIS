"""
iris/services/knowledge/storage.py
----------------------------------
Filesystem and atomic-write persistence layer for the Markdown Knowledge Vault.

Guarantees:
- Strict isolation from `memory/conversations/`.
- Atomic write (`temp file in target dir` -> `flush` + `fsync` -> `os.replace`) to prevent partial notes.
- Predictable, collision-resistant filenames and path-traversal validation.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from pathlib import Path

from iris.services.knowledge.frontmatter import (
    parse_markdown_note,
    serialize_markdown_note,
)
from iris.services.knowledge.models import VAULT_FOLDERS, NoteDocument, NoteMetadata

logger = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify_title(title: str, max_length: int = 60) -> str:
    """Convert a note title into a safe, predictable lowercase filename slug."""
    ascii_approx = title.lower().strip()
    slug = _NON_ALNUM_RE.sub("-", ascii_approx).strip("-")
    if not slug:
        return "untitled"
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-") or "untitled"
    return slug


def atomic_write_text(target_path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Atomically write text content to `target_path`.

    Writes to a hidden temporary file in the same directory, flushes and fsyncs to
    disk, then swaps via `os.replace`. On any failure, removes the temp file and
    leaves any existing destination file completely intact.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{target_path.name}.tmp-{uuid.uuid4().hex[:8]}"
    tmp_path = target_path.parent / tmp_name

    try:
        with open(tmp_path, "w", encoding=encoding, newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


class VaultStorage:
    """Manages Markdown note files inside the structured Knowledge Vault."""

    def __init__(self, vault_dir: Path) -> None:
        self._vault_dir = Path(vault_dir).resolve()
        self._lock = threading.Lock()
        self.ensure_structure()

    @property
    def vault_dir(self) -> Path:
        return self._vault_dir

    def ensure_structure(self) -> None:
        """Ensure all canonical vault subdirectories exist."""
        self._vault_dir.mkdir(parents=True, exist_ok=True)
        for folder in VAULT_FOLDERS:
            (self._vault_dir / folder).mkdir(parents=True, exist_ok=True)

    def resolve_folder(self, folder: str) -> Path:
        """Validate folder name and return its resolved directory inside `vault_dir`."""
        clean_folder = folder.strip()
        if clean_folder not in VAULT_FOLDERS:
            # Allow case-insensitive match against canonical folders
            folder_map = {f.lower(): f for f in VAULT_FOLDERS}
            if clean_folder.lower() in folder_map:
                clean_folder = folder_map[clean_folder.lower()]
            else:
                raise ValueError(
                    f"Unsupported vault folder '{folder}'. Must be one of: {', '.join(VAULT_FOLDERS)}"
                )
        target_dir = (self._vault_dir / clean_folder).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def canonical_folder_name(self, folder: str) -> str:
        """Return exact casing for a supported vault folder name."""
        return self.resolve_folder(folder).name

    def resolve_note_path(self, path_or_rel: Path | str) -> Path:
        """Resolve a relative or absolute path ensuring it stays within `vault_dir`."""
        candidate = Path(path_or_rel)
        resolved = candidate.resolve() if candidate.is_absolute() else (self._vault_dir / candidate).resolve()
        if not resolved.is_relative_to(self._vault_dir):
            raise ValueError(f"Path '{path_or_rel}' escapes the Knowledge Vault directory.")
        return resolved

    def allocate_note_path(self, title: str, folder: str) -> Path:
        """Allocate a unique, predictable `.md` filepath inside `folder` without collisions."""
        target_dir = self.resolve_folder(folder)
        slug = slugify_title(title)
        candidate = target_dir / f"{slug}.md"
        if not candidate.exists():
            return candidate

        for counter in range(2, 100):
            numbered = target_dir / f"{slug}-{counter}.md"
            if not numbered.exists():
                return numbered

        return target_dir / f"{slug}-{uuid.uuid4().hex[:6]}.md"

    def write_note(self, metadata: NoteMetadata, body: str, target_path: Path | None = None) -> NoteDocument:
        """Persist `metadata` and `body` atomically and return the resulting `NoteDocument`."""
        with self._lock:
            canonical_folder = self.canonical_folder_name(metadata.folder)
            metadata.folder = canonical_folder

            if target_path is not None:
                dest = self.resolve_note_path(target_path)
            else:
                dest = self.allocate_note_path(metadata.title, canonical_folder)

            content = serialize_markdown_note(metadata, body)
            atomic_write_text(dest, content)
            rel_path = dest.relative_to(self._vault_dir).as_posix()
            return NoteDocument(
                metadata=metadata,
                body=body.strip("\r\n"),
                path=dest,
                relative_path=rel_path,
            )

    def read_note(self, path_or_rel: Path | str) -> NoteDocument:
        """Read and parse a Markdown note from the vault."""
        resolved = self.resolve_note_path(path_or_rel)
        if not resolved.is_file():
            raise FileNotFoundError(f"Knowledge note not found: {resolved}")

        raw_text = resolved.read_text(encoding="utf-8")
        rel_path = resolved.relative_to(self._vault_dir).as_posix()
        inferred_folder = resolved.parent.name if resolved.parent != self._vault_dir else "Notes"
        if inferred_folder not in VAULT_FOLDERS:
            inferred_folder = "Notes"

        metadata, body = parse_markdown_note(raw_text, file_path=resolved, default_folder=inferred_folder)
        return NoteDocument(
            metadata=metadata,
            body=body,
            path=resolved,
            relative_path=rel_path,
        )

    def move_note(
        self,
        source_path_or_rel: Path | str,
        target_folder: str,
        updated_metadata: NoteMetadata | None = None,
        updated_body: str | None = None,
    ) -> NoteDocument:
        """
        Safely move a note into `target_folder` using write-then-unlink semantics so
        the source file is preserved if writing the destination fails.
        """
        with self._lock:
            src = self.resolve_note_path(source_path_or_rel)
            if not src.is_file():
                raise FileNotFoundError(f"Cannot move missing note: {src}")

            canonical_folder = self.canonical_folder_name(target_folder)
            existing_doc = self.read_note(src)

            metadata = updated_metadata or existing_doc.metadata
            metadata.folder = canonical_folder
            body = updated_body if updated_body is not None else existing_doc.body

            dest_dir = self.resolve_folder(canonical_folder)
            if src.parent.resolve() == dest_dir:
                dest = src
            else:
                dest = dest_dir / src.name
                if dest.exists():
                    dest = self.allocate_note_path(metadata.title, canonical_folder)

            serialized = serialize_markdown_note(metadata, body)
            atomic_write_text(dest, serialized)

            if dest.resolve() != src.resolve() and src.exists():
                src.unlink()

            rel_path = dest.relative_to(self._vault_dir).as_posix()
            return NoteDocument(
                metadata=metadata,
                body=body.strip("\r\n"),
                path=dest,
                relative_path=rel_path,
            )

    def delete_note(self, path_or_rel: Path | str) -> bool:
        """Delete a note file from the vault. Returns False if it does not exist."""
        with self._lock:
            resolved = self.resolve_note_path(path_or_rel)
            if not resolved.is_file():
                return False
            resolved.unlink()
            return True

    def list_note_files(self, folder: str | None = None) -> list[Path]:
        """List all `.md` note files in the vault or a specific folder."""
        if folder is not None:
            search_root = self.resolve_folder(folder)
            files = sorted(
                p for p in search_root.glob("*.md")
                if p.is_file() and not p.name.startswith(".")
            )
            return files

        all_files: list[Path] = []
        for canonical in VAULT_FOLDERS:
            folder_dir = self._vault_dir / canonical
            if folder_dir.is_dir():
                all_files.extend(
                    sorted(
                        p for p in folder_dir.rglob("*.md")
                        if p.is_file() and not p.name.startswith(".")
                    )
                )
        return all_files
