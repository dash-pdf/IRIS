"""
iris/services/knowledge/__init__.py
-----------------------------------
Markdown Knowledge Vault, Inbox Capture, and SQLite FTS5 Search Service.
"""

from iris.services.knowledge.frontmatter import parse_markdown_note, serialize_markdown_note
from iris.services.knowledge.index import KnowledgeIndex
from iris.services.knowledge.models import (
    VALID_NOTE_TYPES,
    VAULT_FOLDERS,
    NoteDocument,
    NoteMetadata,
    SearchResult,
)
from iris.services.knowledge.service import (
    InboxClassifierFn,
    KnowledgeService,
    VaultMemoryAdapter,
)
from iris.services.knowledge.storage import VaultStorage, atomic_write_text, slugify_title

__all__ = [
    "VAULT_FOLDERS",
    "VALID_NOTE_TYPES",
    "NoteMetadata",
    "NoteDocument",
    "SearchResult",
    "InboxClassifierFn",
    "KnowledgeService",
    "VaultMemoryAdapter",
    "VaultStorage",
    "KnowledgeIndex",
    "parse_markdown_note",
    "serialize_markdown_note",
    "atomic_write_text",
    "slugify_title",
]
