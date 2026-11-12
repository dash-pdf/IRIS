"""
iris/services/knowledge/index.py
--------------------------------
Disposable, rebuildable SQLite FTS5 search index over Markdown Knowledge Vault notes.

Design rules:
- Markdown files remain the canonical source of truth.
- Deleting or corrupting this database never deletes notes; `rebuild_from_vault()` restores it.
- Never stores secrets, API keys, OAuth tokens, or conversation raw session transcripts.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path

from iris.services.knowledge.frontmatter import normalize_tags
from iris.services.knowledge.models import NoteDocument, SearchResult
from iris.services.knowledge.storage import VaultStorage

logger = logging.getLogger(__name__)

_FTS_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _build_safe_fts_query(raw_query: str) -> str:
    """Convert arbitrary user text into a syntax-safe FTS5 MATCH query."""
    tokens = _FTS_TOKEN_RE.findall(raw_query)
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)


def _fallback_excerpt(text: str, max_chars: int = 180) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


class KnowledgeIndex:
    """SQLite FTS5 search index synchronised with `VaultStorage`."""

    def __init__(self, db_path: Path, vault_dir: Path) -> None:
        self._db_path = Path(db_path).resolve()
        self._vault_dir = Path(vault_dir).resolve()
        self._lock = threading.Lock()
        self._fts5_supported = False
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def fts5_supported(self) -> bool:
        return self._fts5_supported

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_notes (
                        relative_path TEXT PRIMARY KEY,
                        note_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        body TEXT NOT NULL,
                        folder TEXT NOT NULL,
                        note_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        tags_json TEXT NOT NULL,
                        tags_text TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        source_url TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_kn_note_id ON knowledge_notes(note_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_kn_folder ON knowledge_notes(folder)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_kn_type ON knowledge_notes(note_type)")

                try:
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                            relative_path UNINDEXED,
                            title,
                            body,
                            tags_text,
                            folder UNINDEXED,
                            note_type UNINDEXED,
                            tokenize = 'unicode61'
                        )
                        """
                    )
                    self._fts5_supported = True
                except sqlite3.OperationalError as exc:
                    self._fts5_supported = False
                    logger.warning("SQLite FTS5 unavailable (%s); falling back to LIKE search.", exc)

    def upsert_note(self, doc: NoteDocument) -> None:
        """Insert or update a single Markdown note in SQLite metadata + FTS5 tables."""
        rel_path = doc.relative_path or doc.path.relative_to(self._vault_dir).as_posix()
        tags = normalize_tags(doc.metadata.tags)
        tags_json = json.dumps(tags, ensure_ascii=False)
        tags_text = " ".join(tags)

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO knowledge_notes (
                        relative_path, note_id, title, body, folder, note_type,
                        status, tags_json, tags_text, created_at, updated_at, source_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(relative_path) DO UPDATE SET
                        note_id = excluded.note_id,
                        title = excluded.title,
                        body = excluded.body,
                        folder = excluded.folder,
                        note_type = excluded.note_type,
                        status = excluded.status,
                        tags_json = excluded.tags_json,
                        tags_text = excluded.tags_text,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at,
                        source_url = excluded.source_url
                    """,
                    (
                        rel_path,
                        doc.metadata.note_id,
                        doc.metadata.title,
                        doc.body,
                        doc.metadata.folder,
                        doc.metadata.note_type,
                        doc.metadata.status,
                        tags_json,
                        tags_text,
                        doc.metadata.created_at,
                        doc.metadata.updated_at,
                        doc.metadata.source_url,
                    ),
                )
                if self._fts5_supported:
                    conn.execute("DELETE FROM knowledge_fts WHERE relative_path = ?", (rel_path,))
                    conn.execute(
                        """
                        INSERT INTO knowledge_fts (
                            relative_path, title, body, tags_text, folder, note_type
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rel_path,
                            doc.metadata.title,
                            doc.body,
                            tags_text,
                            doc.metadata.folder,
                            doc.metadata.note_type,
                        ),
                    )

    def remove_by_path(self, relative_path: str) -> None:
        """Remove an entry by vault-relative path."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM knowledge_notes WHERE relative_path = ?", (relative_path,))
                if self._fts5_supported:
                    conn.execute("DELETE FROM knowledge_fts WHERE relative_path = ?", (relative_path,))

    def remove_by_id(self, note_id: str) -> None:
        """Remove any indexed rows matching `note_id`."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT relative_path FROM knowledge_notes WHERE note_id = ?", (note_id,)
                ).fetchall()
                rel_paths = [row["relative_path"] for row in rows]
                conn.execute("DELETE FROM knowledge_notes WHERE note_id = ?", (note_id,))
                if self._fts5_supported:
                    for rel_path in rel_paths:
                        conn.execute("DELETE FROM knowledge_fts WHERE relative_path = ?", (rel_path,))

    def find_path_by_id(self, note_id: str) -> str | None:
        """Lookup current vault-relative path for `note_id` if indexed."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT relative_path FROM knowledge_notes WHERE note_id = ?", (note_id,)
                ).fetchone()
                return str(row["relative_path"]) if row else None

    def rebuild_from_vault(self, storage: VaultStorage) -> int:
        """
        Rebuild the entire index cleanly from Markdown files currently on disk.
        Handles external edits, moves, and deletions safely.
        """
        self._init_schema()
        note_files = storage.list_note_files()
        docs: list[NoteDocument] = []
        for path in note_files:
            try:
                docs.append(storage.read_note(path))
            except Exception as exc:
                logger.warning("Skipping unreadable note during index rebuild (%s): %s", path.name, exc)

        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM knowledge_notes")
                if self._fts5_supported:
                    conn.execute("DELETE FROM knowledge_fts")

        for doc in docs:
            self.upsert_note(doc)
        return len(docs)

    def count_notes(self) -> int:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM knowledge_notes").fetchone()
                return int(row["cnt"]) if row else 0

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        note_type: str | None = None,
        folder: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search indexed notes by full-text query and optional metadata filters
        (`tags`, `note_type`, `folder`).
        """
        norm_tags = normalize_tags(tags)
        clean_type = note_type.strip().lower() if note_type else None
        clean_folder = folder.strip() if folder else None
        safe_fts = _build_safe_fts_query(query)

        with self._lock:
            with self._connect() as conn:
                filters_sql: list[str] = []
                params: list[object] = []

                if clean_type:
                    filters_sql.append("LOWER(n.note_type) = ?")
                    params.append(clean_type)
                if clean_folder:
                    filters_sql.append("LOWER(n.folder) = ?")
                    params.append(clean_folder.lower())
                for tag in norm_tags:
                    # Match exact token inside JSON array via LIKE safely, then exact-check in Python
                    filters_sql.append("n.tags_json LIKE ?")
                    params.append(f'%"{tag}"%')

                where_extra = (" AND " + " AND ".join(filters_sql)) if filters_sql else ""

                if safe_fts and self._fts5_supported:
                    sql = f"""
                        SELECT
                            n.relative_path,
                            n.note_id,
                            n.title,
                            n.body,
                            n.folder,
                            n.note_type,
                            n.tags_json,
                            n.updated_at,
                            snippet(knowledge_fts, 2, '[', ']', '...', 18) AS fts_excerpt,
                            bm25(knowledge_fts, 0.0, 4.0, 1.0, 2.0) AS rank_score
                        FROM knowledge_fts
                        JOIN knowledge_notes n ON n.relative_path = knowledge_fts.relative_path
                        WHERE knowledge_fts MATCH ? {where_extra}
                        ORDER BY rank_score ASC, n.updated_at DESC
                        LIMIT ?
                    """
                    rows = conn.execute(sql, [safe_fts, *params, max(limit * 2, limit)]).fetchall()
                else:
                    # Metadata-only filter or LIKE fallback when query is empty / FTS5 missing
                    if query.strip():
                        like_term = f"%{query.strip()}%"
                        like_clause = "(n.title LIKE ? OR n.body LIKE ? OR n.tags_text LIKE ?)"
                        all_params = [like_term, like_term, like_term, *params, max(limit * 2, limit)]
                        where_stmt = f"WHERE {like_clause} {where_extra}"
                    else:
                        all_params = [*params, max(limit * 2, limit)]
                        where_stmt = ("WHERE " + " AND ".join(filters_sql)) if filters_sql else ""

                    sql = f"""
                        SELECT
                            n.relative_path,
                            n.note_id,
                            n.title,
                            n.body,
                            n.folder,
                            n.note_type,
                            n.tags_json,
                            n.updated_at,
                            '' AS fts_excerpt,
                            0.0 AS rank_score
                        FROM knowledge_notes n
                        {where_stmt}
                        ORDER BY n.updated_at DESC
                        LIMIT ?
                    """
                    rows = conn.execute(sql, all_params).fetchall()

        results: list[SearchResult] = []
        required_tag_set = set(norm_tags)
        for row in rows:
            parsed_tags: list[str] = json.loads(row["tags_json"])
            if required_tag_set and not required_tag_set.issubset(set(parsed_tags)):
                continue

            raw_excerpt = str(row["fts_excerpt"] or "").strip()
            excerpt = raw_excerpt if raw_excerpt else _fallback_excerpt(str(row["body"]))
            rel_path = str(row["relative_path"])

            results.append(
                SearchResult(
                    note_id=str(row["note_id"]),
                    title=str(row["title"]),
                    path=self._vault_dir / rel_path,
                    relative_path=rel_path,
                    folder=str(row["folder"]),
                    note_type=str(row["note_type"]),
                    tags=parsed_tags,
                    updated_at=str(row["updated_at"]),
                    excerpt=excerpt,
                    score=float(row["rank_score"]),
                )
            )
            if len(results) >= limit:
                break

        return results
