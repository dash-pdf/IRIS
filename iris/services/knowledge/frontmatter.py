"""
iris/services/knowledge/frontmatter.py
--------------------------------------
Lightweight, deterministic YAML frontmatter parser and serializer for Markdown notes.

Avoids external parser fragility while supporting quoted strings, colons in values,
URLs, and both inline `[a, b]` and block `- item` tag lists.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from iris.services.knowledge.models import NoteMetadata

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)


def utc_now_iso() -> str:
    """Return current UTC timestamp in second-precision ISO 8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalise, strip, lowercase, and deduplicate tags preserving order."""
    if not tags:
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in tags:
        tag = raw.strip().lstrip("#").lower()
        if tag and tag not in seen:
            seen.add(tag)
            cleaned.append(tag)
    return cleaned


def _format_scalar(value: str) -> str:
    """Serialize a scalar string safely as YAML."""
    if not value:
        return '""'
    # Quote whenever characters could confuse YAML parsers (colons, leading punctuation, quotes, newlines)
    if (
        any(ch in value for ch in (":", "#", '"', "'", "[", "]", "{", "}", "\n", "\r"))
        or value[0] in ("-", "?", "*", "&", "!", "|", ">", "%", "@", "`", " ")
        or value[-1] == " "
        or value.lower() in {"true", "false", "null", "yes", "no"}
    ):
        return json.dumps(value, ensure_ascii=False)
    return value


def _unquote_scalar(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            decoded = json.loads(text)
            return str(decoded)
        except Exception:
            return text[1:-1]
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'")
    return text


def serialize_markdown_note(metadata: NoteMetadata, body: str) -> str:
    """Serialize `NoteMetadata` and Markdown `body` into canonical frontmatter Markdown."""
    tags = normalize_tags(metadata.tags)
    tags_yaml = "[" + ", ".join(_format_scalar(t) for t in tags) + "]" if tags else "[]"

    lines = [
        "---",
        f"id: {_format_scalar(metadata.note_id)}",
        f"title: {_format_scalar(metadata.title)}",
        f"type: {_format_scalar(metadata.note_type)}",
        f"folder: {_format_scalar(metadata.folder)}",
        f"status: {_format_scalar(metadata.status)}",
        f"tags: {tags_yaml}",
        f"created_at: {_format_scalar(metadata.created_at)}",
        f"updated_at: {_format_scalar(metadata.updated_at)}",
    ]
    if metadata.source_url:
        lines.append(f"source_url: {_format_scalar(metadata.source_url)}")
    lines.append("---")

    clean_body = body.strip("\r\n")
    if clean_body:
        return "\n".join(lines) + "\n\n" + clean_body + "\n"
    return "\n".join(lines) + "\n"


def _parse_yaml_block(yaml_text: str) -> dict[str, object]:
    """Parse simple flat YAML key-value pairs and string lists safely."""
    data: dict[str, object] = {}
    current_list_key: str | None = None

    for raw_line in yaml_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if current_list_key and stripped.startswith("- "):
            item = _unquote_scalar(stripped[2:])
            existing = data.setdefault(current_list_key, [])
            if isinstance(existing, list) and item:
                existing.append(item)
            continue

        current_list_key = None
        if ":" not in raw_line:
            continue

        key_part, val_part = raw_line.split(":", 1)
        key = key_part.strip()
        val_str = val_part.strip()

        if val_str == "":
            data[key] = []
            current_list_key = key
        elif val_str.startswith("[") and val_str.endswith("]"):
            inner = val_str[1:-1].strip()
            if not inner:
                data[key] = []
            else:
                items = [_unquote_scalar(part) for part in inner.split(",")]
                data[key] = [item for item in items if item]
        else:
            data[key] = _unquote_scalar(val_str)

    return data


def _infer_title_from_body_or_path(body: str, file_path: Path | None) -> str:
    for line in body.splitlines():
        clean = line.strip()
        if clean.startswith("# "):
            heading = clean[2:].strip()
            if heading:
                return heading
    if file_path is not None:
        return file_path.stem.replace("-", " ").replace("_", " ").strip().title() or "Untitled Note"
    return "Untitled Note"


def parse_markdown_note(
    raw_text: str,
    file_path: Path | None = None,
    default_folder: str = "Notes",
) -> tuple[NoteMetadata, str]:
    """
    Parse a Markdown document with optional YAML frontmatter into `(NoteMetadata, body)`.
    Handles externally authored Markdown files missing frontmatter gracefully.
    """
    match = _FRONTMATTER_RE.match(raw_text)
    if match:
        yaml_block, body = match.group(1), match.group(2).strip("\r\n")
        fields = _parse_yaml_block(yaml_block)
    else:
        fields = {}
        body = raw_text.strip("\r\n")

    now_iso = utc_now_iso()
    note_id = str(fields.get("id") or "").strip() or uuid.uuid4().hex
    title = str(fields.get("title") or "").strip() or _infer_title_from_body_or_path(body, file_path)
    note_type = str(fields.get("type") or fields.get("note_type") or "note").strip().lower() or "note"
    folder = str(fields.get("folder") or default_folder).strip() or default_folder
    status = str(fields.get("status") or "active").strip() or "active"
    source_url = str(fields.get("source_url") or "").strip()
    created_at = str(fields.get("created_at") or "").strip() or now_iso
    updated_at = str(fields.get("updated_at") or "").strip() or created_at

    raw_tags = fields.get("tags")
    if isinstance(raw_tags, list):
        tags = normalize_tags([str(t) for t in raw_tags])
    elif isinstance(raw_tags, str) and raw_tags.strip():
        tags = normalize_tags(raw_tags.split(","))
    else:
        tags = []

    metadata = NoteMetadata(
        note_id=note_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        note_type=note_type,
        folder=folder,
        tags=tags,
        status=status,
        source_url=source_url,
    )
    return metadata, body
