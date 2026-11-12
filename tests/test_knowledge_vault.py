"""
tests/test_knowledge_vault.py
-----------------------------
Comprehensive unit & integration tests for Phase 1: Knowledge Foundation.

Covers:
- Vault directory isolation & Markdown YAML frontmatter CRUD
- Edge cases (quotes, colons, unicode, embedded '---' horizontal rules)
- Collision-resistant filenames & path-traversal protection
- Atomic file writes & cleanup on interrupted replacement
- Capture Inbox & crash-safe recovery when classification fails
- Move, archive, and delete lifecycle syncing disk + index
- Disposable/rebuildable SQLite FTS5 index & punctuation-safe queries
- Explicit-intent-only saving invariant (never auto-saves ordinary dialogue)
- Router, CloudRouter, and IRISAssistant integration cleanly separated from
  `memory/conversations/` session archives.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iris.adapters.model import create_model_adapter
from iris.adapters.model.cloud_router import CloudRouterModelAdapter
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig
from iris.core.router import RouteType, classify_request
from iris.services.archive import ConversationArchive
from iris.services.knowledge import (
    VAULT_FOLDERS,
    KnowledgeService,
    VaultMemoryAdapter,
    atomic_write_text,
)


def _make_config(tmp_path: Path, openrouter_api_key: str = "") -> IRISConfig:
    return IRISConfig(
        llm_provider="cloud_router",
        openrouter_api_key=openrouter_api_key,
        archive_dir=tmp_path / "memory" / "conversations",
        vault_dir=tmp_path / "vault",
        knowledge_index_path=tmp_path / "memory" / "index" / "knowledge.db",
    )


def test_vault_creates_canonical_folders_and_crud(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    service = KnowledgeService(vault_dir=cfg.vault_dir, index_path=cfg.knowledge_index_path)

    for folder in VAULT_FOLDERS:
        assert (cfg.vault_dir / folder).is_dir()

    created = service.create_note(
        title='Architecture: "Phase 1" — Spec & Overview',
        content="# Overview\nFirst section.\n\n---\n\nSecond section after rule.",
        folder="Projects",
        note_type="project",
        tags=["Architecture", "Phase-1", "architecture"],
        source_url="https://example.org/spec",
    )

    assert created.path.exists()
    assert created.path.parent == (cfg.vault_dir / "Projects").resolve()
    assert created.metadata.tags == ["architecture", "phase-1"]
    assert created.metadata.note_type == "project"
    assert created.metadata.source_url == "https://example.org/spec"

    # Read back from disk by ID and verify --- inside body survived round-trip
    fetched = service.get_note(created.metadata.note_id)
    assert fetched is not None
    assert fetched.metadata.title == 'Architecture: "Phase 1" — Spec & Overview'
    assert "---\n\nSecond section after rule." in fetched.body

    # Update note preserves note_id & created_at, refreshes updated_at & index
    updated = service.update_note(
        created.metadata.note_id,
        title="Updated Phase 1 Architecture",
        content="Refined durable Markdown specification.",
        tags=["architecture", "refined"],
    )
    assert updated.metadata.note_id == created.metadata.note_id
    assert updated.metadata.created_at == created.metadata.created_at
    assert "Refined durable" in updated.body

    hits = service.search("durable specification", tags=["refined"], folder="Projects")
    assert len(hits) == 1
    assert hits[0].note_id == created.metadata.note_id
    assert hits[0].title == "Updated Phase 1 Architecture"


def test_collision_resistant_filenames_and_path_traversal_guard(tmp_path: Path) -> None:
    service = KnowledgeService(
        vault_dir=tmp_path / "vault",
        index_path=tmp_path / "memory" / "index" / "knowledge.db",
    )

    first = service.create_note(title="Meeting Summary", content="Alpha topic", folder="Notes")
    second = service.create_note(title="Meeting Summary", content="Beta topic", folder="Notes")

    assert first.path != second.path
    assert first.path.exists() and second.path.exists()
    assert service.get_note(first.metadata.note_id).body == "Alpha topic"
    assert service.get_note(second.metadata.note_id).body == "Beta topic"

    with pytest.raises(ValueError):
        service.create_note(title="Bad Folder", content="X", folder="../../etc")

    with pytest.raises(ValueError):
        service.storage.resolve_note_path("../outside.md")


def test_atomic_write_preserves_existing_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "vault" / "Notes" / "important.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ORIGINAL SAFE CONTENT", encoding="utf-8")

    with patch("iris.services.knowledge.storage.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            atomic_write_text(target, "CORRUPTED PARTIAL WRITE")

    # Original file untouched, leftover temp files cleaned up
    assert target.read_text(encoding="utf-8") == "ORIGINAL SAFE CONTENT"
    leftovers = list(target.parent.glob("*.tmp-*"))
    assert leftovers == []


def test_move_archive_and_delete_lifecycle_updates_index(tmp_path: Path) -> None:
    service = KnowledgeService(
        vault_dir=tmp_path / "vault",
        index_path=tmp_path / "memory" / "index" / "knowledge.db",
    )

    note = service.create_note(
        title="SQLite FTS5 Research",
        content="FTS5 supports fast BM25 ranking over local Markdown notes.",
        folder="Notes",
        tags=["sqlite", "fts5"],
    )

    moved = service.move_note(note.metadata.note_id, "Research")
    assert moved.metadata.folder == "Research"
    assert moved.relative_path.startswith("Research/")
    assert not note.path.exists()
    assert moved.path.exists()
    assert service.search("BM25", folder="Notes") == []
    assert len(service.search("BM25", folder="Research")) == 1

    archived = service.archive_note(note.metadata.note_id)
    assert archived.metadata.folder == "Archive"
    assert archived.metadata.status == "archived"
    assert archived.relative_path.startswith("Archive/")
    assert service.search("BM25", folder="Research") == []
    assert len(service.search("BM25", folder="Archive")) == 1

    assert service.delete_note(note.metadata.note_id) is True
    assert not archived.path.exists()
    assert service.search("BM25") == []
    assert service.delete_note(note.metadata.note_id) is False


def test_inbox_capture_and_crash_safe_processing_recovery(tmp_path: Path) -> None:
    service = KnowledgeService(
        vault_dir=tmp_path / "vault",
        index_path=tmp_path / "memory" / "index" / "knowledge.db",
    )

    captured = service.capture_to_inbox(
        content="Investigate local PySide6 event loop integration with asyncio.",
        tags=["desktop", "ui"],
    )
    assert captured.metadata.folder == "Inbox"
    assert captured.metadata.note_type == "inbox"
    assert captured.metadata.status == "unprocessed"
    assert len(service.list_inbox()) == 1

    # Simulate classifier crash during Inbox triage -> captured item MUST stay intact
    def crashing_classifier(_doc):
        raise RuntimeError("Simulated LLM classifier outage")

    with pytest.raises(RuntimeError, match="Simulated LLM classifier outage"):
        service.process_inbox_item(captured.metadata.note_id, classifier_fn=crashing_classifier)

    still_in_inbox = service.get_note(captured.metadata.note_id)
    assert still_in_inbox is not None
    assert still_in_inbox.path.exists()
    assert still_in_inbox.metadata.folder == "Inbox"
    assert "PySide6 event loop" in still_in_inbox.body
    assert len(service.search("PySide6", folder="Inbox")) == 1

    # Successful processing later promotes note cleanly
    promoted = service.process_inbox_item(
        captured.metadata.note_id,
        target_folder="Research",
        new_title="PySide6 Asyncio Integration",
        new_note_type="research",
    )
    assert promoted.metadata.folder == "Research"
    assert promoted.metadata.status == "active"
    assert promoted.metadata.note_type == "research"
    assert service.list_inbox() == []


def test_disposable_index_rebuild_and_fts_punctuation_safety(tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    index_path = tmp_path / "memory" / "index" / "knowledge.db"

    service = KnowledgeService(vault_dir=vault_dir, index_path=index_path)
    note = service.create_note(
        title="C++ & Python Async Patterns",
        content="Use structured concurrency for C++ coroutines and asyncio tasks.",
        folder="Notes",
        note_type="reference",
        tags=["cpp", "asyncio"],
    )

    # Tricky punctuation / boolean keywords must not trigger FTS syntax errors
    punct_hits = service.search("C++ asyncio: * (tasks)")
    assert len(punct_hits) == 1
    assert punct_hits[0].note_id == note.metadata.note_id
    # Raw boolean tokens treated safely as literal terms, never throwing OperationalError
    assert isinstance(service.search("asyncio AND OR NOT * ()"), list)

    # Delete the SQLite index file completely and re-create service from disk truth
    assert index_path.exists()
    index_path.unlink()
    assert not index_path.exists()

    rebuilt_service = KnowledgeService(vault_dir=vault_dir, index_path=index_path)
    assert index_path.exists()
    rebuilt_hits = rebuilt_service.search(
        query="coroutines",
        tags=["cpp"],
        note_type="reference",
        folder="Notes",
    )
    assert len(rebuilt_hits) == 1
    assert rebuilt_hits[0].title == "C++ & Python Async Patterns"


def test_strict_separation_from_conversation_archives(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)

    # Record conversational turns in archive
    archive = ConversationArchive(archive_dir=cfg.archive_dir, persona_name="IRIS")
    archive.open_session()
    archive.write_turn("user", "What is the capital of Japan?")
    archive.write_turn("assistant", "Tokyo is the capital of Japan.")
    archive.close_session()

    # Initialize knowledge vault separately
    service = KnowledgeService(vault_dir=cfg.vault_dir, index_path=cfg.knowledge_index_path)
    service.create_note(
        title="Renewable Microgrid Note",
        content="Battery storage sizing notes for microgrid pilots.",
        folder="Research",
    )

    # Conversation archive text must NOT leak into Knowledge Vault search
    assert service.search("Tokyo") == []
    # Vault notes must NOT appear in conversation session files
    session_files = archive.get_session_files()
    assert len(session_files) == 1
    assert "Microgrid" not in session_files[0].read_text(encoding="utf-8")


def test_explicit_save_only_invariant_and_router_integration(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    service = KnowledgeService(vault_dir=cfg.vault_dir, index_path=cfg.knowledge_index_path)
    memory_adapter = VaultMemoryAdapter(service)

    # 1. VaultMemoryAdapter.store_turn() MUST NOT auto-create notes from normal turns
    memory_adapter.store_turn(
        user_text="My dentist appointment is next Tuesday.",
        assistant_text="Got it, I can keep track of our conversation.",
    )
    assert list(cfg.vault_dir.rglob("*.md")) == []

    # 2. Ordinary dialogue & questions MUST NOT trigger KNOWLEDGE_SAVE
    for casual in [
        "Tell me something interesting about black holes",
        "Note that gravitational lensing bends light — can you explain why?",
        "What is 14 * 7?",
    ]:
        decision = classify_request(casual, knowledge_service=service, prior_context="Prior turn")
        assert decision.route != RouteType.KNOWLEDGE_SAVE
    assert list(cfg.vault_dir.rglob("*.md")) == []

    # 3. Ambiguous save with no content/context does not invent details
    empty_decision = classify_request("Save this as a note", knowledge_service=service)
    assert empty_decision.route == RouteType.KNOWLEDGE_SAVE
    assert "What exact information" in (empty_decision.direct_response or "")
    assert list(cfg.vault_dir.rglob("*.md")) == []

    # 4. Explicit inline save & Inbox capture succeed deterministically
    save_decision = classify_request(
        "Save this as a note: Renewable battery target is 40 kWh.",
        knowledge_service=service,
    )
    assert save_decision.route == RouteType.KNOWLEDGE_SAVE
    assert "Saved permanent note" in (save_decision.direct_response or "")
    assert len(service.search("40 kWh", folder="Notes")) == 1

    inbox_decision = classify_request(
        "Capture this to my inbox: Review quarterly sensor calibration",
        knowledge_service=service,
    )
    assert inbox_decision.route == RouteType.KNOWLEDGE_SAVE
    assert "Inbox" in (inbox_decision.direct_response or "")
    assert len(service.list_inbox()) == 1

    # 5. Explicit search retrieves permanent note cleanly; session queries hit archive
    search_decision = classify_request("Search my notes for calibration", knowledge_service=service)
    assert search_decision.route == RouteType.KNOWLEDGE_SEARCH
    assert "Knowledge Vault" in (search_decision.direct_response or "")
    assert "calibration" in (search_decision.direct_response or "").lower()

    archive = ConversationArchive(archive_dir=cfg.archive_dir, persona_name="IRIS")
    archive.open_session()
    archive.write_turn("user", "Tell me about Mars")
    archive.write_turn("assistant", "Mars is the fourth planet.")
    archive.close_session()
    mem_decision = classify_request(
        "What did we talk about in our last session?",
        archive_dir=cfg.archive_dir,
        knowledge_service=service,
    )
    assert mem_decision.route == RouteType.MEMORY
    assert "Tell me about Mars" in (mem_decision.direct_response or "")


def test_cloud_router_and_assistant_wiring(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, openrouter_api_key="test-openrouter")
    service = KnowledgeService(vault_dir=cfg.vault_dir, index_path=cfg.knowledge_index_path)

    mock_openrouter = MagicMock()
    mock_openrouter.model_name = "openrouter/free"
    mock_openrouter.generate_response.return_value = "A normal chat reply."

    router = CloudRouterModelAdapter(
        config=cfg,
        openrouter_adapter=mock_openrouter,
        knowledge_service=service,
    )

    # Normal conversation uses OpenRouter and creates 0 notes
    chat_reply = router.generate_response([{"role": "user", "content": "Hello IRIS!"}])
    assert chat_reply == "A normal chat reply."
    assert router.last_route == RouteType.CONVERSATION.value
    assert list(cfg.vault_dir.rglob("*.md")) == []

    # Explicit save referencing prior assistant message captures exact text
    save_reply = router.generate_response(
        [
            {"role": "user", "content": "Summarize key point"},
            {"role": "assistant", "content": "Always isolate permanent notes from transcripts."},
            {"role": "user", "content": "Add this to my notes"},
        ]
    )
    assert router.last_route == RouteType.KNOWLEDGE_SAVE.value
    assert "Saved permanent note" in save_reply
    assert len(service.search("isolate permanent notes")) == 1

    # Explicit search through CloudRouter
    search_reply = router.generate_response(
        [{"role": "user", "content": "Search my notes for isolate"}]
    )
    assert router.last_route == RouteType.KNOWLEDGE_SEARCH.value
    assert "Knowledge Vault" in search_reply

    # create_model_adapter & IRISAssistant wiring smoke test
    wired_adapter = create_model_adapter(cfg, knowledge_service=service)
    assert isinstance(wired_adapter, CloudRouterModelAdapter)

    mock_tts = MagicMock()
    mock_stt = MagicMock()
    assistant = IRISAssistant(config=cfg, tts_adapter=mock_tts, stt_adapter=mock_stt)
    assert assistant.knowledge_service.vault_dir == cfg.vault_dir.resolve()
    assert isinstance(assistant.memory, VaultMemoryAdapter)
    assert isinstance(assistant.model, CloudRouterModelAdapter)
    # VaultMemoryAdapter exposes relevant permanent notes on explicit lookup
    assert "Always isolate permanent notes" in assistant.memory.get_relevant_context("isolate")
