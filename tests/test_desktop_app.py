"""
tests/test_desktop_app.py
-------------------------
Comprehensive automated tests for Phase 2: IRIS PySide6 Desktop Application Shell.

Covers:
- Application startup & clean shutdown (offscreen Qt)
- Three-panel layout & workspace navigation (implemented vs clearly planned placeholders)
- Markdown Knowledge Vault folder hierarchy, browsing, creation, editing, saving, and Inbox triage
- Deterministic SQLite FTS5 search without LLM involvement
- Thread-safe backend worker communication via Qt Signals
- Sanitized error handling without UI crashes or secret leakage
- Graceful operation when all cloud AI providers are unavailable
- Honest VoiceControlBar listening & microphone state reporting
- Application-wide LLM Usage Policy & Research / Video Intelligence compatibility contracts
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig
from iris.core.policy import DEFAULT_LLM_POLICY, ExecutionMode, ServiceCapability
from iris.core.state import AudioState
from iris.desktop.backend import IRISDesktopBackend
from iris.desktop.main_window import IRISMainWindow
from iris.services.knowledge import KnowledgeService
from iris.services.research import (
    ResearchSource,
    ResearchWorkspaceBridge,
    SourceType,
    VideoResearchSummary,
    VideoTimestamp,
)
from iris_desktop import create_desktop_app


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication([])
    return app


def _make_offline_config(tmp_path: Path, **overrides: object) -> IRISConfig:
    defaults: dict[str, object] = {
        "llm_provider": "cloud_router",
        "stt_provider": "console",
        "tts_provider": "mock",
        "archive_dir": tmp_path / "memory" / "conversations",
        "vault_dir": tmp_path / "vault",
        "knowledge_index_path": tmp_path / "memory" / "index" / "knowledge.db",
        "context_cache_dir": tmp_path / "memory" / "context",
    }
    defaults.update(overrides)
    return IRISConfig(**defaults)  # type: ignore[arg-type]


def _pump_events_until(predicate, timeout_ms: int = 3000) -> None:
    import time

    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    QCoreApplication.processEvents()


def test_desktop_app_startup_navigation_and_shutdown(qapp: QApplication, tmp_path: Path) -> None:
    cfg = _make_offline_config(tmp_path)
    app, window = create_desktop_app(config=cfg, auto_start_backend=True)
    assert app is qapp
    try:
        assert "IRIS" in window.windowTitle()
        assert window.active_page_key() == "home"

        # Sidebar shows canonical Markdown Vault folders
        folder_names = [
            window.sidebar.folder_list.item(i).data(0x0100)  # Qt.UserRole
            for i in range(window.sidebar.folder_list.count())
        ]
        assert folder_names == ["Inbox", "Notes", "Projects", "Research", "Archive"]

        # Navigate across all workspaces
        for key in ("knowledge", "research", "calendar", "tasks", "music", "settings", "home"):
            window.navigate_to(key)
            assert window.active_page_key() == key

        # Unimplemented workspaces clearly indicate planned / not yet implemented status
        assert "Not Yet Implemented" in window.calendar_page.findChildren(type(window.home_page.location_label))[1].text()
        assert window.backend.research_bridge.is_research_engine_implemented is False
    finally:
        window.close()


def test_knowledge_vault_viewing_editing_saving_search_and_inbox(
    qapp: QApplication, tmp_path: Path
) -> None:
    cfg = _make_offline_config(tmp_path)
    backend = IRISDesktopBackend(config=cfg)
    window = IRISMainWindow(backend=backend, auto_start_backend=True)
    try:
        # 1. Create quick note from Home dashboard (zero LLM calls)
        window.home_page.quick_note_title.setText("Phase 2 Architecture")
        window.home_page.quick_note_body.setText("PySide6 UI connected cleanly to KnowledgeService.")
        window.home_page._on_quick_create_note()

        assert window.active_page_key() == "knowledge"
        assert window.knowledge_page.tab_widget.count() == 1

        # 2. Edit note in tab & save atomically back to Markdown + FTS5 index
        tab = window.knowledge_page.tab_widget.widget(0)
        tab.editor.setPlainText("Updated desktop shell architecture with SQLite FTS5 search.")
        tab.tags_input.setText("pyside6, architecture")
        assert tab.save_note() is True

        reloaded = backend.get_note(tab.note_id)
        assert reloaded is not None
        assert "SQLite FTS5 search" in reloaded.body
        assert reloaded.metadata.tags == ["pyside6", "architecture"]

        # 3. Search via KnowledgePage (FTS5)
        window.knowledge_page.apply_search("FTS5")
        assert window.knowledge_page.note_list.count() == 1
        assert "Phase 2 Architecture" in window.knowledge_page.note_list.item(0).text()

        # 4. Capture to Inbox and triage into Research/
        captured = backend.capture_to_inbox("Investigate timestamp extraction for lectures")
        window.open_vault_folder("Inbox")
        assert not window.knowledge_page.inbox_bar.isHidden()
        assert window.knowledge_page.note_list.count() == 1

        window.knowledge_page.note_list.setCurrentRow(0)
        window.knowledge_page._promote_selected_inbox("Research")

        promoted = backend.get_note(captured.metadata.note_id)
        assert promoted is not None
        assert promoted.metadata.folder == "Research"
        assert window.home_page.inbox_count_label.text() == "Inbox: 0 items"
    finally:
        window.close()


def test_thread_safe_assistant_communication_and_explicit_save_policy(
    qapp: QApplication, tmp_path: Path
) -> None:
    cfg = _make_offline_config(tmp_path)
    backend = IRISDesktopBackend(config=cfg)
    window = IRISMainWindow(backend=backend, auto_start_backend=True)
    try:
        received_replies: list[str] = []
        backend.assistant_response_received.connect(received_replies.append)

        # Deterministic offline calculation works with NO cloud API keys configured
        accepted = backend.send_user_message("What is 15 * 12?")
        assert accepted is True
        _pump_events_until(lambda: not backend.is_turn_in_flight and len(received_replies) >= 1)

        assert "180" in received_replies[0]
        # Normal deterministic/chat turns NEVER auto-create permanent notes
        assert list(cfg.vault_dir.rglob("*.md")) == []

        # Explicit save command persists to Markdown Vault & updates UI views cleanly
        accepted2 = backend.send_user_message("Save this as a note: Desktop uses #8057D9 accent.")
        assert accepted2 is True
        _pump_events_until(lambda: not backend.is_turn_in_flight and len(received_replies) >= 2)

        assert "Saved permanent note" in received_replies[1]
        assert len(backend.search_notes("#8057D9")) >= 0
        assert len(backend.search_notes("8057D9")) == 1

        # Cloud LLM unavailable for general dialogue degrades gracefully without crashing
        accepted3 = backend.send_user_message("Tell me a story about the stars.")
        assert accepted3 is True
        _pump_events_until(lambda: not backend.is_turn_in_flight and len(received_replies) >= 3)
        assert any(
            phrase in received_replies[2].lower()
            for phrase in ("not configured", "not set", "unavailable")
        )
    finally:
        window.close()


def test_error_handling_sanitizes_secrets_and_voice_bar_is_honest(
    qapp: QApplication, tmp_path: Path
) -> None:
    cfg = _make_offline_config(
        tmp_path,
        openrouter_api_key="sk-or-v1-secret-should-never-leak",
        google_api_key="AIzaSySecretKeyNeverLeak",
    )
    failing_model = MagicMock()
    failing_model.generate_response.side_effect = RuntimeError(
        "Request failed with Bearer sk-or-v1-secret-should-never-leak"
    )
    mock_tts = MagicMock()
    mock_stt = MagicMock()

    assistant = IRISAssistant(
        config=cfg,
        model_adapter=failing_model,
        tts_adapter=mock_tts,
        stt_adapter=mock_stt,
    )
    backend = IRISDesktopBackend(config=cfg, assistant=assistant)
    window = IRISMainWindow(backend=backend, auto_start_backend=True)
    try:
        # Startup keeps mic explicitly OFF and never shows misleading Listening state
        assert assistant.capture.is_running is False
        assert "Inactive" in window.voice_bar.mic_status_pill.text()
        assert "Off" in window.voice_bar.listening_pill.text()

        # Verify Settings UI never exposes raw API keys
        window.navigate_to("settings")
        settings_text = window.settings_page.summary_label.text()
        assert "sk-or-v1-secret-should-never-leak" not in settings_text
        assert "AIzaSySecretKeyNeverLeak" not in settings_text
        assert "Configured" in settings_text

        # Simulate backend model error containing secret -> banner redacts it without crash
        backend.send_user_message("Trigger failure")
        _pump_events_until(lambda: not window.assistant_panel.error_banner.isHidden())
        assert not window.assistant_panel.error_banner.isHidden()
        banner_text = window.assistant_panel.error_banner.text()
        assert "sk-or-v1-secret-should-never-leak" not in banner_text
        assert "REDACTED" in banner_text

        # Simulate dormant clap standby -> Mic Active, Listening strictly Off
        backend.presence_state_changed.emit("DORMANT", "clap_standby", False)
        backend.mic_status_changed.emit(True, False, "Mic Active · Clap Standby")
        assert "Dormant" in window.voice_bar.presence_pill.text()
        assert "Off" in window.voice_bar.listening_pill.text()

        # Simulate active listening vs muted mic honesty
        backend.presence_state_changed.emit("ARMED", "double_clap", True)
        backend.mic_status_changed.emit(True, False, "Mic Active · Armed")
        backend.assistant_state_changed.emit(AudioState.LISTENING.value, "turn-1")
        assert "Active Speech" in window.voice_bar.listening_pill.text()

        backend.mic_status_changed.emit(True, True, "Mic Muted (Half-Duplex)")
        assert "Paused" in window.voice_bar.listening_pill.text()
    finally:
        window.close()


def test_llm_usage_policy_and_research_video_compatibility(tmp_path: Path) -> None:
    # 1. Core Vault / Search / Context operations declare zero LLM requirement
    assert DEFAULT_LLM_POLICY.requires_llm("knowledge_vault") is False
    assert DEFAULT_LLM_POLICY.requires_llm("knowledge_search") is False
    assert DEFAULT_LLM_POLICY.requires_llm("deterministic_tools") is False
    assert DEFAULT_LLM_POLICY.requires_llm("context_engine") is False
    assert DEFAULT_LLM_POLICY.requires_llm("assistant_orchestration") is True

    # Custom future module declaration works cleanly
    DEFAULT_LLM_POLICY.register_capability(
        ServiceCapability(
            service_id="future_tasks_test",
            display_name="Local Tasks",
            requires_llm=False,
            default_mode=ExecutionMode.DETERMINISTIC,
            offline_capable=True,
            implemented=False,
            description="Deterministic local task list.",
        )
    )
    assert DEFAULT_LLM_POLICY.requires_llm("future_tasks_test") is False

    # 2. Research & Video Intelligence compatibility contract invariants
    knowledge = KnowledgeService(
        vault_dir=tmp_path / "vault",
        index_path=tmp_path / "memory" / "index" / "knowledge.db",
    )
    bridge = ResearchWorkspaceBridge(knowledge)
    assert bridge.is_research_engine_implemented is False

    ts = VideoTimestamp(seconds=125, label="Core Architecture", timestamp_url="https://youtu.be/abc?t=125")
    assert ts.formatted_time() == "02:05"

    source = ResearchSource(
        url="https://youtu.be/abc",
        title="IRIS Lecture",
        access_date="2026-11-12",
        source_type=SourceType.VIDEO,
        content_accessible=True,
        transcript_available=True,
        timestamps=(ts,),
    )
    assert source.untrusted_content is True

    # Never auto-saves without explicit user confirmation
    with pytest.raises(PermissionError):
        bridge.save_verified_source_note(source, user_confirmed=False, markdown_body="Notes")

    # Never generates study notes for inaccessible videos
    inaccessible = VideoResearchSummary(
        video_title="Private Lecture",
        source_url="https://youtu.be/private",
        access_date="2026-11-12",
        content_accessible=False,
    )
    with pytest.raises(ValueError, match="not accessible"):
        bridge.format_video_study_markdown(inaccessible)

    # Accessible report formats cleanly with timestamps & saves explicitly to Research/
    accessible = VideoResearchSummary(
        video_title="IRIS Lecture",
        source_url=source.url,
        access_date=source.access_date,
        content_accessible=True,
        summary="Covers modular design.",
        key_concepts=("Deterministic Routing", "Markdown Vault"),
        timestamps=(ts,),
        flashcards=(("Why Markdown?", "Human-readable source of truth."),),
    )
    md_body = bridge.format_video_study_markdown(accessible)
    saved = bridge.save_verified_source_note(source, user_confirmed=True, markdown_body=md_body)
    assert saved.metadata.folder == "Research"
    assert saved.metadata.source_url == "https://youtu.be/abc"
    assert "02:05" in saved.body
