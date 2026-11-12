"""
iris/desktop/backend.py
-----------------------
Thread-safe Qt backend bridge connecting the PySide6 desktop interface to the
existing IRIS backend (`IRISAssistant`, `KnowledgeService`, `ContextManager`,
and modular `ResearchWorkspaceBridge`).

Architectural invariants:
- No second model router: all AI calls flow strictly through `IRISAssistant`.
- UI contains zero API credentials or provider SDK imports.
- Deterministic Knowledge Vault operations run in Python/SQLite with zero LLM calls.
- Long-running operations (assistant turns, context refreshes, index rebuilds,
  microphone activation) execute off the GUI thread and marshal back via Qt Signals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig, load_config
from iris.core.policy import DEFAULT_LLM_POLICY, LLMUsagePolicy
from iris.core.router import RouteType
from iris.core.security import sanitize_error_message
from iris.core.state import AudioState
from iris.services.context.context_manager import ContextManager
from iris.services.context.models import IRISContextSnapshot
from iris.services.knowledge.models import NoteDocument, SearchResult
from iris.services.knowledge.service import InboxClassifierFn, KnowledgeService
from iris.services.research import ResearchWorkspaceBridge

logger = logging.getLogger(__name__)


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _FunctionWorker(QRunnable):
    """Runs a blocking callable inside `QThreadPool` and emits Qt signals on completion."""

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._fn = fn
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = self._fn()
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.failed.emit(sanitize_error_message(exc))


class IRISDesktopBackend(QObject):
    """
    Central coordinator between PySide6 widgets and IRIS core services.
    """

    assistant_state_changed = Signal(str, str)  # (state_value, turn_id)
    user_message_recorded = Signal(str)
    assistant_response_received = Signal(str)
    assistant_turn_started = Signal()
    assistant_turn_finished = Signal()
    assistant_error = Signal(str)

    mic_status_changed = Signal(bool, bool, str)  # (is_running, is_muted, status_text)
    silence_warning_received = Signal()
    echo_detected_received = Signal(str, str)

    vault_updated = Signal()
    context_updated = Signal(object)  # IRISContextSnapshot
    policy_notice = Signal(str)

    def __init__(
        self,
        config: IRISConfig | None = None,
        assistant: IRISAssistant | None = None,
        policy: LLMUsagePolicy | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config or load_config(require_voice=False)
        self.policy = policy or DEFAULT_LLM_POLICY
        self._thread_pool = QThreadPool.globalInstance()
        self._active_workers: set[_FunctionWorker] = set()
        self._shutting_down = False
        self._turn_in_flight = False
        self._speak_text_replies = False

        if assistant is not None:
            self.assistant = assistant
            self._wire_assistant_callbacks()
        else:
            self.assistant = IRISAssistant(
                config=self.config,
                on_state_change=self._on_assistant_state_change,
                on_user_transcript=self._on_user_transcript,
                on_assistant_response=self._on_assistant_response,
                on_echo_detected=self._on_echo_detected,
                on_error=self._on_assistant_error,
                on_silence_warning=self._on_silence_warning,
            )

        self.knowledge_service: KnowledgeService = self.assistant.knowledge_service
        self.context_manager: ContextManager = self.assistant.context_manager
        self.research_bridge = ResearchWorkspaceBridge(self.knowledge_service)

    # ── Lifecycle & Thread Safety ─────────────────────────────────────────────

    def start(self, refresh_context_async: bool = True) -> None:
        """
        Start IRIS backend services safely without blocking the UI thread or
        activating the microphone unexpectedly.
        """
        try:
            self.assistant.start(capture_audio=False, initial_context_refresh=False)
        except Exception as exc:
            safe_msg = sanitize_error_message(exc)
            logger.warning("Desktop backend startup warning: %s", safe_msg)
            self.assistant_error.emit(safe_msg)

        self.emit_mic_status()
        self.context_updated.emit(self.context_manager.get_context())

        if refresh_context_async:
            self.refresh_context_async()

    def shutdown(self) -> None:
        """Stop IRIS services and clean up worker threads cleanly."""
        self._shutting_down = True
        try:
            self.assistant.stop()
        except Exception as exc:
            logger.debug("Assistant stop notice: %s", sanitize_error_message(exc))
        self._thread_pool.waitForDone(2000)

    @property
    def speak_text_replies(self) -> bool:
        return self._speak_text_replies

    def set_speak_text_replies(self, enabled: bool) -> None:
        self._speak_text_replies = bool(enabled)

    @property
    def is_turn_in_flight(self) -> bool:
        return self._turn_in_flight

    # ── Assistant Callbacks (called from worker threads -> emit Qt Signals) ───

    def _wire_assistant_callbacks(self) -> None:
        self.assistant.on_state_change = self._on_assistant_state_change
        self.assistant.on_user_transcript = self._on_user_transcript
        self.assistant.on_assistant_response = self._on_assistant_response
        self.assistant.on_echo_detected = self._on_echo_detected
        self.assistant.on_error = self._on_assistant_error
        self.assistant.on_silence_warning = self._on_silence_warning

    def _safe_emit(self, signal: Signal, *args: Any) -> None:
        if self._shutting_down:
            return
        try:
            signal.emit(*args)
        except RuntimeError:
            pass

    def _on_assistant_state_change(self, _old: AudioState, new_state: AudioState, turn_id: str) -> None:
        self._safe_emit(self.assistant_state_changed, new_state.value, turn_id)
        self.emit_mic_status()

    def _on_user_transcript(self, text: str) -> None:
        self._safe_emit(self.user_message_recorded, text)

    def _on_assistant_response(self, text: str) -> None:
        self._safe_emit(self.assistant_response_received, text)
        route = getattr(self.assistant.model, "last_route", None)
        if route == RouteType.KNOWLEDGE_SAVE.value:
            self._safe_emit(self.vault_updated)

    def _on_echo_detected(self, transcript: str, reason: str) -> None:
        self._safe_emit(self.echo_detected_received, transcript, reason)

    def _on_assistant_error(self, exc: Exception) -> None:
        self._safe_emit(self.assistant_error, sanitize_error_message(exc))

    def _on_silence_warning(self) -> None:
        self._safe_emit(self.silence_warning_received)

    # ── Worker Spawning ───────────────────────────────────────────────────────

    def _submit_worker(
        self,
        fn: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        worker = _FunctionWorker(fn)
        self._active_workers.add(worker)

        def _cleanup_and_succeed(result: Any) -> None:
            self._active_workers.discard(worker)
            if not self._shutting_down and on_success is not None:
                on_success(result)

        def _cleanup_and_fail(err: str) -> None:
            self._active_workers.discard(worker)
            if not self._shutting_down and on_error is not None:
                on_error(err)

        worker.signals.finished.connect(_cleanup_and_succeed)
        worker.signals.failed.connect(_cleanup_and_fail)
        self._thread_pool.start(worker)

    # ── Assistant Interaction (Thread-Safe) ───────────────────────────────────

    def send_user_message(self, text: str) -> bool:
        """
        Dispatch a user message to `IRISAssistant.process_text_turn()` in a background worker.
        Returns True if accepted, False if empty or a turn is already running.
        """
        clean = text.strip()
        if not clean or self._turn_in_flight:
            return False

        self._turn_in_flight = True
        self._safe_emit(self.assistant_turn_started)
        speak_now = self._speak_text_replies

        def _run_turn() -> str:
            return self.assistant.process_text_turn(
                clean,
                wait_for_playback=False,
                speak=speak_now,
            )

        def _on_done(_reply: str) -> None:
            self._turn_in_flight = False
            self.emit_mic_status()
            self._safe_emit(self.assistant_turn_finished)

        def _on_err(safe_err: str) -> None:
            self._turn_in_flight = False
            self.emit_mic_status()
            self._safe_emit(self.assistant_error, safe_err)
            self._safe_emit(self.assistant_turn_finished)

        self._submit_worker(_run_turn, on_success=_on_done, on_error=_on_err)
        return True

    def get_restored_messages(self) -> list[dict[str, str]]:
        """Return recent user/assistant conversation messages already loaded in memory."""
        return [
            m for m in self.assistant.conversation.get_messages()
            if m.get("role") in {"user", "assistant"} and m.get("content", "").strip()
        ]

    # ── Microphone Controls (Strict Honest Status) ────────────────────────────

    def current_mic_status(self) -> tuple[bool, bool, str]:
        """Return (is_running, is_muted, human_status_label)."""
        running = self.assistant.capture.is_running
        muted = self.assistant.capture.is_muted
        if not running:
            return False, muted, "Microphone Inactive"
        if muted:
            return True, True, "Mic Muted (Half-Duplex)"
        return True, False, "Microphone Active"

    def emit_mic_status(self) -> None:
        running, muted, label = self.current_mic_status()
        self._safe_emit(self.mic_status_changed, running, muted, label)

    def toggle_microphone(self) -> None:
        """Safely start or stop existing voice capture off the GUI thread."""
        if self.assistant.capture.is_running:
            self._submit_worker(
                self.assistant.disable_microphone,
                on_success=lambda _: self.emit_mic_status(),
                on_error=self._handle_mic_error,
            )
        else:
            self._submit_worker(
                self.assistant.enable_microphone,
                on_success=lambda _: self.emit_mic_status(),
                on_error=self._handle_mic_error,
            )

    def _handle_mic_error(self, safe_err: str) -> None:
        self.emit_mic_status()
        self._safe_emit(self.assistant_error, f"Voice capture error: {safe_err}")

    # ── Context Engine ────────────────────────────────────────────────────────

    def get_context_snapshot(self) -> IRISContextSnapshot:
        return self.context_manager.get_context()

    def refresh_context_async(self, force: bool = False) -> None:
        self._submit_worker(
            lambda: self.context_manager.refresh(force=force),
            on_success=lambda snap: self._safe_emit(self.context_updated, snap),
            on_error=lambda err: logger.warning("Context refresh failed: %s", err),
        )

    # ── Deterministic Knowledge Vault Operations (Zero LLM) ───────────────────

    def search_notes(
        self,
        query: str = "",
        tags: list[str] | None = None,
        note_type: str | None = None,
        folder: str | None = None,
        limit: int = 100,
    ) -> list[SearchResult]:
        return self.knowledge_service.search(
            query=query,
            tags=tags,
            note_type=note_type,
            folder=folder,
            limit=limit,
        )

    def get_recent_notes(self, limit: int = 8) -> list[SearchResult]:
        return self.knowledge_service.search(query="", limit=limit)

    def get_folder_counts(self) -> dict[str, int]:
        return self.knowledge_service.get_folder_counts()

    def get_note(self, identifier_or_path: str | Path) -> NoteDocument | None:
        return self.knowledge_service.get_note(identifier_or_path)

    def create_note(
        self,
        title: str,
        content: str,
        folder: str = "Notes",
        note_type: str = "note",
        tags: list[str] | None = None,
        source_url: str = "",
    ) -> NoteDocument:
        doc = self.knowledge_service.create_note(
            title=title,
            content=content,
            folder=folder,
            note_type=note_type,
            tags=tags,
            source_url=source_url,
        )
        self._safe_emit(self.vault_updated)
        return doc

    def update_note(
        self,
        identifier_or_path: str | Path,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        note_type: str | None = None,
        source_url: str | None = None,
    ) -> NoteDocument:
        doc = self.knowledge_service.update_note(
            identifier_or_path=identifier_or_path,
            title=title,
            content=content,
            tags=tags,
            note_type=note_type,
            source_url=source_url,
        )
        self._safe_emit(self.vault_updated)
        return doc

    def move_note(self, identifier_or_path: str | Path, target_folder: str) -> NoteDocument:
        doc = self.knowledge_service.move_note(identifier_or_path, target_folder)
        self._safe_emit(self.vault_updated)
        return doc

    def archive_note(self, identifier_or_path: str | Path) -> NoteDocument:
        doc = self.knowledge_service.archive_note(identifier_or_path)
        self._safe_emit(self.vault_updated)
        return doc

    def delete_note(self, identifier_or_path: str | Path) -> bool:
        deleted = self.knowledge_service.delete_note(identifier_or_path)
        if deleted:
            self._safe_emit(self.vault_updated)
        return deleted

    def capture_to_inbox(
        self,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
        source_url: str = "",
    ) -> NoteDocument:
        doc = self.knowledge_service.capture_to_inbox(
            content=content,
            title=title,
            tags=tags,
            source_url=source_url,
        )
        self._safe_emit(self.vault_updated)
        return doc

    def process_inbox_item(
        self,
        identifier_or_path: str | Path,
        target_folder: str = "Notes",
        new_title: str | None = None,
        new_tags: list[str] | None = None,
        new_note_type: str | None = None,
        classifier_fn: InboxClassifierFn | None = None,
    ) -> NoteDocument:
        doc = self.knowledge_service.process_inbox_item(
            identifier_or_path=identifier_or_path,
            target_folder=target_folder,
            new_title=new_title,
            new_tags=new_tags,
            new_note_type=new_note_type,
            classifier_fn=classifier_fn,
        )
        self._safe_emit(self.vault_updated)
        return doc

    def rebuild_index_async(self) -> None:
        def _do_rebuild() -> int:
            return self.knowledge_service.rebuild_index()

        self._submit_worker(
            _do_rebuild,
            on_success=lambda _count: self._safe_emit(self.vault_updated),
            on_error=lambda err: self._safe_emit(self.assistant_error, f"Index rebuild failed: {err}"),
        )

    # ── Non-Secret Configuration Summary for Settings UI ──────────────────────

    @staticmethod
    def _has_configured_secret(value: str) -> bool:
        stripped = (value or "").strip()
        return bool(stripped) and not stripped.startswith("your_")

    def get_non_secret_settings_summary(self) -> dict[str, Any]:
        """Return safe runtime settings for display in Settings (never returns API keys)."""
        return {
            "llm_provider": self.config.llm_provider,
            "llm_model_name": self.config.llm_model_name,
            "tts_provider": self.config.tts_provider,
            "stt_provider": self.config.stt_provider,
            "vault_dir": str(self.config.vault_dir),
            "archive_dir": str(self.config.archive_dir),
            "knowledge_index_path": str(self.config.knowledge_index_path),
            "speak_text_replies": self._speak_text_replies,
            "providers_configured": {
                "OpenRouter": self._has_configured_secret(self.config.openrouter_api_key),
                "Google Gemini": self._has_configured_secret(self.config.google_api_key),
                "ElevenLabs": self._has_configured_secret(self.config.elevenlabs_api_key),
                "OpenAI": self._has_configured_secret(self.config.openai_api_key),
                "Anthropic": self._has_configured_secret(self.config.anthropic_api_key),
            },
        }
