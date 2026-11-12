"""
iris/desktop/main_window.py
---------------------------
Main three-panel window for IRIS — Intelligent Research & Information System.

Layout:
- Left: Navigation sidebar (`Home`, `Knowledge`, `Research`, `Calendar`, `Tasks`,
  `Music`, `Settings`) + live Markdown Vault folder tree.
- Centre: Stacked workspace (`HomePage`, `KnowledgePage`, `ResearchPage`,
  future placeholders, `SettingsPage`).
- Right: Dedicated `AssistantPanel`.
- Bottom: Persistent `VoiceControlBar`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from iris.desktop.backend import IRISDesktopBackend
from iris.desktop.theme import build_iris_stylesheet
from iris.desktop.widgets import (
    AssistantPanel,
    HomePage,
    KnowledgePage,
    PlaceholderWorkspace,
    ResearchPage,
    SettingsPage,
    SidebarWidget,
    VoiceControlBar,
)


class IRISMainWindow(QMainWindow):
    """Top-level IRIS PySide6 desktop workspace window."""

    def __init__(
        self,
        backend: IRISDesktopBackend | None = None,
        auto_start_backend: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.backend = backend or IRISDesktopBackend(parent=self)
        self.setWindowTitle("IRIS — Intelligent Research & Information System")
        self.resize(1380, 860)
        self.setMinimumSize(1040, 680)
        self.setStyleSheet(build_iris_stylesheet())

        self._page_indices: dict[str, int] = {}
        self._setup_ui()
        self._connect_signals()

        if auto_start_backend:
            self.backend.start(refresh_context_async=True)
            self.refresh_all_vault_views()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Three-panel main horizontal splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.sidebar = SidebarWidget()
        self.main_splitter.addWidget(self.sidebar)

        self.center_stack = QStackedWidget()
        self.home_page = HomePage(self.backend)
        self.knowledge_page = KnowledgePage(self.backend)
        self.research_page = ResearchPage(self.backend)
        self.calendar_page = PlaceholderWorkspace(
            title="Calendar",
            phase_label="Planned Phase",
            description=(
                "Google Calendar integration is not yet implemented. "
                "This module will connect deterministically to calendar APIs in a future phase."
            ),
        )
        self.tasks_page = PlaceholderWorkspace(
            title="Tasks",
            phase_label="Planned Phase",
            description=(
                "Task management & reminders are not yet implemented. "
                "Local task storage will operate deterministically via Python & SQLite."
            ),
        )
        self.music_page = PlaceholderWorkspace(
            title="Music",
            phase_label="Planned Phase",
            description=(
                "Spotify media controls are not yet implemented. "
                "Playback commands will route directly to deterministic API handlers."
            ),
        )
        self.settings_page = SettingsPage(self.backend)

        pages: tuple[tuple[str, QWidget], ...] = (
            ("home", self.home_page),
            ("knowledge", self.knowledge_page),
            ("research", self.research_page),
            ("calendar", self.calendar_page),
            ("tasks", self.tasks_page),
            ("music", self.music_page),
            ("settings", self.settings_page),
        )
        for key, widget in pages:
            self._page_indices[key] = self.center_stack.addWidget(widget)

        self.main_splitter.addWidget(self.center_stack)

        self.assistant_panel = AssistantPanel(self.backend)
        self.main_splitter.addWidget(self.assistant_panel)

        self.main_splitter.setSizes([240, 780, 360])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        outer.addWidget(self.main_splitter, stretch=1)

        # Persistent Bottom Voice Control Bar
        self.voice_bar = VoiceControlBar(self.backend)
        outer.addWidget(self.voice_bar)

    def _connect_signals(self) -> None:
        # Sidebar navigation
        self.sidebar.page_selected.connect(self.navigate_to)
        self.sidebar.vault_folder_selected.connect(self.open_vault_folder)

        # Home dashboard actions
        self.home_page.open_note_requested.connect(self.open_note_in_knowledge)
        self.home_page.open_folder_requested.connect(self.open_vault_folder)
        self.home_page.search_requested.connect(self.open_vault_search)

        # Backend -> UI signals
        self.backend.vault_updated.connect(self.refresh_all_vault_views)
        self.backend.context_updated.connect(self.home_page.update_context_snapshot)

        self.backend.user_message_recorded.connect(
            lambda msg: self.assistant_panel.append_message("user", msg)
        )
        self.backend.assistant_response_received.connect(
            lambda msg: self.assistant_panel.append_message("assistant", msg)
        )
        self.backend.assistant_turn_started.connect(self._on_turn_started)
        self.backend.assistant_turn_finished.connect(self._on_turn_finished)
        self.backend.assistant_state_changed.connect(self._on_assistant_state_changed)
        self.backend.assistant_error.connect(self.assistant_panel.show_error)

        self.backend.mic_status_changed.connect(self.voice_bar.update_mic_status)
        self.backend.silence_warning_received.connect(
            lambda: self.assistant_panel.show_error(
                "Microphone captures only silence — verify macOS Microphone permission in System Settings."
            )
        )

    def navigate_to(self, page_key: str) -> None:
        """Switch active centre workspace page cleanly."""
        idx = self._page_indices.get(page_key)
        if idx is None:
            return
        self.center_stack.setCurrentIndex(idx)
        self.sidebar.set_active_page(page_key)
        if page_key == "settings":
            self.settings_page.refresh_view()

    def active_page_key(self) -> str:
        current_idx = self.center_stack.currentIndex()
        for key, idx in self._page_indices.items():
            if idx == current_idx:
                return key
        return "home"

    def open_vault_folder(self, folder_name: str) -> None:
        self.navigate_to("knowledge")
        self.knowledge_page.select_folder(folder_name)

    def open_vault_search(self, query: str) -> None:
        self.navigate_to("knowledge")
        self.knowledge_page.apply_search(query)

    def open_note_in_knowledge(self, identifier_or_path: str) -> None:
        self.navigate_to("knowledge")
        self.knowledge_page.open_note(identifier_or_path)

    def refresh_all_vault_views(self) -> None:
        counts = self.backend.get_folder_counts()
        self.sidebar.update_folder_counts(counts)
        self.home_page.refresh_vault_summary()
        self.knowledge_page.refresh_notes_list()

    def _on_turn_started(self) -> None:
        self.assistant_panel.set_processing(True, "Thinking...")
        self.voice_bar.set_busy(True)

    def _on_turn_finished(self) -> None:
        self.assistant_panel.set_processing(False, "Idle")
        self.voice_bar.set_busy(False)

    def _on_assistant_state_changed(self, state_value: str, _turn_id: str) -> None:
        self.voice_bar.update_assistant_state(state_value)
        if state_value in {"thinking", "speaking", "listening"}:
            self.assistant_panel.state_pill.setText(state_value.title())
        elif state_value == "idle" and not self.backend.is_turn_in_flight:
            self.assistant_panel.state_pill.setText("Idle")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.backend.shutdown()
        super().closeEvent(event)
