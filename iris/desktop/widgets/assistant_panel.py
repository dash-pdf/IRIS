"""
iris/desktop/widgets/assistant_panel.py
---------------------------------------
Right assistant panel connected to the existing `IRISAssistant` backend.
Handles conversation display, message submission, processing indicator,
and sanitized error feedback without duplicating AI orchestration logic.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from iris.desktop.backend import IRISDesktopBackend
from iris.desktop.theme import PALETTE


class AssistantPanel(QFrame):
    """Dedicated right sidebar hosting the IRIS conversational assistant."""

    def __init__(self, backend: IRISDesktopBackend, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self.setObjectName("AssistantPanel")
        self.setMinimumWidth(310)
        self.setMaximumWidth(440)
        self._setup_ui()
        self._hydrate_history()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(10)

        # Header + State Pill
        header = QHBoxLayout()
        title = QLabel("IRIS Assistant")
        title.setObjectName("BrandTitle")
        self.state_pill = QLabel("Idle")
        self.state_pill.setObjectName("HighlightPill")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.state_pill)
        layout.addLayout(header)

        # Explicit-save policy reminder
        note_hint = QLabel("Dialogue saves only to session logs unless you explicitly ask to save a note.")
        note_hint.setObjectName("MutedLabel")
        note_hint.setWordWrap(True)
        layout.addWidget(note_hint)

        # Error Banner (hidden unless error occurs)
        self.error_banner = QLabel("")
        self.error_banner.setObjectName("ErrorBanner")
        self.error_banner.setWordWrap(True)
        self.error_banner.setVisible(False)
        layout.addWidget(self.error_banner)

        # Conversation History Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(2, 4, 2, 4)
        self.history_layout.setSpacing(8)
        self.history_layout.addStretch()
        self.scroll_area.setWidget(self.history_container)
        layout.addWidget(self.scroll_area, stretch=1)

        # Processing Progress Indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate when visible
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Input Row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask IRIS or give a command...")
        self.message_input.returnPressed.connect(self._on_send_clicked)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("PrimaryButton")
        self.send_btn.clicked.connect(self._on_send_clicked)

        input_row.addWidget(self.message_input, stretch=1)
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

    def _hydrate_history(self) -> None:
        for msg in self._backend.get_restored_messages():
            self.append_message(msg["role"], msg["content"])

    def _on_send_clicked(self) -> None:
        text = self.message_input.text().strip()
        if not text:
            return
        self.clear_error()
        if self._backend.send_user_message(text):
            self.message_input.clear()

    def append_message(self, role: str, content: str) -> None:
        """Append a styled user or IRIS message bubble to conversation history."""
        is_user = role == "user"
        bubble = QFrame()
        bg = PALETTE["bg_elevated"] if is_user else "rgba(128, 87, 217, 0.12)"
        border = PALETTE["border_strong"] if is_user else "rgba(182, 154, 255, 0.35)"
        bubble.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: 1px solid {border}; border-radius: 8px; }}"
        )

        b_layout = QVBoxLayout(bubble)
        b_layout.setContentsMargins(10, 8, 10, 8)
        b_layout.setSpacing(4)

        sender = QLabel("You" if is_user else "IRIS")
        sender.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {PALETTE['highlight'] if not is_user else PALETTE['text_secondary']}; border: none; background: transparent;"
        )
        body = QLabel(content)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setStyleSheet("border: none; background: transparent;")

        b_layout.addWidget(sender)
        b_layout.addWidget(body)

        # Insert before trailing stretch
        insert_idx = max(0, self.history_layout.count() - 1)
        self.history_layout.insertWidget(insert_idx, bubble)
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def set_processing(self, busy: bool, status_text: str | None = None) -> None:
        self.progress_bar.setVisible(busy)
        self.send_btn.setEnabled(not busy)
        self.message_input.setEnabled(not busy)
        if status_text:
            self.state_pill.setText(status_text)
        elif not busy:
            self.state_pill.setText("Idle")

    def show_error(self, message: str) -> None:
        self.error_banner.setText(message)
        self.error_banner.setVisible(True)

    def clear_error(self) -> None:
        self.error_banner.clear()
        self.error_banner.setVisible(False)
