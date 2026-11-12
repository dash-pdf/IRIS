"""
iris/desktop/widgets/voice_bar.py
---------------------------------
Persistent bottom voice control bar.

Honesty invariant:
- Never displays an active Listening indicator when microphone capture is inactive or muted.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from iris.desktop.backend import IRISDesktopBackend


class VoiceControlBar(QFrame):
    """Persistent footer bar housing microphone toggle, real status pills, and quick text command."""

    def __init__(self, backend: IRISDesktopBackend, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self.setObjectName("VoiceControlBar")
        self.setFixedHeight(60)
        self._mic_running = False
        self._mic_muted = True
        self._assistant_state = "idle"
        self._presence_state = "disabled"
        self._stt_listening_enabled = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        self.mic_btn = QPushButton("Mic: Off")
        self.mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_btn.clicked.connect(self._backend.toggle_microphone)

        self.standby_btn = QPushButton("Standby")
        self.standby_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.standby_btn.setToolTip("Enter local double-clap standby (no cloud STT)")
        self.standby_btn.clicked.connect(self._backend.enable_dormant_standby)

        shortcut = getattr(self._backend.config, "presence_shortcut", "Ctrl+Shift+I")
        self.arm_btn = QPushButton("Arm")
        self.arm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.arm_btn.setToolTip(f"Toggle Armed listening vs Dormant standby ({shortcut})")
        self.arm_btn.clicked.connect(self._backend.toggle_armed_or_dormant)

        self.disable_mic_btn = QPushButton("Disable Mic")
        self.disable_mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.disable_mic_btn.setToolTip("Stop all microphone capture completely")
        self.disable_mic_btn.clicked.connect(self._backend.disable_microphone)

        self.presence_pill = QLabel("Presence: Disabled")
        self.presence_pill.setObjectName("StatusInactivePill")

        self.mic_status_pill = QLabel("Microphone Inactive")
        self.mic_status_pill.setObjectName("StatusInactivePill")

        self.listening_pill = QLabel("Listening: Off")
        self.listening_pill.setObjectName("StatusInactivePill")

        self.activity_pill = QLabel("Assistant: Idle")
        self.activity_pill.setObjectName("HighlightPill")

        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Quick voice-bar text command (e.g. 'Search my notes for ...')...")
        self.command_input.returnPressed.connect(self._submit_quick_command)

        self.command_btn = QPushButton("Run")
        self.command_btn.clicked.connect(self._submit_quick_command)

        layout.addWidget(self.mic_btn)
        layout.addWidget(self.standby_btn)
        layout.addWidget(self.arm_btn)
        layout.addWidget(self.disable_mic_btn)
        layout.addWidget(self.presence_pill)
        layout.addWidget(self.mic_status_pill)
        layout.addWidget(self.listening_pill)
        layout.addWidget(self.activity_pill)
        layout.addWidget(self.command_input, stretch=1)
        layout.addWidget(self.command_btn)

    def _submit_quick_command(self) -> None:
        text = self.command_input.text().strip()
        if not text:
            return
        if self._backend.send_user_message(text):
            self.command_input.clear()

    def update_presence_state(
        self,
        presence_state: str,
        reason: str = "",
        stt_listening_enabled: bool = False,
    ) -> None:
        normalized = presence_state.strip().lower()
        self._presence_state = normalized
        self._stt_listening_enabled = bool(stt_listening_enabled)

        labels = {
            "disabled": ("Presence: Disabled", "StatusInactivePill"),
            "dormant": ("Presence: Dormant (Clap Standby)", "StatusInactivePill"),
            "armed": ("Presence: Armed (Listening)", "StatusSuccessPill"),
            "processing": ("Presence: Processing", "StatusWarningPill"),
            "speaking": ("Presence: Speaking", "StatusWarningPill"),
            "error": ("Presence: Error", "StatusWarningPill"),
        }
        text, pill_name = labels.get(normalized, (f"Presence: {normalized.title()}", "StatusInactivePill"))
        self.presence_pill.setText(text)
        self.presence_pill.setObjectName(pill_name)
        if reason:
            self.presence_pill.setToolTip(f"Last transition: {reason}")

        if normalized == "armed":
            self.arm_btn.setText("Pause to Standby")
        else:
            self.arm_btn.setText("Arm")

        self._refresh_listening_indicator()
        self._repolish(self.presence_pill, self.arm_btn)

    def update_mic_status(self, is_running: bool, is_muted: bool, label: str) -> None:
        self._mic_running = is_running
        self._mic_muted = is_muted

        if is_running and not is_muted and self._presence_state == "dormant":
            self.mic_btn.setText("Mic: Standby")
            self.mic_btn.setObjectName("")
            self.mic_status_pill.setObjectName("StatusInactivePill")
        elif is_running and not is_muted:
            self.mic_btn.setText("Mic: On")
            self.mic_btn.setObjectName("MicButtonActive")
            self.mic_status_pill.setObjectName("StatusSuccessPill")
        elif is_running and is_muted:
            self.mic_btn.setText("Mic: Muted")
            self.mic_btn.setObjectName("")
            self.mic_status_pill.setObjectName("StatusWarningPill")
        else:
            self.mic_btn.setText("Mic: Off")
            self.mic_btn.setObjectName("")
            self.mic_status_pill.setObjectName("StatusInactivePill")

        self.mic_status_pill.setText(label)
        self._refresh_listening_indicator()
        self._repolish(self.mic_btn, self.mic_status_pill)

    def update_assistant_state(self, state_value: str) -> None:
        self._assistant_state = state_value.lower()
        pretty = state_value.replace("_", " ").title()
        self.activity_pill.setText(f"Assistant: {pretty}")
        self._refresh_listening_indicator()

    def set_busy(self, busy: bool) -> None:
        self.command_btn.setEnabled(not busy)
        self.command_input.setEnabled(not busy)

    def _refresh_listening_indicator(self) -> None:
        """Strictly ensure Listening indicator is only active when mic is live and armed."""
        if not self._mic_running or self._presence_state == "disabled":
            self.listening_pill.setText("Listening: Off")
            self.listening_pill.setObjectName("StatusInactivePill")
        elif self._presence_state == "dormant":
            self.listening_pill.setText("Listening: Off (Clap Only)")
            self.listening_pill.setObjectName("StatusInactivePill")
        elif self._presence_state == "error":
            self.listening_pill.setText("Listening: Unavailable")
            self.listening_pill.setObjectName("StatusWarningPill")
        elif self._mic_muted or self._presence_state in ("processing", "speaking"):
            self.listening_pill.setText("Listening: Paused (Half-Duplex)")
            self.listening_pill.setObjectName("StatusWarningPill")
        elif self._assistant_state == "listening":
            self.listening_pill.setText("Listening: Active Speech")
            self.listening_pill.setObjectName("StatusSuccessPill")
        else:
            self.listening_pill.setText("Listening: Ready")
            self.listening_pill.setObjectName("StatusSuccessPill")
        self._repolish(self.listening_pill)

    @staticmethod
    def _repolish(*widgets: QWidget) -> None:
        for w in widgets:
            w.style().unpolish(w)
            w.style().polish(w)
