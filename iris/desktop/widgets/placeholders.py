"""
iris/desktop/widgets/placeholders.py
------------------------------------
Placeholder workspaces for future integrations (`Calendar`, `Tasks`, `Music`)
and the live non-secret `Settings` & LLM Usage Policy view.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from iris.desktop.backend import IRISDesktopBackend


class PlaceholderWorkspace(QWidget):
    """Clearly labeled future workspace placeholder."""

    def __init__(
        self,
        title: str,
        phase_label: str,
        description: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(14)

        top = QHBoxLayout()
        hdr = QLabel(title)
        hdr.setObjectName("PageTitle")
        pill = QLabel(f"Not Yet Implemented · {phase_label}")
        pill.setObjectName("StatusWarningPill")
        top.addWidget(hdr)
        top.addStretch()
        top.addWidget(pill)
        root.addLayout(top)

        card = QFrame()
        card.setObjectName("SurfaceCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(8)

        info_hdr = QLabel("PLANNED MODULE")
        info_hdr.setObjectName("SectionHeader")
        body = QLabel(description)
        body.setWordWrap(True)
        body.setObjectName("MutedLabel")

        card_layout.addWidget(info_hdr)
        card_layout.addWidget(body)
        root.addWidget(card)
        root.addStretch()


class SettingsPage(QWidget):
    """Non-secret configuration viewer, voice reply option, and LLM Usage Policy inspector."""

    def __init__(self, backend: IRISDesktopBackend, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QLabel("Settings & Application Policy")
        title.setObjectName("PageTitle")
        root.addWidget(title)

        # Runtime Settings Card (Secrets NEVER shown)
        cfg_card = QFrame()
        cfg_card.setObjectName("SurfaceCard")
        cfg_layout = QVBoxLayout(cfg_card)
        cfg_layout.setContentsMargins(16, 14, 16, 14)
        cfg_layout.setSpacing(8)

        cfg_hdr = QLabel("ACTIVE CONFIGURATION (SECRETS REDACTED)")
        cfg_hdr.setObjectName("SectionHeader")
        cfg_layout.addWidget(cfg_hdr)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setObjectName("MutedLabel")
        cfg_layout.addWidget(self.summary_label)

        controls_row = QHBoxLayout()
        self.speak_checkbox = QCheckBox("Speak typed assistant replies aloud (TTS)")
        self.speak_checkbox.setChecked(self._backend.speak_text_replies)
        self.speak_checkbox.toggled.connect(self._backend.set_speak_text_replies)

        self.refresh_ctx_btn = QPushButton("Refresh Environment Context")
        self.refresh_ctx_btn.clicked.connect(lambda: self._backend.refresh_context_async(force=True))

        controls_row.addWidget(self.speak_checkbox)
        controls_row.addStretch()
        controls_row.addWidget(self.refresh_ctx_btn)
        cfg_layout.addLayout(controls_row)

        root.addWidget(cfg_card)

        # LLM Usage Policy Matrix
        policy_card = QFrame()
        policy_card.setObjectName("SurfaceCard")
        policy_layout = QVBoxLayout(policy_card)
        policy_layout.setContentsMargins(16, 14, 16, 14)
        policy_layout.setSpacing(8)

        policy_hdr = QLabel("APPLICATION-WIDE LLM USAGE POLICY")
        policy_hdr.setObjectName("SectionHeader")
        policy_desc = QLabel(
            "IRIS uses deterministic Python & SQLite for ordinary application tasks and "
            "routes LLM requests exclusively through `IRISAssistant` -> `ModelAdapter`."
        )
        policy_desc.setWordWrap(True)
        policy_desc.setObjectName("MutedLabel")

        self.policy_table = QTableWidget(0, 5)
        self.policy_table.setHorizontalHeaderLabels(
            ["Service", "Requires LLM", "Execution Mode", "Offline Ready", "Status"]
        )
        self.policy_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.policy_table.verticalHeader().setVisible(False)
        self.policy_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        policy_layout.addWidget(policy_hdr)
        policy_layout.addWidget(policy_desc)
        policy_layout.addWidget(self.policy_table, stretch=1)
        root.addWidget(policy_card, stretch=1)

        self.refresh_view()

    def refresh_view(self) -> None:
        info = self._backend.get_non_secret_settings_summary()
        provider_badges = ", ".join(
            f"{name}: {'Configured' if ok else 'Not Configured'}"
            for name, ok in info["providers_configured"].items()
        )
        self.summary_label.setText(
            f"Intelligence Provider: {info['llm_provider']} ({info['llm_model_name']})\n"
            f"TTS Provider: {info['tts_provider']}   |   STT Provider: {info['stt_provider']}\n"
            f"Knowledge Vault: {info['vault_dir']}\n"
            f"SQLite FTS5 Index: {info['knowledge_index_path']}\n"
            f"Session Archives: {info['archive_dir']}\n"
            f"Credentials Status: {provider_badges}"
        )

        caps = self._backend.policy.list_capabilities()
        self.policy_table.setRowCount(len(caps))
        for row, cap in enumerate(caps):
            self.policy_table.setItem(row, 0, QTableWidgetItem(cap.display_name))
            self.policy_table.setItem(row, 1, QTableWidgetItem("Yes" if cap.requires_llm else "No"))
            self.policy_table.setItem(row, 2, QTableWidgetItem(cap.default_mode.value))
            self.policy_table.setItem(row, 3, QTableWidgetItem("Yes" if cap.offline_capable else "No"))
            self.policy_table.setItem(
                row, 4, QTableWidgetItem("Implemented" if cap.implemented else "Planned")
            )
