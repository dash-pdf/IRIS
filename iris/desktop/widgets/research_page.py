"""
iris/desktop/widgets/research_page.py
-------------------------------------
Modular Research & Video Intelligence workspace container (Phase 2 boundary).

Prepares the tabbed architecture and `ResearchWorkspaceBridge` integration for
future Phase 4 capabilities (embedded web browser + mandatory Video Intelligence)
without implementing live scraping, browsing, or video engines in Phase 2.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from iris.desktop.backend import IRISDesktopBackend


class ResearchPage(QWidget):
    """Modular container reserved for Phase 4 Research Engine & Video Intelligence."""

    def __init__(self, backend: IRISDesktopBackend, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Research & Video Intelligence")
        title.setObjectName("PageTitle")
        sub = QLabel(
            "Planned Phase 4 workspace — modular architecture boundary connected to Knowledge Vault."
        )
        sub.setObjectName("MutedLabel")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        top.addLayout(title_box, stretch=1)

        badge = QLabel("Not Yet Implemented · Phase 4")
        badge.setObjectName("StatusWarningPill")
        top.addWidget(badge)
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_web_research_tab(), "Web Research (Planned)")
        self.tabs.addTab(self._build_video_intelligence_tab(), "Video Intelligence (Planned)")
        root.addWidget(self.tabs, stretch=1)

    def _build_web_research_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        card = QFrame()
        card.setObjectName("SurfaceCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(8)

        header = QLabel("EMBEDDED RESEARCH WORKSPACE ARCHITECTURE")
        header.setObjectName("SectionHeader")
        card_layout.addWidget(header)

        details = QLabel(
            "• Multi-tab webpage research & structured content extraction (Phase 4).\n"
            "• Source comparison with preserved URLs and access dates via ResearchWorkspaceBridge.\n"
            "• Explicit-save security boundary: external webpage content is always flagged "
            "untrusted and never auto-written into permanent Knowledge Vault notes."
        )
        details.setWordWrap(True)
        details.setObjectName("MutedLabel")
        card_layout.addWidget(details)

        bridge_status = QLabel(
            f"Bridge Ready: {self._backend.research_bridge.CAPABILITY.display_name} "
            f"(Live Engine Active: {self._backend.research_bridge.is_research_engine_implemented})"
        )
        bridge_status.setObjectName("HighlightPill")
        card_layout.addWidget(bridge_status)

        layout.addWidget(card)
        layout.addStretch()
        return tab

    def _build_video_intelligence_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        card = QFrame()
        card.setObjectName("SurfaceCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(8)

        header = QLabel("MANDATORY VIDEO INTELLIGENCE COMPATIBILITY (PHASE 4)")
        header.setObjectName("SectionHeader")
        card_layout.addWidget(header)

        details = QLabel(
            "Designed contracts in `iris.services.research` support future Phase 4 workflows:\n"
            "• Discovering educational videos/lectures and reading permitted captions or transcripts.\n"
            "• Extracting key concepts, examples, structured study notes, linked timestamps, "
            "and revision flashcards.\n"
            "• Comparing multiple video lectures on the same subject.\n"
            "• Explicit-instruction Vault export to `vault/Research/` preserving video title, "
            "URL, and timestamps — never fabricating inaccessible content."
        )
        details.setWordWrap(True)
        details.setObjectName("MutedLabel")
        card_layout.addWidget(details)

        layout.addWidget(card)
        layout.addStretch()
        return tab
