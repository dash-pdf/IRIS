"""
iris/desktop/widgets/sidebar.py
-------------------------------
Left navigation sidebar and live Knowledge Vault folder explorer.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from iris.services.knowledge.models import VAULT_FOLDERS

NAV_ITEMS: tuple[tuple[str, str, bool], ...] = (
    ("home", "Home", True),
    ("knowledge", "Knowledge", True),
    ("research", "Research", False),
    ("calendar", "Calendar", False),
    ("tasks", "Tasks", False),
    ("music", "Music", False),
    ("settings", "Settings", True),
)


class SidebarWidget(QFrame):
    """
    Left navigation panel featuring workspace navigation and the live
    Markdown Vault directory hierarchy.
    """

    page_selected = Signal(str)
    vault_folder_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SidebarPanel")
        self.setMinimumWidth(220)
        self.setMaximumWidth(275)
        self._nav_buttons: dict[str, QPushButton] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(10)

        # IRIS Identity Header
        brand_box = QVBoxLayout()
        brand_box.setSpacing(2)
        title = QLabel("IRIS")
        title.setObjectName("BrandTitle")
        subtitle = QLabel("Intelligent Research &\nInformation System")
        subtitle.setObjectName("BrandSubtitle")
        subtitle.setWordWrap(True)
        brand_box.addWidget(title)
        brand_box.addWidget(subtitle)
        layout.addLayout(brand_box)

        layout.addSpacing(6)

        nav_label = QLabel("Workspaces")
        nav_label.setObjectName("SectionHeader")
        layout.addWidget(nav_label)

        for key, label, implemented in NAV_ITEMS:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            btn_text = label if implemented else f"{label}  · Planned"
            btn = QPushButton(btn_text)
            btn.setObjectName("NavButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("active", "true" if key == "home" else "false")
            btn.clicked.connect(lambda _checked=False, k=key: self._on_nav_clicked(k))
            self._nav_buttons[key] = btn
            row_layout.addWidget(btn)
            layout.addWidget(row)

        layout.addSpacing(10)

        vault_header = QLabel("Markdown Vault")
        vault_header.setObjectName("SectionHeader")
        layout.addWidget(vault_header)

        self.folder_list = QListWidget()
        self.folder_list.setObjectName("VaultFolderList")
        self.folder_list.itemClicked.connect(self._on_folder_clicked)
        layout.addWidget(self.folder_list, stretch=1)

        self.update_folder_counts({})

        phase_pill = QLabel("Phase 2 · Desktop Shell")
        phase_pill.setObjectName("HighlightPill")
        phase_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(phase_pill)

    def _on_nav_clicked(self, key: str) -> None:
        self.set_active_page(key)
        self.page_selected.emit(key)

    def _on_folder_clicked(self, item: QListWidgetItem) -> None:
        folder_name = item.data(Qt.ItemDataRole.UserRole)
        if folder_name:
            self.set_active_page("knowledge")
            self.vault_folder_selected.emit(str(folder_name))

    def set_active_page(self, key: str) -> None:
        for btn_key, btn in self._nav_buttons.items():
            btn.setProperty("active", "true" if btn_key == key else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def update_folder_counts(self, counts: dict[str, int]) -> None:
        """Refresh canonical Vault folder entries (`Inbox` .. `Archive`) with note counts."""
        current_folder = None
        if self.folder_list.currentItem() is not None:
            current_folder = self.folder_list.currentItem().data(Qt.ItemDataRole.UserRole)

        self.folder_list.clear()
        for folder in VAULT_FOLDERS:
            count = counts.get(folder, 0)
            item = QListWidgetItem(f"{folder}   ({count})")
            item.setData(Qt.ItemDataRole.UserRole, folder)
            self.folder_list.addItem(item)
            if folder == current_folder:
                self.folder_list.setCurrentItem(item)
