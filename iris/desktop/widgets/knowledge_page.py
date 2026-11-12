"""
iris/desktop/widgets/knowledge_page.py
--------------------------------------
Full Knowledge Vault workspace connected deterministically to `KnowledgeService`.

Supports:
- Canonical folder browsing (`All`, `Inbox`, `Notes`, `Projects`, `Research`, `Archive`)
- Full-text SQLite FTS5 search + tag filter + manual index rebuild
- Tabbed note viewing & editing (`Edit` Markdown source + `Preview` rendered Markdown)
- Creating, editing, saving, moving, Inbox triage, archiving, and deleting notes
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from iris.desktop.backend import IRISDesktopBackend
from iris.services.knowledge.frontmatter import normalize_tags
from iris.services.knowledge.models import VALID_NOTE_TYPES, VAULT_FOLDERS, NoteDocument


class NoteEditorTab(QWidget):
    """Single tabbed Markdown viewer/editor bound to a Vault `NoteDocument`."""

    note_saved = Signal(str)
    note_removed = Signal(str)
    title_changed = Signal(str, str)  # (note_id, display_title)

    def __init__(
        self,
        backend: IRISDesktopBackend,
        doc: NoteDocument,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend
        self.note_id = doc.metadata.note_id
        self._doc = doc
        self._dirty = False
        self._loading = False
        self._setup_ui()
        self._load_doc(doc)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Top Metadata Bar
        meta_row = QHBoxLayout()
        meta_row.setSpacing(8)

        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Note title...")
        self.title_input.textChanged.connect(self._mark_dirty)

        self.folder_combo = QComboBox()
        self.folder_combo.addItems(list(VAULT_FOLDERS))

        self.type_combo = QComboBox()
        self.type_combo.addItems(list(VALID_NOTE_TYPES))
        self.type_combo.currentTextChanged.connect(self._mark_dirty)

        meta_row.addWidget(self.title_input, stretch=3)
        meta_row.addWidget(self.folder_combo, stretch=1)
        meta_row.addWidget(self.type_combo, stretch=1)
        layout.addLayout(meta_row)

        # Tags, Source & Action Toolbar
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("Tags (comma-separated, e.g. python, research)")
        self.tags_input.textChanged.connect(self._mark_dirty)

        self.mode_btn = QPushButton("Preview Markdown")
        self.mode_btn.clicked.connect(self._toggle_view_mode)

        self.save_btn = QPushButton("Save Note")
        self.save_btn.setObjectName("PrimaryButton")
        self.save_btn.clicked.connect(self.save_note)

        self.move_btn = QPushButton("Move Folder")
        self.move_btn.clicked.connect(self._move_to_selected_folder)

        self.archive_btn = QPushButton("Archive")
        self.archive_btn.clicked.connect(self._archive_note)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setObjectName("DangerButton")
        self.delete_btn.clicked.connect(self._delete_note)

        action_row.addWidget(self.tags_input, stretch=1)
        action_row.addWidget(self.mode_btn)
        action_row.addWidget(self.move_btn)
        action_row.addWidget(self.archive_btn)
        action_row.addWidget(self.delete_btn)
        action_row.addWidget(self.save_btn)
        layout.addLayout(action_row)

        # Editor / Markdown Preview Stack
        self.view_stack = QStackedWidget()

        self.editor = QPlainTextEdit()
        self.editor.setObjectName("MarkdownEditor")
        self.editor.setPlaceholderText("Write structured Markdown content here...")
        self.editor.textChanged.connect(self._mark_dirty)

        self.preview = QTextBrowser()
        self.preview.setObjectName("MarkdownPreview")
        self.preview.setOpenExternalLinks(True)

        self.view_stack.addWidget(self.editor)   # index 0 = Edit
        self.view_stack.addWidget(self.preview)  # index 1 = Preview
        layout.addWidget(self.view_stack, stretch=1)

        # Footer Info Bar
        footer_row = QHBoxLayout()
        self.path_label = QLabel("")
        self.path_label.setObjectName("MutedLabel")
        self.status_label = QLabel("Saved on disk")
        self.status_label.setObjectName("MutedLabel")
        footer_row.addWidget(self.path_label, stretch=1)
        footer_row.addWidget(self.status_label)
        layout.addLayout(footer_row)

    def _load_doc(self, doc: NoteDocument) -> None:
        self._loading = True
        self._doc = doc
        self.title_input.setText(doc.metadata.title)
        self.folder_combo.setCurrentText(doc.metadata.folder)
        self.type_combo.setCurrentText(doc.metadata.note_type)
        self.tags_input.setText(", ".join(doc.metadata.tags))
        self.editor.setPlainText(doc.body)
        self.preview.setMarkdown(doc.body)
        src = f"  ·  Source: {doc.metadata.source_url}" if doc.metadata.source_url else ""
        self.path_label.setText(f"{doc.relative_path}  ·  Updated {doc.metadata.updated_at[:19]}{src}")
        self._dirty = False
        self._loading = False
        self.status_label.setText("Saved on disk")
        self.title_changed.emit(self.note_id, doc.metadata.title)

    def _mark_dirty(self, *_args: object) -> None:
        if self._loading:
            return
        self._dirty = True
        self.status_label.setText("Unsaved changes")
        self.title_changed.emit(self.note_id, f"{self.title_input.text().strip() or 'Untitled'} *")

    def _toggle_view_mode(self) -> None:
        if self.view_stack.currentIndex() == 0:
            self.preview.setMarkdown(self.editor.toPlainText())
            self.view_stack.setCurrentIndex(1)
            self.mode_btn.setText("Edit Markdown")
        else:
            self.view_stack.setCurrentIndex(0)
            self.mode_btn.setText("Preview Markdown")

    def save_note(self) -> bool:
        """Persist current editor state atomically via `KnowledgeService`."""
        raw_tags = [t.strip() for t in self.tags_input.text().split(",") if t.strip()]
        try:
            updated = self._backend.update_note(
                self.note_id,
                title=self.title_input.text().strip() or self._doc.metadata.title,
                content=self.editor.toPlainText(),
                tags=normalize_tags(raw_tags),
                note_type=self.type_combo.currentText(),
            )
            target_folder = self.folder_combo.currentText()
            if target_folder != updated.metadata.folder:
                updated = self._backend.move_note(self.note_id, target_folder)
            self._load_doc(updated)
            self.status_label.setText("Saved atomically to Markdown")
            self.note_saved.emit(self.note_id)
            return True
        except Exception as exc:
            self.status_label.setText(f"Save error: {exc}")
            return False

    def _move_to_selected_folder(self) -> None:
        target = self.folder_combo.currentText()
        if target == self._doc.metadata.folder:
            return
        try:
            moved = self._backend.move_note(self.note_id, target)
            self._load_doc(moved)
            self.status_label.setText(f"Moved to {target}/")
            self.note_saved.emit(self.note_id)
        except Exception as exc:
            self.status_label.setText(f"Move failed: {exc}")

    def _archive_note(self) -> None:
        try:
            archived = self._backend.archive_note(self.note_id)
            self._load_doc(archived)
            self.status_label.setText("Archived in Archive/")
            self.note_saved.emit(self.note_id)
        except Exception as exc:
            self.status_label.setText(f"Archive failed: {exc}")

    def _delete_note(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete Note",
            f"Permanently delete '{self._doc.metadata.title}' from the Knowledge Vault?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            if self._backend.delete_note(self.note_id):
                self.note_removed.emit(self.note_id)
        except Exception as exc:
            self.status_label.setText(f"Delete failed: {exc}")


class KnowledgePage(QWidget):
    """Three-part Knowledge Workspace: search/filter bar, note list + Inbox triage, tabbed editor."""

    def __init__(self, backend: IRISDesktopBackend, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._active_folder: str | None = None
        self._open_tabs: dict[str, NoteEditorTab] = {}
        self._setup_ui()
        self.refresh_notes_list()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ── Top Search & Creation Toolbar ─────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.folder_filter = QComboBox()
        self.folder_filter.addItem("All Folders", None)
        for folder in VAULT_FOLDERS:
            self.folder_filter.addItem(folder, folder)
        self.folder_filter.currentIndexChanged.connect(self._on_filter_changed)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search Knowledge Vault (full-text SQLite FTS5)...")
        self.search_input.textChanged.connect(self.refresh_notes_list)

        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Tag filter...")
        self.tag_input.setMaximumWidth(140)
        self.tag_input.textChanged.connect(self.refresh_notes_list)

        self.new_note_btn = QPushButton("+ New Note")
        self.new_note_btn.setObjectName("PrimaryButton")
        self.new_note_btn.clicked.connect(self.prompt_new_note)

        self.capture_inbox_btn = QPushButton("Capture Inbox")
        self.capture_inbox_btn.clicked.connect(self.prompt_inbox_capture)

        self.rebuild_btn = QPushButton("Rebuild Index")
        self.rebuild_btn.clicked.connect(self._backend.rebuild_index_async)

        toolbar.addWidget(self.folder_filter)
        toolbar.addWidget(self.search_input, stretch=2)
        toolbar.addWidget(self.tag_input, stretch=1)
        toolbar.addWidget(self.capture_inbox_btn)
        toolbar.addWidget(self.new_note_btn)
        toolbar.addWidget(self.rebuild_btn)
        root.addLayout(toolbar)

        # ── Inbox Triage Banner (visible when Inbox selected) ─────────────────
        self.inbox_bar = QFrame()
        self.inbox_bar.setObjectName("SurfaceCard")
        inbox_layout = QHBoxLayout(self.inbox_bar)
        inbox_layout.setContentsMargins(12, 8, 12, 8)
        inbox_layout.setSpacing(8)

        inbox_lbl = QLabel("Inbox Triage — Promote selected capture:")
        inbox_lbl.setObjectName("HighlightPill")
        inbox_layout.addWidget(inbox_lbl)

        for target in ("Notes", "Projects", "Research", "Archive"):
            btn = QPushButton(f"Move -> {target}")
            btn.clicked.connect(lambda _c=False, t=target: self._promote_selected_inbox(t))
            inbox_layout.addWidget(btn)
        inbox_layout.addStretch()
        self.inbox_bar.setVisible(False)
        root.addWidget(self.inbox_bar)

        # ── Splitter: Note Results List | Tabbed Note Editor ──────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(6)

        self.results_header = QLabel("NOTES (0)")
        self.results_header.setObjectName("SectionHeader")
        self.note_list = QListWidget()
        self.note_list.itemClicked.connect(self._on_note_item_clicked)
        self.note_list.itemActivated.connect(self._on_note_item_clicked)

        left_layout.addWidget(self.results_header)
        left_layout.addWidget(self.note_list, stretch=1)
        splitter.addWidget(left_pane)

        # Tabbed Editor Area
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)

        self.empty_editor = QLabel(
            "Select a note from the list to view or edit it,\n"
            "or click '+ New Note' to create a Markdown note in the Vault."
        )
        self.empty_editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_editor.setObjectName("MutedLabel")

        self.editor_stack = QStackedWidget()
        self.editor_stack.addWidget(self.empty_editor)
        self.editor_stack.addWidget(self.tab_widget)
        splitter.addWidget(self.editor_stack)

        splitter.setSizes([280, 620])
        root.addWidget(splitter, stretch=1)

    def select_folder(self, folder: str | None) -> None:
        self.search_input.blockSignals(True)
        self.tag_input.blockSignals(True)
        self.search_input.clear()
        self.tag_input.clear()
        self.search_input.blockSignals(False)
        self.tag_input.blockSignals(False)

        idx = self.folder_filter.findData(folder)
        if idx >= 0:
            self.folder_filter.setCurrentIndex(idx)
        self._active_folder = folder
        self.inbox_bar.setVisible(folder == "Inbox")
        self.refresh_notes_list()

    def apply_search(self, query: str) -> None:
        self.folder_filter.setCurrentIndex(0)
        self.search_input.setText(query)
        self.refresh_notes_list()

    def _on_filter_changed(self) -> None:
        self._active_folder = self.folder_filter.currentData()
        self.inbox_bar.setVisible(self._active_folder == "Inbox")
        self.refresh_notes_list()

    def refresh_notes_list(self) -> None:
        query = self.search_input.text().strip()
        raw_tag = self.tag_input.text().strip()
        tags = [t.strip() for t in raw_tag.split(",") if t.strip()] if raw_tag else None

        results = self._backend.search_notes(
            query=query,
            tags=tags,
            folder=self._active_folder,
            limit=200,
        )
        self.results_header.setText(f"NOTES ({len(results)})")
        self.note_list.clear()

        if not results:
            empty = QListWidgetItem("No matching Markdown notes found.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.note_list.addItem(empty)
            return

        for res in results:
            tag_str = f"  [{', '.join(res.tags)}]" if res.tags else ""
            item = QListWidgetItem(f"{res.title}\n{res.folder}{tag_str}")
            item.setData(Qt.ItemDataRole.UserRole, res.note_id)
            self.note_list.addItem(item)

    def _on_note_item_clicked(self, item: QListWidgetItem) -> None:
        note_id = item.data(Qt.ItemDataRole.UserRole)
        if note_id:
            self.open_note(str(note_id))

    def open_note(self, identifier_or_path: str) -> None:
        doc = self._backend.get_note(identifier_or_path)
        if doc is None:
            return
        note_id = doc.metadata.note_id
        if note_id in self._open_tabs:
            tab = self._open_tabs[note_id]
            self.tab_widget.setCurrentWidget(tab)
            self.editor_stack.setCurrentIndex(1)
            return

        tab = NoteEditorTab(self._backend, doc)
        tab.note_saved.connect(lambda _nid: self.refresh_notes_list())
        tab.note_removed.connect(self._on_note_deleted)
        tab.title_changed.connect(self._update_tab_title)

        self._open_tabs[note_id] = tab
        idx = self.tab_widget.addTab(tab, doc.metadata.title[:28])
        self.tab_widget.setCurrentIndex(idx)
        self.editor_stack.setCurrentIndex(1)

    def _update_tab_title(self, note_id: str, new_title: str) -> None:
        tab = self._open_tabs.get(note_id)
        if tab is not None:
            idx = self.tab_widget.indexOf(tab)
            if idx >= 0:
                self.tab_widget.setTabText(idx, new_title[:28])

    def _close_tab(self, index: int) -> None:
        widget = self.tab_widget.widget(index)
        if isinstance(widget, NoteEditorTab):
            self._open_tabs.pop(widget.note_id, None)
            self.tab_widget.removeTab(index)
            widget.deleteLater()
        if self.tab_widget.count() == 0:
            self.editor_stack.setCurrentIndex(0)

    def _on_note_deleted(self, note_id: str) -> None:
        tab = self._open_tabs.get(note_id)
        if tab is not None:
            idx = self.tab_widget.indexOf(tab)
            if idx >= 0:
                self._close_tab(idx)
        self.refresh_notes_list()

    def prompt_new_note(self) -> None:
        title, ok = QInputDialog.getText(self, "New Markdown Note", "Note title:")
        if not ok or not title.strip():
            return
        folder = self._active_folder or "Notes"
        note_type = (
            "project" if folder == "Projects"
            else "research" if folder == "Research"
            else "inbox" if folder == "Inbox"
            else "note"
        )
        doc = self._backend.create_note(
            title=title.strip(),
            content=f"# {title.strip()}\n\n",
            folder=folder,
            note_type=note_type,
        )
        self.refresh_notes_list()
        self.open_note(doc.metadata.note_id)

    def prompt_inbox_capture(self) -> None:
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Capture to Inbox",
            "Enter raw note or idea to store safely in vault/Inbox/:",
        )
        if not ok or not text.strip():
            return
        doc = self._backend.capture_to_inbox(content=text.strip())
        self.select_folder("Inbox")
        self.open_note(doc.metadata.note_id)

    def _promote_selected_inbox(self, target_folder: str) -> None:
        current_item = self.note_list.currentItem()
        if current_item is None:
            return
        note_id = current_item.data(Qt.ItemDataRole.UserRole)
        if not note_id:
            return
        try:
            self._backend.process_inbox_item(str(note_id), target_folder=target_folder)
            self.refresh_notes_list()
        except Exception as exc:
            QMessageBox.warning(self, "Inbox Triage Failed", str(exc))
