"""
iris/desktop/widgets/home_page.py
---------------------------------
Home dashboard displaying live context (weather/location), Inbox count,
recently modified notes, quick deterministic note creation, and quick search.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from iris.desktop.backend import IRISDesktopBackend
from iris.services.context.models import IRISContextSnapshot


class HomePage(QWidget):
    """
    Initial IRIS dashboard backed entirely by live Context Engine and
    KnowledgeService data (no hardcoded sample notes).
    """

    open_note_requested = Signal(str)
    open_folder_requested = Signal(str)
    search_requested = Signal(str)

    def __init__(self, backend: IRISDesktopBackend, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # ── Welcome Header ────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)

        welcome = QLabel("IRIS — Intelligent Research & Information System")
        welcome.setObjectName("PageTitle")
        subtitle = QLabel(
            "Local-first Markdown Knowledge Vault & deterministic workspace. "
            "Ordinary note, search, and context operations work offline with zero LLM calls."
        )
        subtitle.setObjectName("MutedLabel")
        subtitle.setWordWrap(True)

        title_col.addWidget(welcome)
        title_col.addWidget(subtitle)
        header_row.addLayout(title_col, stretch=1)

        policy_badge = QLabel("Offline-Ready Policy Active")
        policy_badge.setObjectName("HighlightPill")
        header_row.addWidget(policy_badge, alignment=Qt.AlignmentFlag.AlignTop)
        root.addLayout(header_row)

        # ── Context & Metrics Row ─────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        # Environment Context Card
        self.context_card = QFrame()
        self.context_card.setObjectName("SurfaceCard")
        ctx_layout = QVBoxLayout(self.context_card)
        ctx_layout.setContentsMargins(16, 14, 16, 14)
        ctx_layout.setSpacing(6)

        ctx_top = QHBoxLayout()
        ctx_title = QLabel("ENVIRONMENT CONTEXT")
        ctx_title.setObjectName("SectionHeader")
        self.refresh_ctx_btn = QPushButton("Refresh")
        self.refresh_ctx_btn.clicked.connect(lambda: self._backend.refresh_context_async(force=True))
        ctx_top.addWidget(ctx_title)
        ctx_top.addStretch()
        ctx_top.addWidget(self.refresh_ctx_btn)

        self.location_label = QLabel("Location: Detecting...")
        self.weather_label = QLabel("Weather: Refreshing...")
        self.weather_label.setWordWrap(True)
        self.time_label = QLabel("Local Time: --")
        self.time_label.setObjectName("MutedLabel")

        ctx_layout.addLayout(ctx_top)
        ctx_layout.addWidget(self.location_label)
        ctx_layout.addWidget(self.weather_label)
        ctx_layout.addWidget(self.time_label)
        cards_row.addWidget(self.context_card, stretch=2)

        # Vault Overview / Inbox Count Card
        inbox_card = QFrame()
        inbox_card.setObjectName("SurfaceCard")
        inbox_layout = QVBoxLayout(inbox_card)
        inbox_layout.setContentsMargins(16, 14, 16, 14)
        inbox_layout.setSpacing(6)

        inbox_header = QLabel("VAULT STATUS")
        inbox_header.setObjectName("SectionHeader")
        self.inbox_count_label = QLabel("Inbox: 0 items")
        self.inbox_count_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.total_notes_label = QLabel("Total Indexed Notes: 0")
        self.total_notes_label.setObjectName("MutedLabel")

        self.open_inbox_btn = QPushButton("Review Inbox")
        self.open_inbox_btn.clicked.connect(lambda: self.open_folder_requested.emit("Inbox"))

        inbox_layout.addWidget(inbox_header)
        inbox_layout.addWidget(self.inbox_count_label)
        inbox_layout.addWidget(self.total_notes_label)
        inbox_layout.addStretch()
        inbox_layout.addWidget(self.open_inbox_btn)
        cards_row.addWidget(inbox_card, stretch=1)

        root.addLayout(cards_row)

        # ── Quick Actions Card (Zero-LLM Deterministic) ───────────────────────
        quick_card = QFrame()
        quick_card.setObjectName("SurfaceCard")
        quick_layout = QVBoxLayout(quick_card)
        quick_layout.setContentsMargins(16, 14, 16, 14)
        quick_layout.setSpacing(10)

        quick_header = QLabel("QUICK ACTIONS (DETERMINISTIC PYTHON & SQLITE)")
        quick_header.setObjectName("SectionHeader")
        quick_layout.addWidget(quick_header)

        # Quick Search
        search_row = QHBoxLayout()
        self.quick_search_input = QLineEdit()
        self.quick_search_input.setPlaceholderText("Search Knowledge Vault notes (FTS5)...")
        self.quick_search_input.returnPressed.connect(self._on_quick_search)
        self.quick_search_btn = QPushButton("Search Vault")
        self.quick_search_btn.clicked.connect(self._on_quick_search)
        search_row.addWidget(self.quick_search_input, stretch=1)
        search_row.addWidget(self.quick_search_btn)
        quick_layout.addLayout(search_row)

        # Quick Note Creation
        note_row = QHBoxLayout()
        self.quick_note_title = QLineEdit()
        self.quick_note_title.setPlaceholderText("Quick note title...")
        self.quick_note_body = QLineEdit()
        self.quick_note_body.setPlaceholderText("Note text or idea (saved directly to Markdown)...")
        self.quick_note_body.returnPressed.connect(self._on_quick_create_note)

        self.quick_save_note_btn = QPushButton("Save Note")
        self.quick_save_note_btn.setObjectName("PrimaryButton")
        self.quick_save_note_btn.clicked.connect(self._on_quick_create_note)

        self.quick_capture_inbox_btn = QPushButton("Capture to Inbox")
        self.quick_capture_inbox_btn.clicked.connect(self._on_quick_capture_inbox)

        note_row.addWidget(self.quick_note_title, stretch=1)
        note_row.addWidget(self.quick_note_body, stretch=2)
        note_row.addWidget(self.quick_capture_inbox_btn)
        note_row.addWidget(self.quick_save_note_btn)
        quick_layout.addLayout(note_row)

        self.quick_feedback_label = QLabel("")
        self.quick_feedback_label.setObjectName("MutedLabel")
        quick_layout.addWidget(self.quick_feedback_label)

        root.addWidget(quick_card)

        # ── Recently Modified Notes ───────────────────────────────────────────
        recent_card = QFrame()
        recent_card.setObjectName("SurfaceCard")
        recent_layout = QVBoxLayout(recent_card)
        recent_layout.setContentsMargins(16, 14, 16, 14)
        recent_layout.setSpacing(8)

        recent_header = QLabel("RECENTLY MODIFIED NOTES")
        recent_header.setObjectName("SectionHeader")
        recent_layout.addWidget(recent_header)

        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._on_recent_activated)
        self.recent_list.itemClicked.connect(self._on_recent_activated)
        recent_layout.addWidget(self.recent_list, stretch=1)

        root.addWidget(recent_card, stretch=1)

        self.refresh_vault_summary()
        self.update_context_snapshot(self._backend.get_context_snapshot())

    def _on_quick_search(self) -> None:
        query = self.quick_search_input.text().strip()
        self.search_requested.emit(query)

    def _on_quick_create_note(self) -> None:
        title = self.quick_note_title.text().strip()
        body = self.quick_note_body.text().strip()
        if not title and not body:
            self.quick_feedback_label.setText("Enter a title or content to create a note.")
            return
        try:
            doc = self._backend.create_note(
                title=title or body[:50],
                content=body or f"# {title}\n",
                folder="Notes",
                note_type="note",
            )
            self.quick_note_title.clear()
            self.quick_note_body.clear()
            self.quick_feedback_label.setText(f"Saved note: {doc.relative_path}")
            self.open_note_requested.emit(doc.metadata.note_id)
        except Exception as exc:
            self.quick_feedback_label.setText(f"Could not save note: {exc}")

    def _on_quick_capture_inbox(self) -> None:
        title = self.quick_note_title.text().strip()
        body = self.quick_note_body.text().strip() or title
        if not body:
            self.quick_feedback_label.setText("Enter text to capture into Inbox.")
            return
        try:
            doc = self._backend.capture_to_inbox(content=body, title=title)
            self.quick_note_title.clear()
            self.quick_note_body.clear()
            self.quick_feedback_label.setText(f"Captured to Inbox: {doc.relative_path}")
        except Exception as exc:
            self.quick_feedback_label.setText(f"Inbox capture failed: {exc}")

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        note_id = item.data(Qt.ItemDataRole.UserRole)
        if note_id:
            self.open_note_requested.emit(str(note_id))

    def refresh_vault_summary(self) -> None:
        """Reload real Inbox counts and recent notes from SQLite FTS5."""
        counts = self._backend.get_folder_counts()
        inbox_count = counts.get("Inbox", 0)
        total = sum(counts.values())
        self.inbox_count_label.setText(f"Inbox: {inbox_count} item{'s' if inbox_count != 1 else ''}")
        self.total_notes_label.setText(f"Total Indexed Notes: {total}")

        recent = self._backend.get_recent_notes(limit=8)
        self.recent_list.clear()
        if not recent:
            empty_item = QListWidgetItem(
                "No permanent notes yet. Use Quick Actions or the Knowledge workspace to create one."
            )
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent_list.addItem(empty_item)
            return

        for result in recent:
            stamp = result.updated_at[:16].replace("T", " ")
            item = QListWidgetItem(f"[{result.folder}]  {result.title}   ·   {stamp}")
            item.setData(Qt.ItemDataRole.UserRole, result.note_id)
            self.recent_list.addItem(item)

    def update_context_snapshot(self, snapshot: IRISContextSnapshot | None) -> None:
        """Update environment context labels safely without exposing coordinates."""
        if snapshot is None:
            self.location_label.setText("Location: Unavailable")
            self.weather_label.setText("Weather: Unavailable")
            self.time_label.setText("Local Time: --")
            return

        loc_name = snapshot.location.safe_display_name() if snapshot.location else "Local Device"
        loc_src = f" ({snapshot.location.source})" if snapshot.location else ""
        self.location_label.setText(f"Location: {loc_name}{loc_src}")

        if snapshot.weather is not None:
            w = snapshot.weather
            stale_note = " (cached)" if w.is_stale else ""
            self.weather_label.setText(
                f"Weather: {w.condition}, {w.temperature_c:.1f}°C "
                f"(feels like {w.feels_like_c:.1f}°C){stale_note}"
            )
        else:
            status = "Offline / unavailable" if not snapshot.is_online else "Not yet fetched"
            self.weather_label.setText(f"Weather: {status}")

        if snapshot.timezone is not None:
            tz = snapshot.timezone
            self.time_label.setText(f"Local Time: {tz.current_time_str} ({tz.timezone_name})")
