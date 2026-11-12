"""
iris/desktop/theme.py
---------------------
IRIS Desktop visual identity and Qt stylesheet (QSS).

Palette:
- Background: #09090D
- Surface:    #14131B
- Accent:     #8057D9
- Highlight:  #B69AFF
"""

from __future__ import annotations

PALETTE = {
    "bg_root": "#09090D",
    "bg_surface": "#14131B",
    "bg_elevated": "#1C1A26",
    "bg_hover": "#232030",
    "border": "#2A263A",
    "border_strong": "#3B3552",
    "accent": "#8057D9",
    "accent_hover": "#9069E6",
    "accent_pressed": "#6D44C7",
    "highlight": "#B69AFF",
    "text_primary": "#F2F0FA",
    "text_secondary": "#B0ABC7",
    "text_muted": "#726D8A",
    "success": "#4ADE80",
    "warning": "#FBBF24",
    "error": "#F87171",
}


def build_iris_stylesheet() -> str:
    """Return the global dark Obsidian-inspired QSS theme for IRIS."""
    p = PALETTE
    return f"""
    QMainWindow, QWidget {{
        background-color: {p['bg_root']};
        color: {p['text_primary']};
        font-family: -apple-system, "SF Pro Text", "Inter", "Segoe UI", sans-serif;
        font-size: 13px;
    }}

    QFrame#SidebarPanel, QFrame#AssistantPanel, QFrame#VoiceControlBar, QFrame#SurfaceCard {{
        background-color: {p['bg_surface']};
        border: 1px solid {p['border']};
        border-radius: 10px;
    }}

    QFrame#SidebarPanel {{
        border-radius: 0px;
        border-top: none;
        border-bottom: none;
        border-left: none;
        border-right: 1px solid {p['border']};
    }}

    QFrame#AssistantPanel {{
        border-radius: 0px;
        border-top: none;
        border-bottom: none;
        border-right: none;
        border-left: 1px solid {p['border']};
    }}

    QFrame#VoiceControlBar {{
        border-radius: 0px;
        border-left: none;
        border-right: none;
        border-bottom: none;
        border-top: 1px solid {p['border']};
    }}

    QFrame#CentreHeader {{
        background-color: {p['bg_root']};
        border-bottom: 1px solid {p['border']};
    }}

    QLabel#BrandTitle {{
        font-size: 16px;
        font-weight: 700;
        color: {p['text_primary']};
        letter-spacing: 1px;
    }}

    QLabel#BrandSubtitle {{
        font-size: 11px;
        color: {p['highlight']};
    }}

    QLabel#SectionHeader {{
        font-size: 11px;
        font-weight: 600;
        color: {p['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 4px 2px;
    }}

    QLabel#PageTitle {{
        font-size: 20px;
        font-weight: 700;
        color: {p['text_primary']};
    }}

    QLabel#MutedLabel {{
        color: {p['text_secondary']};
        font-size: 12px;
    }}

    QLabel#HighlightPill {{
        background-color: rgba(128, 87, 217, 0.18);
        color: {p['highlight']};
        border: 1px solid rgba(182, 154, 255, 0.35);
        border-radius: 10px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
    }}

    QLabel#StatusSuccessPill {{
        background-color: rgba(74, 222, 128, 0.14);
        color: {p['success']};
        border: 1px solid rgba(74, 222, 128, 0.35);
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
        font-weight: 600;
    }}

    QLabel#StatusWarningPill {{
        background-color: rgba(251, 191, 36, 0.14);
        color: {p['warning']};
        border: 1px solid rgba(251, 191, 36, 0.35);
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
        font-weight: 600;
    }}

    QLabel#StatusInactivePill {{
        background-color: {p['bg_elevated']};
        color: {p['text_muted']};
        border: 1px solid {p['border']};
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
        font-weight: 600;
    }}

    QLabel#ErrorBanner {{
        background-color: rgba(248, 113, 113, 0.14);
        color: {p['error']};
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 12px;
    }}

    QPushButton {{
        background-color: {p['bg_elevated']};
        color: {p['text_primary']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        padding: 7px 14px;
        font-weight: 500;
    }}

    QPushButton:hover {{
        background-color: {p['bg_hover']};
        border-color: {p['border_strong']};
    }}

    QPushButton:pressed {{
        background-color: {p['bg_surface']};
    }}

    QPushButton:disabled {{
        color: {p['text_muted']};
        background-color: {p['bg_surface']};
        border-color: {p['border']};
    }}

    QPushButton#PrimaryButton {{
        background-color: {p['accent']};
        color: #FFFFFF;
        border: 1px solid {p['accent']};
        font-weight: 600;
    }}

    QPushButton#PrimaryButton:hover {{
        background-color: {p['accent_hover']};
        border-color: {p['highlight']};
    }}

    QPushButton#PrimaryButton:pressed {{
        background-color: {p['accent_pressed']};
    }}

    QPushButton#NavButton {{
        background-color: transparent;
        color: {p['text_secondary']};
        border: 1px solid transparent;
        border-radius: 8px;
        text-align: left;
        padding: 8px 12px;
        font-weight: 500;
    }}

    QPushButton#NavButton:hover {{
        background-color: {p['bg_hover']};
        color: {p['text_primary']};
    }}

    QPushButton#NavButton[active="true"] {{
        background-color: rgba(128, 87, 217, 0.22);
        color: {p['highlight']};
        border: 1px solid rgba(128, 87, 217, 0.5);
        font-weight: 600;
    }}

    QPushButton#MicButtonActive {{
        background-color: rgba(74, 222, 128, 0.16);
        color: {p['success']};
        border: 1px solid {p['success']};
        border-radius: 8px;
        padding: 7px 14px;
        font-weight: 600;
    }}

    QPushButton#DangerButton {{
        background-color: rgba(248, 113, 113, 0.12);
        color: {p['error']};
        border: 1px solid rgba(248, 113, 113, 0.35);
    }}

    QPushButton#DangerButton:hover {{
        background-color: rgba(248, 113, 113, 0.22);
    }}

    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{
        background-color: {p['bg_elevated']};
        color: {p['text_primary']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        padding: 7px 10px;
        selection-background-color: {p['accent']};
    }}

    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
        border: 1px solid {p['accent']};
    }}

    QPlainTextEdit#MarkdownEditor {{
        font-family: "SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace;
        font-size: 13px;
        line-height: 1.5;
        padding: 12px;
    }}

    QTextBrowser#MarkdownPreview {{
        background-color: {p['bg_surface']};
        color: {p['text_primary']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        padding: 14px;
    }}

    QListWidget, QTreeWidget, QTableWidget {{
        background-color: {p['bg_surface']};
        color: {p['text_primary']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        outline: 0;
    }}

    QListWidget::item, QTreeWidget::item {{
        padding: 7px 10px;
        border-radius: 6px;
        margin: 2px 4px;
    }}

    QListWidget::item:hover, QTreeWidget::item:hover {{
        background-color: {p['bg_hover']};
    }}

    QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: rgba(128, 87, 217, 0.26);
        color: {p['highlight']};
        border: 1px solid rgba(182, 154, 255, 0.35);
    }}

    QTabWidget::pane {{
        border: 1px solid {p['border']};
        border-radius: 8px;
        background-color: {p['bg_surface']};
        top: -1px;
    }}

    QTabBar::tab {{
        background-color: {p['bg_root']};
        color: {p['text_secondary']};
        border: 1px solid {p['border']};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 7px 14px;
        margin-right: 4px;
    }}

    QTabBar::tab:selected {{
        background-color: {p['bg_surface']};
        color: {p['highlight']};
        border-color: {p['border_strong']};
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {p['bg_elevated']};
        color: {p['text_primary']};
    }}

    QSplitter::handle {{
        background-color: {p['border']};
    }}

    QProgressBar {{
        background-color: {p['bg_elevated']};
        border: 1px solid {p['border']};
        border-radius: 4px;
        height: 6px;
        text-align: center;
    }}

    QProgressBar::chunk {{
        background-color: {p['accent']};
        border-radius: 3px;
    }}

    QScrollArea {{
        border: none;
        background-color: transparent;
    }}

    QScrollBar:vertical {{
        background: {p['bg_root']};
        width: 8px;
        margin: 0px;
    }}

    QScrollBar::handle:vertical {{
        background: {p['border_strong']};
        border-radius: 4px;
        min-height: 24px;
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    """
