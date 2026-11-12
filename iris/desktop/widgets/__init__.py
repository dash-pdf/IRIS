"""
iris/desktop/widgets/__init__.py
--------------------------------
Modular PySide6 UI widgets for IRIS Phase 2 Desktop Application Shell.
"""

from iris.desktop.widgets.assistant_panel import AssistantPanel
from iris.desktop.widgets.home_page import HomePage
from iris.desktop.widgets.knowledge_page import KnowledgePage
from iris.desktop.widgets.placeholders import PlaceholderWorkspace, SettingsPage
from iris.desktop.widgets.research_page import ResearchPage
from iris.desktop.widgets.sidebar import SidebarWidget
from iris.desktop.widgets.voice_bar import VoiceControlBar

__all__ = [
    "AssistantPanel",
    "HomePage",
    "KnowledgePage",
    "PlaceholderWorkspace",
    "ResearchPage",
    "SettingsPage",
    "SidebarWidget",
    "VoiceControlBar",
]
