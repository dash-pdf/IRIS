"""
iris/desktop/__init__.py
------------------------
PySide6 Desktop Application Shell for IRIS (Phase 2).
"""

from iris.desktop.backend import IRISDesktopBackend
from iris.desktop.main_window import IRISMainWindow

__all__ = ["IRISDesktopBackend", "IRISMainWindow"]
