"""
iris_desktop.py
---------------
Main entry point for the IRIS PySide6 Desktop Application (Phase 2).

Usage:
    ./run.sh --desktop
    .venv/bin/python iris_desktop.py
"""

from __future__ import annotations

import logging
import signal
import sys

from PySide6.QtWidgets import QApplication

from iris.core.config import IRISConfig, load_config
from iris.desktop.backend import IRISDesktopBackend
from iris.desktop.main_window import IRISMainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def create_desktop_app(
    argv: list[str] | None = None,
    config: IRISConfig | None = None,
    auto_start_backend: bool = True,
) -> tuple[QApplication, IRISMainWindow]:
    """Build or reuse a QApplication and construct the IRISMainWindow cleanly."""
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(argv or sys.argv)
    app.setApplicationName("IRIS")
    app.setApplicationDisplayName("IRIS — Intelligent Research & Information System")

    resolved_config = config or load_config(require_voice=False)
    backend = IRISDesktopBackend(config=resolved_config)
    window = IRISMainWindow(backend=backend, auto_start_backend=auto_start_backend)
    return app, window


def main() -> int:
    app, window = create_desktop_app(sys.argv)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
