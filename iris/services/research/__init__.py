"""
iris/services/research/__init__.py
--------------------------------
Modular contracts and architecture bridges for the future Phase 4 Research Engine
and mandatory Video Intelligence system.
"""

from iris.services.research.bridge import ResearchWorkspaceBridge
from iris.services.research.models import (
    ResearchSource,
    SourceType,
    VideoResearchSummary,
    VideoTimestamp,
)

__all__ = [
    "ResearchSource",
    "ResearchWorkspaceBridge",
    "SourceType",
    "VideoResearchSummary",
    "VideoTimestamp",
]
