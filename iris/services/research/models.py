"""
iris/services/research/models.py
------------------------------
Data structures for future Phase 4 multi-tab web research and mandatory Video
Intelligence workflows.

Security & Trust Invariants:
- External webpage text, captions, and transcripts are ALWAYS flagged as untrusted
  input (`untrusted_content=True`) and must never act as system instructions.
- Researching webpages or analyzing videos NEVER auto-creates permanent Knowledge
  Vault notes; persistence requires explicit user instruction.
- Video summaries preserve source URLs, titles, access dates, and timestamps,
  and never claim to have analyzed inaccessible video content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceType(str, Enum):
    WEBPAGE = "webpage"
    VIDEO = "video"
    DOCUMENT = "document"


@dataclass(frozen=True)
class VideoTimestamp:
    """Timestamped chapter/moment inside an analyzed video source."""

    seconds: int
    label: str
    timestamp_url: str = ""

    def formatted_time(self) -> str:
        mins, secs = divmod(max(0, self.seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"


@dataclass(frozen=True)
class ResearchSource:
    """
    Metadata record for a future Research Engine source (webpage or video).
    Always retains provenance (`url`, `access_date`) and `untrusted_content=True`.
    """

    url: str
    title: str
    access_date: str
    source_type: SourceType = SourceType.WEBPAGE
    untrusted_content: bool = True
    content_accessible: bool = False
    transcript_available: bool = False
    timestamps: tuple[VideoTimestamp, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VideoResearchSummary:
    """
    Contract for future Phase 4 Video Intelligence outputs (key concepts,
    study notes, comparison findings, and revision flashcards).
    """

    video_title: str
    source_url: str
    access_date: str
    content_accessible: bool
    summary: str = ""
    key_concepts: tuple[str, ...] = field(default_factory=tuple)
    timestamps: tuple[VideoTimestamp, ...] = field(default_factory=tuple)
    flashcards: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    auto_saved_to_vault: bool = False
