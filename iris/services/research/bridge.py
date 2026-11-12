"""
iris/services/research/bridge.py
------------------------------
Architectural bridge connecting future Phase 4 Research & Video Intelligence
modules to the existing Markdown Knowledge Vault and LLM Usage Policy.

Notice: Phase 2 provides this interface boundary ONLY; live web browsing,
transcript fetching, and video analysis belong to the Research Engine phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris.core.policy import DEFAULT_LLM_POLICY, ExecutionMode, ServiceCapability
from iris.services.research.models import ResearchSource, VideoResearchSummary

if TYPE_CHECKING:
    from iris.services.knowledge.models import NoteDocument
    from iris.services.knowledge.service import KnowledgeService


class ResearchWorkspaceBridge:
    """
    Modularity anchor for Phase 4 Research Engine and Video Intelligence integration.

    Enforces:
    - Explicit-user-intent saving into `vault/Research/` with `source_url` provenance.
    - Explicit refusal to store/claim analysis of inaccessible video content.
    - Untrusted-input marking for external web/transcript text.
    """

    CAPABILITY = ServiceCapability(
        service_id="research_workspace_bridge",
        display_name="Research & Video Intelligence Bridge",
        requires_llm=False,
        default_mode=ExecutionMode.DETERMINISTIC,
        offline_capable=True,
        implemented=True,
        description="Modular bridge exporting explicitly confirmed research/video sources to the Knowledge Vault.",
    )

    def __init__(self, knowledge_service: KnowledgeService | None = None) -> None:
        self._knowledge = knowledge_service
        if DEFAULT_LLM_POLICY.get_capability(self.CAPABILITY.service_id) is None:
            DEFAULT_LLM_POLICY.register_capability(self.CAPABILITY)

    @property
    def is_research_engine_implemented(self) -> bool:
        """Phase 2 boundary: live browser and Video Intelligence engine are future phases."""
        return False

    def save_verified_source_note(
        self,
        source: ResearchSource,
        user_confirmed: bool,
        markdown_body: str,
        tags: list[str] | None = None,
    ) -> NoteDocument:
        """
        Persist a research source note ONLY when `user_confirmed=True` and
        the Knowledge Vault service is connected.
        """
        if not user_confirmed:
            raise PermissionError(
                "Refusing to auto-save research source: explicit user save instruction is required."
            )
        if self._knowledge is None:
            raise RuntimeError("KnowledgeService is not connected to ResearchWorkspaceBridge.")

        header = (
            f"> Source URL: {source.url}\n"
            f"> Access Date: {source.access_date}\n"
            f"> Untrusted External Input: {source.untrusted_content}\n\n"
        )
        return self._knowledge.create_note(
            title=source.title or source.url,
            content=header + markdown_body.strip(),
            folder="Research",
            note_type="research",
            tags=tags or [source.source_type.value, "research"],
            source_url=source.url,
        )

    def format_video_study_markdown(self, report: VideoResearchSummary) -> str:
        """Format a structured Video Intelligence summary with preserved timestamps."""
        if not report.content_accessible:
            raise ValueError(
                f"Cannot generate study notes for '{report.video_title}': video content was not accessible."
            )

        sections = [
            f"# {report.video_title}",
            "",
            f"- **Source URL:** {report.source_url}",
            f"- **Access Date:** {report.access_date}",
            "",
        ]
        if report.summary.strip():
            sections.extend(["## Summary", report.summary.strip(), ""])
        if report.key_concepts:
            sections.append("## Key Concepts")
            sections.extend(f"- {concept}" for concept in report.key_concepts)
            sections.append("")
        if report.timestamps:
            sections.append("## Timestamps")
            for ts in report.timestamps:
                link_suffix = f" ({ts.timestamp_url})" if ts.timestamp_url else ""
                sections.append(f"- `{ts.formatted_time()}` — {ts.label}{link_suffix}")
            sections.append("")
        if report.flashcards:
            sections.append("## Revision Flashcards")
            for q, a in report.flashcards:
                sections.append(f"- **Q:** {q}\n  - **A:** {a}")
            sections.append("")
        return "\n".join(sections).strip()
