"""
iris/core/policy.py
-------------------
Application-wide LLM Usage Policy for IRIS.

Fundamental rule:
    IRIS must not need an LLM to perform ordinary application operations.

- File operations use Python.
- Note indexing and full-text search use SQLite FTS5.
- Folder and Inbox management use Python.
- UI navigation and workspace state never invoke an LLM unnecessarily.
- External APIs and local services handle deterministic tasks (weather, time,
  location, math, file persistence).
- UI layers contain zero API credentials or model SDK clients; all LLM execution
  happens strictly through the existing `ModelAdapter` / `CloudRouterModelAdapter`.
- The application remains useful when all cloud AI providers are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ExecutionMode(str, Enum):
    """Execution tier declared by an IRIS service or capability."""

    DETERMINISTIC = "deterministic"
    LIGHTWEIGHT_INTERPRETATION = "lightweight_interpretation"
    CLOUD_LLM = "cloud_llm"


@dataclass(frozen=True)
class ServiceCapability:
    """Formal declaration of whether an IRIS service requires an LLM."""

    service_id: str
    display_name: str
    requires_llm: bool
    default_mode: ExecutionMode
    offline_capable: bool
    implemented: bool
    description: str


BUILTIN_CAPABILITIES: tuple[ServiceCapability, ...] = (
    ServiceCapability(
        service_id="knowledge_vault",
        display_name="Markdown Knowledge Vault",
        requires_llm=False,
        default_mode=ExecutionMode.DETERMINISTIC,
        offline_capable=True,
        implemented=True,
        description="Atomic Markdown note CRUD, YAML frontmatter, folder moves, and Inbox persistence in Python.",
    ),
    ServiceCapability(
        service_id="knowledge_search",
        display_name="Knowledge FTS5 Search",
        requires_llm=False,
        default_mode=ExecutionMode.DETERMINISTIC,
        offline_capable=True,
        implemented=True,
        description="Disposable SQLite FTS5 full-text, tag, note-type, and folder search.",
    ),
    ServiceCapability(
        service_id="deterministic_tools",
        display_name="Calculator, Date/Time & Session Recall",
        requires_llm=False,
        default_mode=ExecutionMode.DETERMINISTIC,
        offline_capable=True,
        implemented=True,
        description="Deterministic Python math evaluation, local timezone formatting, and session transcript recall.",
    ),
    ServiceCapability(
        service_id="context_engine",
        display_name="Context Engine (Location & Weather)",
        requires_llm=False,
        default_mode=ExecutionMode.DETERMINISTIC,
        offline_capable=True,
        implemented=True,
        description="Native macOS CoreLocation / manual location and Open-Meteo weather with local cache fallback.",
    ),
    ServiceCapability(
        service_id="assistant_orchestration",
        display_name="Assistant & Cloud Intelligence Router",
        requires_llm=True,
        default_mode=ExecutionMode.CLOUD_LLM,
        offline_capable=False,
        implemented=True,
        description="Complex synthesis and general dialogue routed strictly through IRIS's ModelAdapter layer.",
    ),
    ServiceCapability(
        service_id="research_browser",
        display_name="Embedded Research Browser",
        requires_llm=False,
        default_mode=ExecutionMode.LIGHTWEIGHT_INTERPRETATION,
        offline_capable=False,
        implemented=False,
        description="Future Phase 4 multi-tab web viewer and untrusted content extraction; notes saved only on explicit command.",
    ),
    ServiceCapability(
        service_id="video_intelligence",
        display_name="Video Intelligence Engine",
        requires_llm=True,
        default_mode=ExecutionMode.CLOUD_LLM,
        offline_capable=False,
        implemented=False,
        description="Future Phase 4 transcript, timestamp, concept, flashcard, and study-note synthesis for permitted videos.",
    ),
)


class LLMUsagePolicy:
    """Registry and enforcement helper for IRIS service LLM declarations."""

    def __init__(self, capabilities: Iterable[ServiceCapability] | None = None) -> None:
        self._registry: dict[str, ServiceCapability] = {}
        for cap in (capabilities if capabilities is not None else BUILTIN_CAPABILITIES):
            self.register_capability(cap)

    def register_capability(self, capability: ServiceCapability) -> None:
        """Register or update a service capability declaration."""
        if not capability.service_id.strip():
            raise ValueError("Capability service_id must not be empty.")
        self._registry[capability.service_id] = capability

    def get_capability(self, service_id: str) -> ServiceCapability | None:
        return self._registry.get(service_id)

    def list_capabilities(self) -> list[ServiceCapability]:
        return list(self._registry.values())

    def requires_llm(self, service_id: str) -> bool:
        """Return True only if the declared service genuinely requires an LLM."""
        cap = self._registry.get(service_id)
        if cap is None:
            return False
        return cap.requires_llm

    def offline_capabilities(self) -> list[ServiceCapability]:
        """Return implemented capabilities that remain operational without cloud AI."""
        return [
            cap for cap in self._registry.values()
            if cap.implemented and cap.offline_capable and not cap.requires_llm
        ]


DEFAULT_LLM_POLICY = LLMUsagePolicy()
