"""
iris/adapters/memory_stub.py
-----------------------------
Phase 1 stub for the Memory Adapter interface.

In Phase 1, IRIS has no long-term memory. This stub satisfies the
MemoryAdapter interface so that all other components can call it without
error. It silently returns empty results.

In a later phase, this will be replaced by a HonchoMemoryAdapter that
queries honcho.dev for synthesised user preferences and past context.

Think of this like a placeholder — the shelf where memory will live
is already built into the architecture; it's just empty for now.
"""

from __future__ import annotations


class MemoryAdapter:
    """
    Abstract interface for IRIS's long-term memory system.

    All IRIS components that need memory retrieval should depend on
    this interface, not on any concrete implementation. This way,
    swapping in Honcho later requires no changes outside this module.
    """

    def get_relevant_context(self, user_message: str) -> str:
        """
        Retrieve memory context relevant to the user's message.

        Returns a string to be injected into the prompt, or an empty
        string if no relevant context is found.

        Args:
            user_message: The user's most recent message.

        Returns:
            A string of retrieved memory context (may be empty).
        """
        raise NotImplementedError

    def store_turn(self, user_text: str, assistant_text: str) -> None:
        """
        Store a completed conversation turn for future memory retrieval.

        Args:
            user_text:      What the user said.
            assistant_text: What IRIS replied.
        """
        raise NotImplementedError


class StubMemoryAdapter(MemoryAdapter):
    """
    No-op memory adapter for Phase 1.

    Returns empty context for every query. Stores nothing.
    Drop-in replacement for HonchoMemoryAdapter (Phase 2+).
    """

    def get_relevant_context(self, user_message: str) -> str:  # noqa: ARG002
        # Phase 1: no memory retrieval. Return empty string.
        return ""

    def store_turn(self, user_text: str, assistant_text: str) -> None:  # noqa: ARG002
        # Phase 1: no memory storage. Do nothing.
        pass

