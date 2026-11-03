"""
iris/adapters/model/base.py
---------------------------
Abstract interface for LLM / Intelligence Providers.

Decouples IRIS reasoning and intelligence from any speech or voice vendor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """
    Abstract Model Adapter interface.
    """

    @abstractmethod
    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a conversational response from the underlying LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            system_prompt: Optional override system personality prompt.

        Returns:
            The generated response string.
        """
        pass

