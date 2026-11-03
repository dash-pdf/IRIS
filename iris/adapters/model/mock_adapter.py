"""
iris/adapters/model/mock_adapter.py
-----------------------------------
Mock Model Adapter for offline testing and development.
"""

from __future__ import annotations

from iris.adapters.model.base import ModelAdapter


class MockModelAdapter(ModelAdapter):
    """
    Predictable offline model adapter for testing and verification.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [
            "I'm with you. Let's tackle that systematically.",
            "Understood. The turn-taking pipeline and echo filters are active.",
            "Indeed. That sounds entirely reasonable.",
        ]
        self._call_count = 0
        self.received_messages: list[list[dict[str, str]]] = []

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        self.received_messages.append(messages)
        resp = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return resp

