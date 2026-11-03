"""
iris/adapters/model/openai_adapter.py
-------------------------------------
OpenAI model adapter for IRIS using REST API.
"""

from __future__ import annotations

import logging
from typing import Any

from iris.adapters.model.base import ModelAdapter

logger = logging.getLogger(__name__)


class OpenAIModelAdapter(ModelAdapter):
    """
    Integrates OpenAI models (e.g. gpt-4o / gpt-4o-mini).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        self.model_name = model_name

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        import httpx

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)

        payload = {
            "model": self.model_name,
            "messages": api_messages,
        }

        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return "Understood."
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise RuntimeError(f"OpenAI generation error: {exc}") from exc

