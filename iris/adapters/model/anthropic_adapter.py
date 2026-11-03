"""
iris/adapters/model/anthropic_adapter.py
----------------------------------------
Anthropic Claude model adapter for IRIS.

Uses httpx REST API (or the anthropic SDK if installed).
"""

from __future__ import annotations

import logging
from typing import Any

from iris.adapters.model.base import ModelAdapter

logger = logging.getLogger(__name__)


class AnthropicModelAdapter(ModelAdapter):
    """
    Integrates Anthropic Claude models (e.g. Claude 3.5 Sonnet / Haiku).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 600,
    ) -> None:
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        extracted_system = system_prompt or ""
        api_messages: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if not extracted_system:
                    extracted_system = content
            elif role in {"user", "assistant"}:
                api_messages.append({"role": role, "content": content})

        # Anthropic requires conversations to begin with a user turn
        while api_messages and api_messages[0]["role"] != "user":
            api_messages.pop(0)

        if not api_messages:
            return "How can I help you?"

        logger.info("Calling Anthropic Claude (%s) with %d turns", self.model_name, len(api_messages))

        # Try official SDK if installed
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=self.api_key)
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "messages": api_messages,
            }
            if extracted_system:
                kwargs["system"] = extracted_system
            response = client.messages.create(**kwargs)
            if response.content:
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                return "".join(text_blocks).strip()
            return "Understood."
        except ImportError:
            # Fallback to direct REST API via httpx
            import httpx

            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "messages": api_messages,
            }
            if extracted_system:
                payload["system"] = extracted_system

            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("content", [])
                text_blocks = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "".join(text_blocks).strip() or "Understood."
            except Exception as exc:
                logger.error("Anthropic REST call failed: %s", exc)
                raise RuntimeError(f"Anthropic API generation failed: {exc}") from exc

