"""
iris/adapters/model/gemini_adapter.py
-------------------------------------
Google Gemini model adapter for IRIS using official google-genai SDK.
"""

from __future__ import annotations

import logging
from typing import Any

from iris.adapters.model.base import ModelAdapter

logger = logging.getLogger(__name__)


class GeminiModelAdapter(ModelAdapter):
    """
    Integrates Google Gemini models (e.g. gemini-3.8-flash / gemini-2.5-flash)
    using the official google-genai SDK.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.8-flash",
    ) -> None:
        if not api_key:
            raise ValueError("Google API key is required.")
        self.api_key = api_key
        self.model_name = model_name
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except ImportError as err:
                raise ImportError(
                    "google-genai package is required for GeminiModelAdapter. "
                    "Install it via `pip install google-genai`."
                ) from err
        return self._client

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        extracted_system = system_prompt or ""
        conversation_turns: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                if not extracted_system:
                    extracted_system = content
            elif role == "user":
                conversation_turns.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })
            elif role == "assistant":
                conversation_turns.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })

        # Gemini conversations must begin with a user turn
        while conversation_turns and conversation_turns[0]["role"] != "user":
            conversation_turns.pop(0)

        if not conversation_turns:
            return "How can I help you?"

        logger.info(
            "Calling Gemini model (%s) with %d turns",
            self.model_name,
            len(conversation_turns),
        )

        try:
            client = self._get_client()
            from google.genai import types

            config = None
            if extracted_system:
                config = types.GenerateContentConfig(
                    system_instruction=extracted_system,
                )

            response = client.models.generate_content(
                model=self.model_name,
                contents=conversation_turns,
                config=config,
            )

            text = response.text or ""
            return text.strip() or "Understood."
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise RuntimeError(f"Gemini generation error: {exc}") from exc


# Alias for backwards compatibility and nomenclature alignment
GoogleModelAdapter = GeminiModelAdapter


