"""
iris/adapters/model/gemini_adapter.py
-------------------------------------
Google Gemini model adapter for IRIS using official google-genai SDK.
Enforces strict model allowlist restricting usage exclusively to verified free-tier models.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from iris.adapters.model.base import ModelAdapter

logger = logging.getLogger(__name__)

# Strict free-tier allowlist for Gemini Developer API.
# Any attempt to request a paid model (e.g. Pro, Ultra) is blocked before execution.
GEMINI_FREE_MODELS: frozenset[str] = frozenset([
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
])

DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"


class GeminiQuotaExceededError(RuntimeError):
    """Raised when Gemini Developer API free-tier quota is exhausted or rate-limited."""
    pass


class GeminiModelAdapter(ModelAdapter):
    """
    Integrates Google Gemini models (e.g. gemini-3.8-flash / gemini-2.5-flash)
    using the official google-genai SDK. Enforces strict free-tier allowlist.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_GEMINI_MODEL,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("Google API key is required.")

        self._validate_model(model_name)

        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self._client: Any = None
        self.last_selected_model: str | None = None
        self.last_http_duration_s: float = 0.0

    @staticmethod
    def _validate_model(model_name: str) -> None:
        if model_name not in GEMINI_FREE_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not permitted. "
                f"Strict free-tier allowlist permits only: {sorted(GEMINI_FREE_MODELS)}. "
                "Requests to paid or unapproved Gemini models are blocked."
            )

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
        # Re-verify allowlist before making any network request
        self._validate_model(self.model_name)

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
            "Calling Gemini Developer API model (%s) with %d turns",
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

            t0 = time.perf_counter()
            response = client.models.generate_content(
                model=self.model_name,
                contents=conversation_turns,
                config=config,
            )
            self.last_http_duration_s = time.perf_counter() - t0
            self.last_selected_model = self.model_name

            text = response.text or ""
            return text.strip() or "Understood."
        except Exception as exc:
            err_str = str(exc)
            if any(code in err_str.lower() for code in ["429", "resource_exhausted", "quota", "rate limit"]):
                logger.warning("Gemini Developer API quota reached: %s", exc)
                raise GeminiQuotaExceededError(
                    f"Gemini Developer API free-tier quota or rate limit exceeded for '{self.model_name}'."
                ) from exc
            logger.error("Gemini API call failed: %s", exc)
            raise RuntimeError(f"Gemini generation error: {exc}") from exc


# Alias for backwards compatibility and nomenclature alignment
GoogleModelAdapter = GeminiModelAdapter

__all__ = [
    "GeminiModelAdapter",
    "GoogleModelAdapter",
    "GeminiQuotaExceededError",
    "GEMINI_FREE_MODELS",
    "DEFAULT_GEMINI_MODEL",
]
