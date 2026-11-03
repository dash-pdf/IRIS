"""
iris/adapters/model/openrouter_adapter.py
-----------------------------------------
OpenRouter model adapter for IRIS using OpenAI-compatible REST API.
Enforces strict model allowlist restricting usage exclusively to free tier models.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from iris.adapters.model.base import ModelAdapter

logger = logging.getLogger(__name__)

# Strict allowlist: Only free router openrouter/free is permitted.
# Any attempt to request a paid model or unapproved model will be blocked before execution.
ALLOWED_MODELS: frozenset[str] = frozenset([
    "openrouter/free",
])

DEFAULT_OPENROUTER_MODEL: str = "openrouter/free"
OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterModelAdapter(ModelAdapter):
    """
    Integrates OpenRouter with strict model allowlist enforcement.
    Blocks paid models, rejects fallbacks, and stops immediately on failure.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_OPENROUTER_MODEL,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is required.")

        self._validate_model(model_name)

        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.last_selected_model: str | None = None
        self.last_http_duration_s: float = 0.0

    @staticmethod
    def _validate_model(model_name: str) -> None:
        if model_name not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not permitted. "
                f"Strict allowlist permits only: {sorted(ALLOWED_MODELS)}. "
                "Requests to paid or unapproved models are blocked."
            )

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        # Re-verify allowlist before making any network request
        self._validate_model(self.model_name)

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

        if extracted_system:
            api_messages.insert(0, {"role": "system", "content": extracted_system})

        if not api_messages or all(m.get("role") == "system" for m in api_messages):
            return "How can I help you?"

        logger.info(
            "Calling OpenRouter model (%s) with %d turns",
            self.model_name,
            len(api_messages),
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/iris-voice",
            "X-Title": "IRIS Voice Assistant",
        }

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
        }

        import httpx

        try:
            t0 = time.perf_counter()
            resp = httpx.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            self.last_http_duration_s = time.perf_counter() - t0

            if resp.status_code == 429:
                sanitized_body = resp.text.strip()[:300] if resp.text else "Free capacity is temporarily exhausted."
                err_msg = (
                    f"OpenRouter rate limit exceeded (HTTP 429) for '{self.model_name}'. "
                    f"Details: {sanitized_body}. Halting without fallback."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            if resp.status_code == 404:
                sanitized_body = resp.text.strip()[:300] if resp.text else "Model not found or unavailable."
                err_msg = (
                    f"OpenRouter model '{self.model_name}' not found or unavailable (HTTP 404). "
                    f"Details: {sanitized_body}. Halting without fallback."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            resp.raise_for_status()
            data = resp.json()
            selected_model = data.get("model")
            if selected_model:
                self.last_selected_model = str(selected_model)
                logger.info(
                    "OpenRouter selected model: '%s' (requested: '%s', HTTP request took %.2fs)",
                    self.last_selected_model,
                    self.model_name,
                    self.last_http_duration_s,
                )
            else:
                self.last_selected_model = None

            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(
                    f"OpenRouter returned empty choices for model '{self.model_name}'."
                )

            reply = choices[0].get("message", {}).get("content", "")
            if not reply or not reply.strip():
                return "Understood."
            return reply.strip()

        except httpx.TimeoutException as exc:
            err_msg = (
                f"OpenRouter request timed out after {self.timeout_seconds}s for '{self.model_name}'. "
                "Halting without fallback."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc
        except httpx.HTTPStatusError as exc:
            sanitized_body = exc.response.text.strip()[:300] if exc.response.text else "No response body."
            err_msg = (
                f"OpenRouter HTTP error {exc.response.status_code}: {sanitized_body}. "
                "Halting without fallback."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            err_msg = (
                f"OpenRouter generation error: {exc}. "
                "Halting without fallback."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc

