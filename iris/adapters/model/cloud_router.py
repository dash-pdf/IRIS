"""
iris/adapters/model/cloud_router.py
-----------------------------------
Cloud Intelligence Router for IRIS.

Implements deterministic routing across cloud intelligence providers and deterministic tools:
- Everyday conversation: OpenRouter free models (openrouter/free)
- Complex reasoning: Gemini Flash on the Gemini Developer API free tier
- Exact calculations: Deterministic Python math evaluator
- Simple date/time: Deterministic Python/system utilities
- Memory retrieval: Local conversation Markdown archives
- Free-only enforcement: Rejects paid models, halts on quota exhaustion,
  asks user before degrading to weaker free models, and never incurs costs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iris.adapters.model.base import ModelAdapter
from iris.adapters.model.gemini_adapter import (
    DEFAULT_GEMINI_MODEL,
    GeminiModelAdapter,
    GeminiQuotaExceededError,
)
from iris.adapters.model.openrouter_adapter import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterModelAdapter,
)
from iris.core.router import (
    RouteType,
    classify_request,
    is_user_confirmation,
)

if TYPE_CHECKING:
    from iris.core.config import IRISConfig
    from iris.services.context.context_manager import ContextManager

logger = logging.getLogger(__name__)

DEGRADATION_PROMPT = (
    "Gemini Flash is currently unavailable (free-tier quota reached or service error). "
    "Would you like me to try answering this with our standard conversational model instead?"
)


class CloudRouterModelAdapter(ModelAdapter):
    """
    Orchestrates deterministic model routing and tool execution for IRIS.
    Guarantees strict free-only enforcement with no silent or paid fallbacks.
    """

    def __init__(
        self,
        config: IRISConfig,
        openrouter_adapter: OpenRouterModelAdapter | None = None,
        gemini_adapter: GeminiModelAdapter | None = None,
        archive_dir: Path | None = None,
        persona_name: str = "IRIS",
        context_manager: ContextManager | None = None,
    ) -> None:
        self.config = config
        self.persona_name = persona_name
        self.archive_dir = archive_dir or getattr(config, "archive_dir", None)
        self._context_manager = context_manager

        # OpenRouter Adapter (Everyday Conversation)
        if openrouter_adapter is not None:
            self._openrouter = openrouter_adapter
        elif config.openrouter_api_key:
            model = config.llm_model_name if config.llm_model_name in {"openrouter/free"} else DEFAULT_OPENROUTER_MODEL
            self._openrouter = OpenRouterModelAdapter(
                api_key=config.openrouter_api_key,
                model_name=model,
            )
        else:
            self._openrouter = None

        # Gemini Flash Adapter (Complex Reasoning)
        if gemini_adapter is not None:
            self._gemini = gemini_adapter
        elif config.google_api_key:
            from iris.adapters.model.gemini_adapter import GEMINI_FREE_MODELS

            model = (
                config.llm_model_name
                if config.llm_model_name in GEMINI_FREE_MODELS
                else DEFAULT_GEMINI_MODEL
            )
            self._gemini = GeminiModelAdapter(
                api_key=config.google_api_key,
                model_name=model,
            )
        else:
            self._gemini = None

        # Diagnostic Tracking
        self.last_route: str = "init"
        self.last_provider_name: str = "none"
        self.last_selected_model: str = "none"
        self.last_routing_reason: str = ""
        self.last_duration_s: float = 0.0
        self.last_fallback_decision: str | None = None
        self.last_metrics: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self.last_selected_model or "cloud-router"

    def _find_prior_complex_question(self, messages: list[dict[str, str]]) -> str | None:
        """Find the user question asked prior to a degradation inquiry."""
        user_turns: list[str] = []
        for msg in messages:
            if msg.get("role") == "user":
                user_turns.append(msg.get("content", "").strip())
        if len(user_turns) >= 2:
            return user_turns[-2]
        return None

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        t0 = time.perf_counter()
        self.last_fallback_decision = None
        self.last_metrics = {}

        # Extract current user query
        current_query = ""
        last_assistant_reply = ""
        for msg in reversed(messages):
            role = msg.get("role")
            if role == "user" and not current_query:
                current_query = msg.get("content", "").strip()
            elif role == "assistant" and not last_assistant_reply:
                last_assistant_reply = msg.get("content", "").strip()

        if not current_query:
            self.last_route = "empty"
            self.last_provider_name = "Deterministic"
            self.last_selected_model = "none"
            self.last_routing_reason = "Empty user input"
            self.last_duration_s = time.perf_counter() - t0
            return "How can I help you?"

        # Check for user confirmation to degrade after Gemini was unavailable
        if DEGRADATION_PROMPT in last_assistant_reply and is_user_confirmation(current_query):
            prior_question = self._find_prior_complex_question(messages)
            target_question = prior_question or current_query
            logger.info(
                "User confirmed degradation to standard free model for question: '%s'",
                target_question[:60],
            )
            self.last_route = RouteType.CONVERSATION.value
            self.last_provider_name = "openrouter"
            self.last_selected_model = getattr(self._openrouter, "model_name", DEFAULT_OPENROUTER_MODEL)
            self.last_routing_reason = "User approved degradation to standard free model"
            self.last_fallback_decision = "User confirmed fallback to openrouter/free"

            if self._openrouter is None:
                self.last_duration_s = time.perf_counter() - t0
                return (
                    "OpenRouter API credentials (OPENROUTER_API_KEY) are not set. "
                    "Cannot degrade without free model credentials."
                )

            # Build messages substituting the prior question for execution
            fallback_messages = [m for m in messages if m.get("content") != last_assistant_reply and m.get("content") != current_query]
            fallback_messages.append({"role": "user", "content": target_question})
            try:
                reply = self._openrouter.generate_response(fallback_messages, system_prompt=system_prompt)
                self.last_duration_s = time.perf_counter() - t0
                return reply
            except Exception as exc:
                self.last_duration_s = time.perf_counter() - t0
                return (
                    f"OpenRouter free capacity is temporarily unavailable ({exc}). "
                    "Halting without paid fallback."
                )

        # 1. Deterministic Classification
        context_snapshot = self._context_manager.get_context() if self._context_manager else None
        decision = classify_request(
            current_query,
            archive_dir=self.archive_dir,
            persona_name=self.persona_name,
            context_snapshot=context_snapshot,
        )

        self.last_route = decision.route.value
        self.last_routing_reason = decision.reason
        self.last_provider_name = decision.target_provider
        self.last_selected_model = decision.target_model

        # 2. Direct Deterministic Tools Execution
        if decision.direct_response is not None:
            self.last_duration_s = time.perf_counter() - t0
            self.last_metrics = {
                "execution_type": "deterministic",
                "route": decision.route.value,
                "duration_s": self.last_duration_s,
            }
            logger.info(
                "Executed deterministic route [%s]: %s (took %.4fs)",
                decision.route.value,
                decision.reason,
                self.last_duration_s,
            )
            return decision.direct_response

        # Prepare system prompt enriched with real-time environment context if available
        effective_system_prompt = system_prompt
        if self._context_manager:
            try:
                context_prompt = self._context_manager.get_prompt_context()
                if context_prompt:
                    effective_system_prompt = (
                        f"{system_prompt}\n\n{context_prompt}"
                        if system_prompt
                        else context_prompt
                    )
            except Exception as exc:
                logger.debug("Could not attach context prompt: %s", exc)

        # 3. Complex Reasoning (Gemini Flash Free Tier)
        if decision.route == RouteType.COMPLEX_REASONING:
            if self._gemini is None:
                self.last_duration_s = time.perf_counter() - t0
                self.last_fallback_decision = "GOOGLE_API_KEY not configured; prompting user"
                logger.warning("Gemini API key not configured for complex request: '%s'", current_query[:60])
                return (
                    "Gemini Developer API credentials (GOOGLE_API_KEY) are not configured for complex reasoning. "
                    "Would you like me to try answering this with our standard conversational model instead?"
                )

            try:
                reply = self._gemini.generate_response(messages, system_prompt=effective_system_prompt)
                self.last_duration_s = time.perf_counter() - t0
                self.last_selected_model = getattr(self._gemini, "last_selected_model", self.last_selected_model)
                self.last_metrics = {
                    "provider": "google",
                    "http_duration_s": getattr(self._gemini, "last_http_duration_s", self.last_duration_s),
                    "total_duration_s": self.last_duration_s,
                }
                return reply
            except GeminiQuotaExceededError as exc:
                self.last_duration_s = time.perf_counter() - t0
                self.last_fallback_decision = "Gemini quota exceeded; prompting user before degradation"
                logger.warning("Gemini free-tier quota exhausted: %s. Asking user before degrading.", exc)
                return DEGRADATION_PROMPT
            except Exception as exc:
                self.last_duration_s = time.perf_counter() - t0
                self.last_fallback_decision = f"Gemini error: {exc}; prompting user before degradation"
                logger.warning("Gemini call failed: %s. Asking user before degrading.", exc)
                return DEGRADATION_PROMPT

        # 4. Everyday Conversation (OpenRouter Free Tier)
        if decision.route == RouteType.CONVERSATION:
            if self._openrouter is None:
                self.last_duration_s = time.perf_counter() - t0
                self.last_fallback_decision = "OPENROUTER_API_KEY not configured"
                logger.warning("OpenRouter API key not configured for query: '%s'", current_query[:60])
                return (
                    "OpenRouter API credentials (OPENROUTER_API_KEY) are not configured. "
                    "Please set your free-tier API key in .env to enable conversation."
                )

            try:
                reply = self._openrouter.generate_response(messages, system_prompt=effective_system_prompt)
                self.last_duration_s = time.perf_counter() - t0
                self.last_selected_model = getattr(self._openrouter, "last_selected_model", self.last_selected_model)
                self.last_metrics = {
                    "provider": "openrouter",
                    "http_duration_s": getattr(self._openrouter, "last_http_duration_s", self.last_duration_s),
                    "total_duration_s": self.last_duration_s,
                }
                return reply
            except Exception as exc:
                self.last_duration_s = time.perf_counter() - t0
                self.last_fallback_decision = "OpenRouter error / rate-limited; halted without paid fallback"
                logger.error("OpenRouter free call failed: %s. Halting without paid fallback.", exc)
                return (
                    "OpenRouter free capacity is temporarily unavailable (rate limit reached). "
                    "To prevent any unintended charges, IRIS will not fall back to paid models. "
                    "Please try again in a few moments."
                )

        # Fallback safeguard
        self.last_duration_s = time.perf_counter() - t0
        return "Understood."


__all__ = [
    "CloudRouterModelAdapter",
    "DEGRADATION_PROMPT",
]
