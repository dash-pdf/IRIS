"""
iris/adapters/model/__init__.py
-------------------------------
LLM and Intelligence adapters for IRIS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris.adapters.model.anthropic_adapter import AnthropicModelAdapter
from iris.adapters.model.base import ModelAdapter
from iris.adapters.model.gemini_adapter import GeminiModelAdapter, GoogleModelAdapter
from iris.adapters.model.mock_adapter import MockModelAdapter
from iris.adapters.model.ollama_adapter import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
    OllamaModelAdapter,
)
from iris.adapters.model.openai_adapter import OpenAIModelAdapter
from iris.adapters.model.openrouter_adapter import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterModelAdapter,
)

if TYPE_CHECKING:
    from iris.core.config import IRISConfig


def create_model_adapter(config: IRISConfig) -> ModelAdapter:
    """
    Instantiate the configured model adapter based on config.
    """
    provider = config.llm_provider.lower()

    if provider == "ollama":
        host = getattr(config, "ollama_host", "") or DEFAULT_OLLAMA_HOST
        model = config.llm_model_name or DEFAULT_OLLAMA_MODEL
        keep_alive = getattr(config, "ollama_keep_alive", DEFAULT_OLLAMA_KEEP_ALIVE) or DEFAULT_OLLAMA_KEEP_ALIVE
        return OllamaModelAdapter(host=host, model_name=model, keep_alive=keep_alive)
    elif provider == "openrouter":
        if not config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required for OpenRouter model provider.")
        model = config.llm_model_name or DEFAULT_OPENROUTER_MODEL
        return OpenRouterModelAdapter(api_key=config.openrouter_api_key, model_name=model)
    elif provider == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for Anthropic model provider.")
        model = config.llm_model_name or "claude-3-5-haiku-20241022"
        return AnthropicModelAdapter(api_key=config.anthropic_api_key, model_name=model)
    elif provider in {"google", "gemini"}:
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY required for Google model provider.")
        model = config.llm_model_name or "gemini-3.8-flash"
        return GeminiModelAdapter(api_key=config.google_api_key, model_name=model)
    elif provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI model provider.")
        model = config.llm_model_name or "gpt-4o-mini"
        return OpenAIModelAdapter(api_key=config.openai_api_key, model_name=model)
    elif provider == "mock":
        return MockModelAdapter()
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Supported: ollama, openrouter, anthropic, google, gemini, openai, mock"
        )


__all__ = [
    "ModelAdapter",
    "OllamaModelAdapter",
    "DEFAULT_OLLAMA_KEEP_ALIVE",
    "OpenRouterModelAdapter",
    "AnthropicModelAdapter",
    "GeminiModelAdapter",
    "GoogleModelAdapter",
    "OpenAIModelAdapter",
    "MockModelAdapter",
    "create_model_adapter",
]

