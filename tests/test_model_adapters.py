"""
tests/test_model_adapters.py
----------------------------
Unit tests for iris/adapters/model/ (LLM Intelligence Adapters).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iris.adapters.model import (
    AnthropicModelAdapter,
    CloudRouterModelAdapter,
    GeminiModelAdapter,
    GoogleModelAdapter,
    MockModelAdapter,
    OllamaModelAdapter,
    OpenAIModelAdapter,
    OpenRouterModelAdapter,
    create_model_adapter,
)
from iris.adapters.model.openrouter_adapter import (
    ALLOWED_MODELS,
    DEFAULT_OPENROUTER_MODEL,
)
from iris.core.config import IRISConfig


class TestModelAdapters:
    def test_mock_model_adapter_response(self) -> None:
        adapter = MockModelAdapter(responses=["Response 1", "Response 2"])
        resp1 = adapter.generate_response([{"role": "user", "content": "Hi"}])
        assert resp1 == "Response 1"

        resp2 = adapter.generate_response([{"role": "user", "content": "How are you?"}])
        assert resp2 == "Response 2"

        # Cycles back to response 1
        resp3 = adapter.generate_response([{"role": "user", "content": "What next?"}])
        assert resp3 == "Response 1"

    def test_anthropic_model_adapter_requires_key(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            AnthropicModelAdapter(api_key="")

    def test_gemini_model_adapter_requires_key(self) -> None:
        with pytest.raises(ValueError, match="Google API key is required"):
            GeminiModelAdapter(api_key="")

    def test_google_model_adapter_alias(self) -> None:
        assert GoogleModelAdapter is GeminiModelAdapter

    def test_openai_model_adapter_requires_key(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            OpenAIModelAdapter(api_key="")

    def test_create_model_adapter_factory(self) -> None:
        cfg_router = IRISConfig(llm_provider="router")
        model_router = create_model_adapter(cfg_router)
        assert isinstance(model_router, CloudRouterModelAdapter)

        cfg_mock = IRISConfig(llm_provider="mock")
        model_mock = create_model_adapter(cfg_mock)
        assert isinstance(model_mock, MockModelAdapter)

        cfg_anthropic = IRISConfig(
            llm_provider="anthropic",
            anthropic_api_key="test-key",
            llm_model_name="claude-3-5-sonnet",
        )
        model_anthropic = create_model_adapter(cfg_anthropic)
        assert isinstance(model_anthropic, AnthropicModelAdapter)
        assert model_anthropic.model_name == "claude-3-5-sonnet"

        # Google provider with default gemini-3.8-flash model
        cfg_google = IRISConfig(
            llm_provider="google",
            google_api_key="test-key",
        )
        model_google = create_model_adapter(cfg_google)
        assert isinstance(model_google, GeminiModelAdapter)
        assert model_google.model_name == "gemini-3.8-flash"

        # Gemini alias with custom free model
        cfg_gemini = IRISConfig(
            llm_provider="gemini",
            google_api_key="test-key",
            llm_model_name="gemini-2.5-flash",
        )
        model_gemini = create_model_adapter(cfg_gemini)
        assert isinstance(model_gemini, GeminiModelAdapter)
        assert model_gemini.model_name == "gemini-2.5-flash"

    def test_gemini_rejects_paid_or_unapproved_models(self) -> None:
        paid_models = ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-ultra", "gpt-4o"]
        for model in paid_models:
            with pytest.raises(ValueError, match="Strict free-tier allowlist permits only"):
                GeminiModelAdapter(api_key="test-key", model_name=model)

    def test_create_google_model_adapter_missing_key_raises(self) -> None:
        cfg_google_missing = IRISConfig(llm_provider="google", google_api_key="")
        with pytest.raises(ValueError, match="GOOGLE_API_KEY required"):
            create_model_adapter(cfg_google_missing)

    def test_create_openrouter_model_adapter_factory(self) -> None:
        cfg_openrouter = IRISConfig(
            llm_provider="openrouter",
            openrouter_api_key="sk-or-v1-test",
        )
        model_openrouter = create_model_adapter(cfg_openrouter)
        assert isinstance(model_openrouter, OpenRouterModelAdapter)
        assert model_openrouter.model_name == DEFAULT_OPENROUTER_MODEL

    def test_create_openrouter_model_adapter_missing_key_raises(self) -> None:
        cfg_or_missing = IRISConfig(llm_provider="openrouter", openrouter_api_key="")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY required"):
            create_model_adapter(cfg_or_missing)

    def test_openrouter_model_adapter_requires_key(self) -> None:
        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            OpenRouterModelAdapter(api_key="")

    def test_create_ollama_model_adapter_factory(self) -> None:
        cfg_ollama = IRISConfig(llm_provider="ollama")
        model_ollama = create_model_adapter(cfg_ollama)
        assert isinstance(model_ollama, OllamaModelAdapter)
        assert model_ollama.model_name == "llama3.2:1b"
        assert model_ollama.host == "http://localhost:11434"
        assert model_ollama.keep_alive == "30m"

    def test_create_ollama_model_adapter_custom_settings(self) -> None:
        cfg_ollama = IRISConfig(
            llm_provider="ollama",
            llm_model_name="qwen2.5:1.5b",
            ollama_host="http://127.0.0.1:11434/",
            ollama_keep_alive="1h",
        )
        model_ollama = create_model_adapter(cfg_ollama)
        assert isinstance(model_ollama, OllamaModelAdapter)
        assert model_ollama.model_name == "qwen2.5:1.5b"
        assert model_ollama.host == "http://127.0.0.1:11434"
        assert model_ollama.keep_alive == "1h"


class TestGeminiModelAdapterMocked:
    def test_generate_response_single_turn_with_system_prompt(self) -> None:
        adapter = GeminiModelAdapter(api_key="fake-key", model_name="gemini-3.8-flash")

        mock_response = MagicMock()
        mock_response.text = "Hello! I am IRIS, ready to assist."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        adapter._client = mock_client

        messages = [
            {"role": "system", "content": "You are IRIS, a sophisticated British AI."},
            {"role": "user", "content": "Greetings IRIS."},
        ]

        reply = adapter.generate_response(messages)
        assert reply == "Hello! I am IRIS, ready to assist."

        # Verify arguments passed to client.models.generate_content
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.8-flash"
        assert len(call_kwargs["contents"]) == 1
        assert call_kwargs["contents"][0]["role"] == "user"
        assert call_kwargs["contents"][0]["parts"][0]["text"] == "Greetings IRIS."
        assert call_kwargs["config"].system_instruction == "You are IRIS, a sophisticated British AI."

    def test_generate_response_multi_turn_history(self) -> None:
        adapter = GeminiModelAdapter(api_key="fake-key", model_name="gemini-3.8-flash")

        mock_response = MagicMock()
        mock_response.text = "Your project is IRIS."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        adapter._client = mock_client

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "I am working on IRIS."},
            {"role": "assistant", "content": "Understood. How is IRIS going?"},
            {"role": "user", "content": "What was my project name?"},
        ]

        reply = adapter.generate_response(messages)
        assert reply == "Your project is IRIS."

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "I am working on IRIS."
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == "Understood. How is IRIS going?"
        assert contents[2]["role"] == "user"
        assert contents[2]["parts"][0]["text"] == "What was my project name?"

    def test_generate_response_api_error_raises_runtime_error(self) -> None:
        """API errors must be raised cleanly, NOT silently substituted with mock text."""
        adapter = GeminiModelAdapter(api_key="fake-key", model_name="gemini-3.8-flash")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("403 Forbidden")
        adapter._client = mock_client

        with pytest.raises(RuntimeError, match="Gemini generation error: 403 Forbidden"):
            adapter.generate_response([{"role": "user", "content": "Hello"}])

    def test_generate_response_empty_dialogue_fallback(self) -> None:
        adapter = GeminiModelAdapter(api_key="fake-key")
        # Only system message, no user turn
        reply = adapter.generate_response([{"role": "system", "content": "You are IRIS"}])
        assert reply == "How can I help you?"


class TestOpenRouterModelAdapter:
    def test_default_model_and_allowlist(self) -> None:
        adapter = OpenRouterModelAdapter(api_key="fake-key")
        assert adapter.model_name == "openrouter/free"
        assert adapter.model_name in ALLOWED_MODELS
        assert len(ALLOWED_MODELS) == 1
        assert "openrouter/free" in ALLOWED_MODELS

    def test_rejects_paid_or_unapproved_models_at_init(self) -> None:
        unapproved_models = [
            "inclusionai/ling-3.0-flash-vl:free",
            "qwen/qwen3.6-plus:free",
            "google/gemma-4-31b-it:free",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash",
            "openrouter/auto",  # General router (not restricted to free)
            "mistralai/mistral-7b-instruct",
        ]
        for model in unapproved_models:
            with pytest.raises(ValueError, match="Strict allowlist permits only"):
                OpenRouterModelAdapter(api_key="fake-key", model_name=model)

    def test_pre_request_allowlist_check_catches_mutated_model(self) -> None:
        adapter = OpenRouterModelAdapter(api_key="fake-key")
        adapter.model_name = "openai/gpt-4o"  # Unauthorized mutation

        with pytest.raises(ValueError, match="Strict allowlist permits only"):
            adapter.generate_response([{"role": "user", "content": "Hello"}])

    @patch("httpx.post")
    def test_generate_response_success(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Good day! I am IRIS, powered by OpenRouter Free.",
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        adapter = OpenRouterModelAdapter(api_key="fake-key")
        messages = [
            {"role": "system", "content": "You are IRIS."},
            {"role": "user", "content": "Status report."},
        ]

        reply = adapter.generate_response(messages)
        assert reply == "Good day! I am IRIS, powered by OpenRouter Free."

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer fake-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["HTTP-Referer"] == "https://github.com/iris-voice"
        assert kwargs["headers"]["X-Title"] == "IRIS Voice Assistant"
        assert kwargs["json"]["model"] == "openrouter/free"
        assert kwargs["json"]["messages"] == [
            {"role": "system", "content": "You are IRIS."},
            {"role": "user", "content": "Status report."},
        ]

    def test_generate_response_empty_dialogue(self) -> None:
        adapter = OpenRouterModelAdapter(api_key="fake-key")
        reply = adapter.generate_response([{"role": "system", "content": "You are IRIS"}])
        assert reply == "How can I help you?"

    @patch("httpx.post")
    def test_records_selected_model_and_timing_from_response(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello from routed free model.",
                    }
                }
            ],
        }
        mock_post.return_value = mock_response

        adapter = OpenRouterModelAdapter(api_key="fake-key")
        messages = [{"role": "user", "content": "Ping"}]
        reply = adapter.generate_response(messages)

        assert reply == "Hello from routed free model."
        assert adapter.last_selected_model == "meta-llama/llama-3.2-3b-instruct:free"
        assert adapter.last_http_duration_s >= 0.0

    @patch("httpx.post")
    def test_rate_limit_429_halts_without_fallback(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response

        adapter = OpenRouterModelAdapter(api_key="fake-key")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="OpenRouter rate limit exceeded \\(HTTP 429\\).*Halting without fallback"):
            adapter.generate_response(messages)

    @patch("httpx.post")
    def test_model_unavailable_404_halts_without_fallback(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        adapter = OpenRouterModelAdapter(api_key="fake-key")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="OpenRouter model .* not found or unavailable \\(HTTP 404\\).*Halting without fallback"):
            adapter.generate_response(messages)

    @patch("httpx.post")
    def test_timeout_halts_without_fallback(self, mock_post: MagicMock) -> None:
        import httpx

        mock_post.side_effect = httpx.TimeoutException("Read timed out")

        adapter = OpenRouterModelAdapter(api_key="fake-key")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="OpenRouter request timed out.*Halting without fallback"):
            adapter.generate_response(messages)

    @patch("httpx.post")
    def test_http_status_error_halts_without_fallback(self, mock_post: MagicMock) -> None:
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"
        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Gateway", request=req, response=mock_response
        )
        mock_post.return_value = mock_response

        adapter = OpenRouterModelAdapter(api_key="fake-key")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="OpenRouter HTTP error 502.*Halting without fallback"):
            adapter.generate_response(messages)

    @patch("httpx.post")
    def test_empty_choices_raises_runtime_error(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_post.return_value = mock_response

        adapter = OpenRouterModelAdapter(api_key="fake-key")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError, match="OpenRouter returned empty choices"):
            adapter.generate_response(messages)



