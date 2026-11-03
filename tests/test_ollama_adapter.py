"""
tests/test_ollama_adapter.py
----------------------------
Unit tests for iris/adapters/model/ollama_adapter.py (Local Ollama Adapter).
All tests are completely offline and use mocked HTTP responses.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iris.adapters.model.ollama_adapter import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_MODEL,
    OllamaModelAdapter,
)
from iris.adapters.tts.mock_tts import MockTTS
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig
from iris.services.conversation import IRIS_LOCAL_SYSTEM_PROMPT, IRIS_SYSTEM_PROMPT


class MockHTTPResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self.body = body.encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> MockHTTPResponse:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class TestOllamaModelAdapter:
    def test_init_defaults(self) -> None:
        adapter = OllamaModelAdapter()
        assert adapter.host == DEFAULT_OLLAMA_HOST
        assert adapter.model_name == DEFAULT_OLLAMA_MODEL
        assert adapter.timeout_seconds == 60.0
        assert adapter.keep_alive == DEFAULT_OLLAMA_KEEP_ALIVE
        assert adapter.last_metrics is None
        assert adapter.chat_url == f"{DEFAULT_OLLAMA_HOST}/api/chat"

    def test_init_custom_settings_and_strips_trailing_slash(self) -> None:
        adapter = OllamaModelAdapter(
            host="http://192.168.1.50:11434/",
            model_name="qwen2.5:1.5b",
            timeout_seconds=45.0,
            keep_alive="1h",
        )
        assert adapter.host == "http://192.168.1.50:11434"
        assert adapter.model_name == "qwen2.5:1.5b"
        assert adapter.timeout_seconds == 45.0
        assert adapter.keep_alive == "1h"
        assert adapter.chat_url == "http://192.168.1.50:11434/api/chat"

    @patch("urllib.request.urlopen")
    def test_generate_response_single_turn_success(self, mock_urlopen: MagicMock) -> None:
        mock_data = {
            "model": "llama3.2:1b",
            "message": {
                "role": "assistant",
                "content": "All systems nominal, sir.",
            },
            "done": True,
        }
        mock_urlopen.return_value = MockHTTPResponse(json.dumps(mock_data))

        adapter = OllamaModelAdapter(model_name="llama3.2:1b")
        messages = [
            {"role": "user", "content": "Status check."},
        ]

        resp = adapter.generate_response(messages)
        assert resp == "All systems nominal, sir."
        assert adapter.last_selected_model == "llama3.2:1b"
        assert adapter.last_http_duration_s >= 0.0

        # Verify request details
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/chat"
        assert req.get_method() == "POST"
        assert req.headers.get("Content-type") == "application/json"

        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "llama3.2:1b"
        assert body["stream"] is False
        assert body["keep_alive"] == "30m"
        assert body["messages"] == [{"role": "user", "content": "Status check."}]

    @patch("urllib.request.urlopen")
    def test_generate_response_multi_turn_with_system_prompt(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_data = {
            "model": "llama3.2:1b",
            "message": {
                "role": "assistant",
                "content": "Understood, proceeding now.",
            },
            "done": True,
        }
        mock_urlopen.return_value = MockHTTPResponse(json.dumps(mock_data))

        adapter = OllamaModelAdapter()
        messages = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
        ]

        resp = adapter.generate_response(messages, system_prompt="You are IRIS.")
        assert resp == "Understood, proceeding now."

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["messages"][0] == {"role": "system", "content": "You are IRIS."}
        assert body["messages"][1] == {"role": "user", "content": "Turn 1"}
        assert body["messages"][2] == {"role": "assistant", "content": "Reply 1"}
        assert body["messages"][3] == {"role": "user", "content": "Turn 2"}

    def test_generate_response_empty_messages_returns_greeting(self) -> None:
        adapter = OllamaModelAdapter()
        assert adapter.generate_response([]) == "How can I help you?"
        assert adapter.generate_response([{"role": "system", "content": "Prompt"}]) == "How can I help you?"

    @patch("urllib.request.urlopen")
    def test_generate_response_empty_content_returns_default(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_data = {
            "model": "llama3.2:1b",
            "message": {"role": "assistant", "content": "   "},
            "done": True,
        }
        mock_urlopen.return_value = MockHTTPResponse(json.dumps(mock_data))

        adapter = OllamaModelAdapter()
        resp = adapter.generate_response([{"role": "user", "content": "Test"}])
        assert resp == "Understood."

    @patch("urllib.request.urlopen")
    def test_generate_response_connection_refused_raises_actionable_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.side_effect = urllib.error.URLError(
            ConnectionRefusedError(61, "Connection refused")
        )

        adapter = OllamaModelAdapter()
        with pytest.raises(ConnectionError, match="Could not connect to Ollama server"):
            adapter.generate_response([{"role": "user", "content": "Hello"}])

    @patch("urllib.request.urlopen")
    def test_generate_response_timeout_raises_timeout_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("timed out")

        adapter = OllamaModelAdapter(timeout_seconds=5.0)
        with pytest.raises(TimeoutError, match="timed out after 5.0s"):
            adapter.generate_response([{"role": "user", "content": "Hello"}])

    @patch("urllib.request.urlopen")
    def test_generate_response_missing_model_404_raises_clear_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        err_fp = io.BytesIO(b'{"error": "model llama3.2:1b not found"}')
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs={},  # type: ignore
            fp=err_fp,
        )

        adapter = OllamaModelAdapter(model_name="llama3.2:1b")
        with pytest.raises(ValueError, match="ollama pull llama3.2:1b"):
            adapter.generate_response([{"role": "user", "content": "Hello"}])

    @patch("urllib.request.urlopen")
    def test_generate_response_http_500_raises_runtime_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        err_fp = io.BytesIO(b"Internal Server Error")
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:11434/api/chat",
            code=500,
            msg="Internal Server Error",
            hdrs={},  # type: ignore
            fp=err_fp,
        )

        adapter = OllamaModelAdapter()
        with pytest.raises(RuntimeError, match="Ollama HTTP error 500"):
            adapter.generate_response([{"role": "user", "content": "Hello"}])

    @patch("urllib.request.urlopen")
    def test_generate_response_invalid_json_raises_value_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = MockHTTPResponse("not a valid json")

        adapter = OllamaModelAdapter()
        with pytest.raises(ValueError, match="Invalid JSON received from Ollama"):
            adapter.generate_response([{"role": "user", "content": "Hello"}])

    @patch("urllib.request.urlopen")
    def test_generate_response_missing_message_structure_raises_runtime_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = MockHTTPResponse('{"model": "llama3.2:1b"}')

        adapter = OllamaModelAdapter()
        with pytest.raises(RuntimeError, match="missing expected 'message.content'"):
            adapter.generate_response([{"role": "user", "content": "Hello"}])

    @patch("urllib.request.urlopen")
    def test_generate_response_telemetry_metrics_extraction(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_data = {
            "model": "llama3.2:1b",
            "message": {"role": "assistant", "content": "Direct expansion."},
            "done": True,
            "total_duration": 5000000000,
            "load_duration": 1500000000,
            "prompt_eval_count": 30,
            "prompt_eval_duration": 500000000,
            "eval_count": 60,
            "eval_duration": 3000000000,
        }
        mock_urlopen.return_value = MockHTTPResponse(json.dumps(mock_data))

        adapter = OllamaModelAdapter()
        resp = adapter.generate_response([{"role": "user", "content": "(a+b)^2"}])
        assert resp == "Direct expansion."

        metrics = adapter.last_metrics
        assert metrics is not None
        assert metrics["total_duration_s"] == pytest.approx(5.0)
        assert metrics["load_duration_s"] == pytest.approx(1.5)
        assert metrics["prompt_eval_count"] == 30
        assert metrics["prompt_eval_duration_s"] == pytest.approx(0.5)
        assert metrics["eval_count"] == 60
        assert metrics["eval_duration_s"] == pytest.approx(3.0)
        assert metrics["tokens_per_second"] == pytest.approx(20.0)

    @patch("urllib.request.urlopen")
    def test_generate_response_telemetry_zero_or_missing_duration_safe(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_data = {
            "model": "llama3.2:1b",
            "message": {"role": "assistant", "content": "Done."},
            "done": True,
            "eval_count": 10,
            "eval_duration": 0,
        }
        mock_urlopen.return_value = MockHTTPResponse(json.dumps(mock_data))

        adapter = OllamaModelAdapter()
        resp = adapter.generate_response([{"role": "user", "content": "Hi"}])
        assert resp == "Done."

        metrics = adapter.last_metrics
        assert metrics is not None
        assert metrics["eval_count"] == 10
        assert metrics["eval_duration_s"] == 0.0
        assert metrics["tokens_per_second"] == 0.0
        assert metrics["load_duration_s"] is None

    @patch("urllib.request.urlopen")
    def test_generate_response_telemetry_resets_on_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        adapter = OllamaModelAdapter()
        adapter.last_metrics = {"eval_count": 50}

        mock_urlopen.side_effect = urllib.error.URLError("timed out")
        with pytest.raises(TimeoutError):
            adapter.generate_response([{"role": "user", "content": "Hi"}])

        assert adapter.last_metrics is None


class TestOllamaAssistantIntegration:
    @patch("urllib.request.urlopen")
    def test_assistant_orchestration_with_ollama(
        self, mock_urlopen: MagicMock, tmp_path: Path
    ) -> None:
        mock_data = {
            "model": "llama3.2:1b",
            "message": {
                "role": "assistant",
                "content": "All systems nominal. Ready to assist.",
            },
            "done": True,
        }
        mock_urlopen.return_value = MockHTTPResponse(json.dumps(mock_data))

        cfg = IRISConfig(
            llm_provider="ollama",
            llm_model_name="llama3.2:1b",
            archive_dir=tmp_path / "conversations",
            tts_provider="mock",
            stt_provider="mock",
        )
        adapter = OllamaModelAdapter(model_name="llama3.2:1b")
        tts = MockTTS()
        assistant = IRISAssistant(
            config=cfg,
            model_adapter=adapter,
            tts_adapter=tts,
        )
        assistant.start()

        resp = assistant.process_text_turn("Status please.")
        assert resp == "All systems nominal. Ready to assist."
        assert tts.last_synthesized_text == "All systems nominal. Ready to assist."

        # Verify archive recorded turn
        md_files = list((tmp_path / "conversations").glob("**/*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "Status please." in content
        assert "All systems nominal. Ready to assist." in content

        assistant.stop()

    def test_assistant_selects_local_system_prompt_for_ollama(self) -> None:
        cfg_ollama = IRISConfig(llm_provider="ollama", tts_provider="mock", stt_provider="mock")
        assistant_ollama = IRISAssistant(config=cfg_ollama)
        assert assistant_ollama.conversation.system_prompt == IRIS_LOCAL_SYSTEM_PROMPT
        assert "(a+b)^2 = a^2 + 2ab + b^2" in assistant_ollama.conversation.system_prompt

        cfg_mock = IRISConfig(llm_provider="mock", tts_provider="mock", stt_provider="mock")
        assistant_mock = IRISAssistant(config=cfg_mock)
        assert assistant_mock.conversation.system_prompt == IRIS_SYSTEM_PROMPT

    @patch("urllib.request.urlopen")
    def test_assistant_captures_model_metrics_in_turn_timings(
        self, mock_urlopen: MagicMock, tmp_path: Path
    ) -> None:
        mock_data = {
            "model": "llama3.2:1b",
            "message": {"role": "assistant", "content": "a^2 + 2ab + b^2"},
            "done": True,
            "total_duration": 4000000000,
            "load_duration": 1000000000,
            "prompt_eval_count": 25,
            "prompt_eval_duration": 500000000,
            "eval_count": 50,
            "eval_duration": 2500000000,
        }
        mock_urlopen.return_value = MockHTTPResponse(json.dumps(mock_data))

        cfg = IRISConfig(
            llm_provider="ollama",
            llm_model_name="llama3.2:1b",
            archive_dir=tmp_path / "conversations",
            tts_provider="mock",
            stt_provider="mock",
        )
        adapter = OllamaModelAdapter(model_name="llama3.2:1b")
        assistant = IRISAssistant(config=cfg, model_adapter=adapter, tts_adapter=MockTTS())
        assistant.start()

        resp = assistant.process_text_turn("What is (a+b)^2?")
        assert resp == "a^2 + 2ab + b^2"

        timings = assistant.last_turn_timings
        assert "model_metrics" in timings
        metrics = timings["model_metrics"]
        assert metrics["tokens_per_second"] == pytest.approx(20.0)
        assert metrics["load_duration_s"] == pytest.approx(1.0)
        assert metrics["prompt_eval_count"] == 25
        assert metrics["eval_count"] == 50

        assistant.stop()

