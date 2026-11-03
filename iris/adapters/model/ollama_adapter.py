"""
iris/adapters/model/ollama_adapter.py
-------------------------------------
Local Ollama model adapter for IRIS using Ollama's local REST API.
Operates completely locally and offline. No paid APIs, external tokens,
or hosted fallbacks.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from iris.adapters.model.base import ModelAdapter

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL: str = "llama3.2:1b"
DEFAULT_OLLAMA_KEEP_ALIVE: str = "30m"


class OllamaModelAdapter(ModelAdapter):
    """
    Adapter for local LLM inference via Ollama (/api/chat).
    """

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        timeout_seconds: float = 60.0,
        keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE,
    ) -> None:
        if not host:
            host = DEFAULT_OLLAMA_HOST
        if not model_name:
            model_name = DEFAULT_OLLAMA_MODEL

        self.host = host.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive if keep_alive is not None else DEFAULT_OLLAMA_KEEP_ALIVE
        self.last_selected_model: str | None = None
        self.last_http_duration_s: float = 0.0
        self.last_metrics: dict[str, Any] | None = None

    @property
    def chat_url(self) -> str:
        return f"{self.host}/api/chat"

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

        if extracted_system:
            api_messages.insert(0, {"role": "system", "content": extracted_system})

        if not api_messages or all(m.get("role") == "system" for m in api_messages):
            return "How can I help you?"

        logger.info(
            "Calling local Ollama model (%s) at %s with %d turns",
            self.model_name,
            self.host,
            len(api_messages),
        )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=self.chat_url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            self.last_http_duration_s = time.perf_counter() - t_start
            self.last_metrics = None
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                pass

            if exc.code == 404:
                raise ValueError(
                    f"Ollama model '{self.model_name}' not found at {self.host} (HTTP 404). "
                    f"Pull the model first by running: ollama pull {self.model_name}"
                ) from exc

            raise RuntimeError(
                f"Ollama HTTP error {exc.code} from {self.host}: {err_body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            self.last_http_duration_s = time.perf_counter() - t_start
            self.last_metrics = None
            reason_str = str(exc.reason)
            if "timed out" in reason_str.lower() or isinstance(exc.reason, TimeoutError):
                raise TimeoutError(
                    f"Ollama request to {self.host} timed out after {self.timeout_seconds}s."
                ) from exc

            raise ConnectionError(
                f"Could not connect to Ollama server at {self.host} ({reason_str}). "
                "Ensure Ollama is running (e.g. run 'ollama serve' or open Ollama)."
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            self.last_http_duration_s = time.perf_counter() - t_start
            self.last_metrics = None
            raise TimeoutError(
                f"Ollama request to {self.host} timed out after {self.timeout_seconds}s."
            ) from exc

        self.last_http_duration_s = time.perf_counter() - t_start

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            self.last_metrics = None
            raise ValueError(
                f"Invalid JSON received from Ollama at {self.host}: {raw_body[:200]}"
            ) from exc

        self.last_selected_model = data.get("model", self.model_name)

        # Parse Ollama generation telemetry (durations in nanoseconds -> seconds)
        load_duration_ns = data.get("load_duration")
        prompt_eval_count = data.get("prompt_eval_count")
        prompt_eval_duration_ns = data.get("prompt_eval_duration")
        eval_count = data.get("eval_count")
        eval_duration_ns = data.get("eval_duration")
        total_duration_ns = data.get("total_duration")

        load_duration_s = (
            (load_duration_ns / 1e9)
            if isinstance(load_duration_ns, (int, float))
            else None
        )
        prompt_eval_duration_s = (
            (prompt_eval_duration_ns / 1e9)
            if isinstance(prompt_eval_duration_ns, (int, float))
            else None
        )
        eval_duration_s = (
            (eval_duration_ns / 1e9)
            if isinstance(eval_duration_ns, (int, float))
            else None
        )
        total_duration_s = (
            (total_duration_ns / 1e9)
            if isinstance(total_duration_ns, (int, float))
            else None
        )

        tokens_per_second: float = 0.0
        if (
            isinstance(eval_count, (int, float))
            and isinstance(eval_duration_s, (int, float))
            and eval_duration_s > 0
        ):
            tokens_per_second = eval_count / eval_duration_s

        self.last_metrics = {
            "total_duration_s": total_duration_s,
            "load_duration_s": load_duration_s,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration_s": prompt_eval_duration_s,
            "eval_count": eval_count,
            "eval_duration_s": eval_duration_s,
            "tokens_per_second": tokens_per_second,
        }

        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise RuntimeError(
                f"Ollama response missing expected 'message.content' structure: {raw_body[:200]}"
            )

        content = message.get("content", "").strip()
        if not content:
            return "Understood."

        return content

