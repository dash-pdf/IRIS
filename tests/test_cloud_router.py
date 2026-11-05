"""
tests/test_cloud_router.py
--------------------------
Unit and integration tests for IRIS Cloud Intelligence Router and Deterministic Tools.

Verifies:
1. Deterministic routing for calculations, date/time, and session archive retrieval.
2. Route classification for everyday dialogue (OpenRouter) vs. complex reasoning (Gemini Flash).
3. Free-only enforcement:
   - Rejection of paid models on both OpenRouter and Gemini.
   - Halting without paid fallback on OpenRouter 429 rate limit.
   - Asking user before degrading to a weaker free model on Gemini quota exhaustion / failure.
   - User confirmation handling ("yes" / "sure") executing fallback to standard free model.
   - Clear error reporting for missing credentials.
4. Telemetry and diagnostics capture (route, provider, model, latency, fallback decisions).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iris.adapters.model.cloud_router import (
    DEGRADATION_PROMPT,
    CloudRouterModelAdapter,
)
from iris.adapters.model.gemini_adapter import (
    GEMINI_FREE_MODELS,
    GeminiModelAdapter,
    GeminiQuotaExceededError,
)
from iris.adapters.model.openrouter_adapter import (
    ALLOWED_MODELS,
    OpenRouterModelAdapter,
)
from iris.core.config import IRISConfig
from iris.core.router import (
    RouteType,
    classify_request,
    is_user_confirmation,
    try_evaluate_calculation,
    try_evaluate_datetime,
    try_evaluate_memory,
)
from iris.services.archive import ConversationArchive


# ── 1. Deterministic Tool Evaluation Tests ────────────────────────────────────

class TestDeterministicTools:
    def test_calculation_arithmetic_expressions(self) -> None:
        assert try_evaluate_calculation("what is 25 * 4") == "The answer is 100."
        assert try_evaluate_calculation("calculate 144 / 12") == "The answer is 12."
        assert try_evaluate_calculation("15 + 27") == "The answer is 42."
        assert try_evaluate_calculation("50 - 18") == "The answer is 32."
        assert try_evaluate_calculation("2 ** 8") == "The answer is 256."
        assert try_evaluate_calculation("2 ^ 8") == "The answer is 256."

    def test_calculation_natural_language_words(self) -> None:
        assert try_evaluate_calculation("what is 25 times 4?") == "The answer is 100."
        assert try_evaluate_calculation("144 divided by 12") == "The answer is 12."
        assert try_evaluate_calculation("what is 15 plus 27") == "The answer is 42."
        assert try_evaluate_calculation("what is 2 to the power of 8") == "The answer is 256."

    def test_calculation_percentages_and_roots(self) -> None:
        assert try_evaluate_calculation("what is 15% of 200") == "15% of 200 is 30."
        assert try_evaluate_calculation("calculate 20 percent of 150") == "20% of 150 is 30."
        assert try_evaluate_calculation("what is the square root of 144?") == "The square root of 144 is 12."

    def test_calculation_algebraic_identities(self) -> None:
        assert try_evaluate_calculation("what is (a+b)^2 expanded?") == "The standard expansion is (a + b)² = a² + 2ab + b²."
        assert try_evaluate_calculation("what is (a - b)^2?") == "The standard expansion is (a - b)² = a² - 2ab + b²."
        assert try_evaluate_calculation("what is a^2 - b^2?") == "The standard expansion is a² - b² = (a - b)(a + b)."

    def test_calculation_division_by_zero(self) -> None:
        assert try_evaluate_calculation("what is 10 / 0") == "Division by zero is undefined."

    def test_calculation_ignores_non_arithmetic_text(self) -> None:
        assert try_evaluate_calculation("hello iris how are you") is None
        assert try_evaluate_calculation("what is the weather today") is None

    def test_datetime_queries(self) -> None:
        fixed_dt = datetime(2026, 11, 5, 14, 30)  # Thursday, 5 Nov 2026 2:30 PM
        time_reply = try_evaluate_datetime("what time is it?", now_dt=fixed_dt)
        assert time_reply == "It is currently 2:30 PM."

        date_reply = try_evaluate_datetime("what is today's date?", now_dt=fixed_dt)
        assert date_reply == "Today is Thursday, 5 November 2026."

        day_reply = try_evaluate_datetime("what day of the week is it?", now_dt=fixed_dt)
        assert day_reply == "Today is Thursday."

        year_reply = try_evaluate_datetime("what year is this?", now_dt=fixed_dt)
        assert year_reply == "The year is 2026."

        month_reply = try_evaluate_datetime("what month is it?", now_dt=fixed_dt)
        assert month_reply == "The current month is November."

    def test_datetime_ignores_unrelated(self) -> None:
        assert try_evaluate_datetime("tell me a story about time") is None

    def test_memory_retrieval_with_session_archive(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "memory" / "conversations"
        archive = ConversationArchive(archive_dir=archive_dir)
        archive.open_session()
        archive.write_turn("user", "My favorite color is emerald green.")
        archive.write_turn("assistant", "Noted, emerald green is a lovely color.")
        archive.close_session()

        reply = try_evaluate_memory("what did we talk about earlier?", archive_dir=archive_dir)
        assert reply is not None
        assert "Key topics included" in reply
        assert "My favorite color is emerald green" in reply

        last_q_reply = try_evaluate_memory("what was the last thing I asked?", archive_dir=archive_dir)
        assert last_q_reply == 'In your last recorded turn, you asked: "My favorite color is emerald green.".'

    def test_memory_retrieval_empty_archive(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_archive"
        empty_dir.mkdir(parents=True)
        reply = try_evaluate_memory("what did we discuss last time?", archive_dir=empty_dir)
        assert "I don't have any prior conversation sessions recorded" in reply


# ── 2. Request Classifier Tests ───────────────────────────────────────────────

class TestRequestClassifier:
    def test_classify_exact_calculation(self) -> None:
        d = classify_request("what is 50 * 20?")
        assert d.route == RouteType.CALCULATION
        assert d.direct_response == "The answer is 1000."
        assert d.target_provider == "deterministic"

    def test_classify_datetime_query(self) -> None:
        d = classify_request("what time is it?")
        assert d.route == RouteType.DATETIME
        assert d.target_provider == "deterministic"
        assert d.direct_response is not None

    def test_classify_complex_reasoning_keywords(self) -> None:
        complex_queries = [
            "Analyze the architectural trade-offs between monolithic and microservice designs.",
            "Explain in detail why quicksort has an average time complexity of O(n log n).",
            "Compare and contrast supervised and unsupervised machine learning.",
            "Write a Python function to implement binary search step-by-step.",
            "What are the philosophical implications of artificial general intelligence?",
            "Can you deduce the solution to this logic puzzle step by step?",
        ]
        for q in complex_queries:
            d = classify_request(q)
            assert d.route == RouteType.COMPLEX_REASONING, f"Expected COMPLEX_REASONING for: '{q}'"
            assert d.target_provider == "google"
            assert d.target_model == "gemini-3.8-flash"

    def test_classify_everyday_conversation(self) -> None:
        everyday_queries = [
            "Hello IRIS, how are you today?",
            "Good morning!",
            "Who are you and what can you do?",
            "Thanks for the help.",
            "Tell me a short joke.",
            "I'm feeling a bit tired today.",
        ]
        for q in everyday_queries:
            d = classify_request(q)
            assert d.route == RouteType.CONVERSATION, f"Expected CONVERSATION for: '{q}'"
            assert d.target_provider == "openrouter"
            assert d.target_model == "openrouter/free"

    def test_user_confirmation_patterns(self) -> None:
        assert is_user_confirmation("yes") is True
        assert is_user_confirmation("yes please") is True
        assert is_user_confirmation("Sure.") is True
        assert is_user_confirmation("please") is True
        assert is_user_confirmation("go ahead") is True
        assert is_user_confirmation("no, cancel that") is False


# ── 3. Cloud Router Model Adapter & Free-Only Enforcement Tests ───────────────

class TestCloudRouterModelAdapter:
    def test_deterministic_calculation_route_execution(self) -> None:
        cfg = IRISConfig(llm_provider="router")
        router = CloudRouterModelAdapter(config=cfg)

        reply = router.generate_response([{"role": "user", "content": "calculate 25 * 4"}])
        assert reply == "The answer is 100."
        assert router.last_route == "calculation"
        assert router.last_provider_name == "deterministic"
        assert router.last_selected_model == "python_math"
        assert router.last_fallback_decision is None
        assert router.last_duration_s >= 0.0

    def test_deterministic_datetime_route_execution(self) -> None:
        cfg = IRISConfig(llm_provider="router")
        router = CloudRouterModelAdapter(config=cfg)

        reply = router.generate_response([{"role": "user", "content": "what year is it?"}])
        assert "The year is" in reply
        assert router.last_route == "datetime"
        assert router.last_provider_name == "deterministic"

    def test_everyday_conversation_routed_to_openrouter(self) -> None:
        cfg = IRISConfig(llm_provider="router", openrouter_api_key="mock-key")
        mock_openrouter = MagicMock(spec=OpenRouterModelAdapter)
        mock_openrouter.model_name = "openrouter/free"
        mock_openrouter.generate_response.return_value = "Good morning! How may I assist you?"

        router = CloudRouterModelAdapter(config=cfg, openrouter_adapter=mock_openrouter)
        reply = router.generate_response([{"role": "user", "content": "Hello IRIS, good morning!"}])

        assert reply == "Good morning! How may I assist you?"
        assert router.last_route == "conversation"
        assert router.last_provider_name == "openrouter"
        assert router.last_selected_model == "openrouter/free"
        mock_openrouter.generate_response.assert_called_once()

    def test_complex_reasoning_routed_to_gemini_flash(self) -> None:
        cfg = IRISConfig(llm_provider="router", google_api_key="mock-key")
        mock_gemini = MagicMock(spec=GeminiModelAdapter)
        mock_gemini.model_name = "gemini-3.8-flash"
        mock_gemini.generate_response.return_value = "Here is a step-by-step architectural analysis..."

        router = CloudRouterModelAdapter(config=cfg, gemini_adapter=mock_gemini)
        reply = router.generate_response([
            {"role": "user", "content": "Analyze the architectural differences between TCP and UDP in detail."}
        ])

        assert reply == "Here is a step-by-step architectural analysis..."
        assert router.last_route == "complex_reasoning"
        assert router.last_provider_name == "google"
        assert router.last_selected_model == "gemini-3.8-flash"
        mock_gemini.generate_response.assert_called_once()

    def test_missing_openrouter_credentials_reported_cleanly(self) -> None:
        cfg = IRISConfig(llm_provider="router", openrouter_api_key="")
        router = CloudRouterModelAdapter(config=cfg)

        reply = router.generate_response([{"role": "user", "content": "How are you today?"}])
        assert "OPENROUTER_API_KEY" in reply
        assert "not configured" in reply
        assert router.last_fallback_decision == "OPENROUTER_API_KEY not configured"

    def test_missing_gemini_credentials_prompts_user_without_silent_switch(self) -> None:
        cfg = IRISConfig(llm_provider="router", google_api_key="")
        router = CloudRouterModelAdapter(config=cfg)

        reply = router.generate_response([
            {"role": "user", "content": "Analyze and compare sorting algorithms in detail."}
        ])
        assert "GOOGLE_API_KEY" in reply
        assert "Would you like me to try answering this with our standard conversational model instead?" in reply
        assert "prompting user" in router.last_fallback_decision

    def test_openrouter_rate_limit_halts_without_paid_fallback(self) -> None:
        cfg = IRISConfig(llm_provider="router", openrouter_api_key="mock-key")
        mock_openrouter = MagicMock(spec=OpenRouterModelAdapter)
        mock_openrouter.generate_response.side_effect = RuntimeError("OpenRouter rate limit exceeded (HTTP 429)")

        router = CloudRouterModelAdapter(config=cfg, openrouter_adapter=mock_openrouter)
        reply = router.generate_response([{"role": "user", "content": "Tell me a joke."}])

        # Must report free capacity exhausted and halt without spending money
        assert "OpenRouter free capacity is temporarily unavailable" in reply
        assert "will not fall back to paid models" in reply
        assert "halted without paid fallback" in router.last_fallback_decision

    def test_gemini_quota_exhausted_asks_user_before_degradation(self) -> None:
        cfg = IRISConfig(llm_provider="router", google_api_key="mock-key")
        mock_gemini = MagicMock(spec=GeminiModelAdapter)
        mock_gemini.generate_response.side_effect = GeminiQuotaExceededError("Free-tier quota exceeded")

        router = CloudRouterModelAdapter(config=cfg, gemini_adapter=mock_gemini)
        reply = router.generate_response([
            {"role": "user", "content": "Deep dive into quantum computing theory and analyze its implications."}
        ])

        assert reply == DEGRADATION_PROMPT
        assert "Gemini quota exceeded" in router.last_fallback_decision

    def test_user_confirms_degradation_after_gemini_quota_exhaustion(self) -> None:
        cfg = IRISConfig(llm_provider="router", openrouter_api_key="mock-key", google_api_key="mock-key")
        mock_openrouter = MagicMock(spec=OpenRouterModelAdapter)
        mock_openrouter.model_name = "openrouter/free"
        mock_openrouter.generate_response.return_value = "Quantum computing relies on superposition and entanglement."

        router = CloudRouterModelAdapter(config=cfg, openrouter_adapter=mock_openrouter)

        messages = [
            {"role": "user", "content": "Explain quantum computing theory in detail."},
            {"role": "assistant", "content": DEGRADATION_PROMPT},
            {"role": "user", "content": "Yes, please."},
        ]

        reply = router.generate_response(messages)
        assert reply == "Quantum computing relies on superposition and entanglement."
        assert router.last_route == "conversation"
        assert router.last_provider_name == "openrouter"
        assert "User confirmed fallback to openrouter/free" in router.last_fallback_decision
        mock_openrouter.generate_response.assert_called_once()

    def test_free_tier_allowlists_block_paid_models(self) -> None:
        # OpenRouter blocks anything not in ALLOWED_MODELS
        assert ALLOWED_MODELS == frozenset(["openrouter/free"])
        with pytest.raises(ValueError, match="Strict allowlist permits only"):
            OpenRouterModelAdapter(api_key="test-key", model_name="openai/gpt-4o")

        # Gemini blocks paid Pro/Ultra models
        assert "gemini-3.8-flash" in GEMINI_FREE_MODELS
        assert "gemini-2.5-flash" in GEMINI_FREE_MODELS
        with pytest.raises(ValueError, match="Strict free-tier allowlist permits only"):
            GeminiModelAdapter(api_key="test-key", model_name="gemini-2.5-pro")

        with pytest.raises(ValueError, match="Strict free-tier allowlist permits only"):
            GeminiModelAdapter(api_key="test-key", model_name="gemini-ultra")


# ── 4. Assistant Integration with Cloud Router ────────────────────────────────

class TestCloudRouterAssistantIntegration:
    def test_assistant_orchestration_with_cloud_router(self, tmp_path: Path) -> None:
        archive_dir = tmp_path / "conversations"
        cfg = IRISConfig(
            llm_provider="router",
            tts_provider="mock",
            stt_provider="mock",
            archive_dir=archive_dir,
            openrouter_api_key="mock-key",
            google_api_key="mock-key",
            cooldown_ms=50,
        )

        mock_openrouter = MagicMock(spec=OpenRouterModelAdapter)
        mock_openrouter.model_name = "openrouter/free"
        mock_openrouter.generate_response.return_value = "Hello! I am IRIS, all systems online."

        mock_gemini = MagicMock(spec=GeminiModelAdapter)
        mock_gemini.model_name = "gemini-3.8-flash"
        mock_gemini.generate_response.return_value = "Distributed consensus requires quorum and leader election."

        router = CloudRouterModelAdapter(
            config=cfg,
            openrouter_adapter=mock_openrouter,
            gemini_adapter=mock_gemini,
            archive_dir=archive_dir,
        )

        from iris.adapters.stt.mock_stt import MockSTT
        from iris.adapters.tts.mock_tts import MockTTS
        from iris.core.assistant import IRISAssistant

        assistant = IRISAssistant(
            config=cfg,
            model_adapter=router,
            tts_adapter=MockTTS(),
            stt_adapter=MockSTT(),
        )

        assistant.start()

        # Turn 1: Exact calculation
        calc_reply = assistant.process_text_turn("what is 12 * 12?")
        assert calc_reply == "The answer is 144."
        assert assistant.last_turn_timings["route"] == "calculation"
        assert assistant.last_turn_timings["provider_name"] == "deterministic"
        assert assistant.last_turn_timings["selected_model"] == "python_math"

        # Turn 2: Everyday conversation
        chat_reply = assistant.process_text_turn("Hello IRIS, how are you?")
        assert chat_reply == "Hello! I am IRIS, all systems online."
        assert assistant.last_turn_timings["route"] == "conversation"
        assert assistant.last_turn_timings["provider_name"] == "openrouter"

        # Turn 3: Complex reasoning
        complex_reply = assistant.process_text_turn(
            "Analyze the trade-offs of distributed consensus algorithms in detail."
        )
        assert complex_reply == "Distributed consensus requires quorum and leader election."
        assert assistant.last_turn_timings["route"] == "complex_reasoning"
        assert assistant.last_turn_timings["provider_name"] == "google"

        assistant.stop()

        # Verify Markdown conversation archive recorded all turns
        md_files = list(archive_dir.glob("**/*.md"))
        assert len(md_files) == 1
        archived_text = md_files[0].read_text(encoding="utf-8")
        assert "what is 12 * 12?" in archived_text
        assert "The answer is 144." in archived_text
        assert "Hello IRIS, how are you?" in archived_text
        assert "Analyze the trade-offs" in archived_text

