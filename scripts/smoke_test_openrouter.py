"""
scripts/smoke_test_openrouter.py
---------------------------------
Single live smoke test for IRIS OpenRouter Free Models Router integration.

Verifies:
1. OPENROUTER_API_KEY is configured in .env (never printed).
2. Strict allowlist enforcement blocks unapproved models (including retired models, paid models, and general routers).
3. Configuration and adapter exact model ID match "openrouter/free" with no silent substitution.
4. Exactly ONE live completion call to OpenRouter with openrouter/free.
5. Logs HTTP status and sanitized error body on failure. Never logs API keys or headers.
6. Response received successfully with no fallback behavior.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from iris.adapters.model import OpenRouterModelAdapter
from iris.adapters.model.openrouter_adapter import ALLOWED_MODELS, DEFAULT_OPENROUTER_MODEL
from iris.core.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_openrouter")

TARGET_MODEL_ID = "openrouter/free"


def run_smoke_test() -> None:
    logger.info("==================================================================")
    logger.info("   Starting IRIS Phase 1 OpenRouter Free Models Router Smoke Test")
    logger.info("==================================================================")

    # 1. Load config and check key presence without revealing secrets
    config = load_config(require_voice=False)
    api_key = config.openrouter_api_key
    if not api_key:
        logger.error("FAIL: No OPENROUTER_API_KEY found in .env or environment.")
        sys.exit(1)

    logger.info("Credential Check: OPENROUTER_API_KEY is present and loaded.")
    logger.info("Configured Provider: %s", config.llm_provider)
    logger.info("Configured Model: %s", config.llm_model_name or DEFAULT_OPENROUTER_MODEL)

    # 2. Pre-flight verification: exact model ID match & no silent substitution
    if config.llm_model_name != TARGET_MODEL_ID:
        logger.error(
            "FAIL: Configured model '%s' does not match required '%s'",
            config.llm_model_name,
            TARGET_MODEL_ID,
        )
        sys.exit(1)

    if DEFAULT_OPENROUTER_MODEL != TARGET_MODEL_ID:
        logger.error(
            "FAIL: Adapter default model '%s' does not match required '%s'",
            DEFAULT_OPENROUTER_MODEL,
            TARGET_MODEL_ID,
        )
        sys.exit(1)

    if ALLOWED_MODELS != frozenset([TARGET_MODEL_ID]):
        logger.error(
            "FAIL: Allowlist %s contains more or less than required '%s'",
            ALLOWED_MODELS,
            TARGET_MODEL_ID,
        )
        sys.exit(1)

    logger.info("Pre-flight: Exact model ID verified. Silent substitution is impossible.")

    # 3. Verify strict allowlist rejection
    logger.info("--- Step 1: Verifying paid/unapproved model rejection ---")
    blocked_models = [
        "openai/gpt-4o",
        "inclusionai/ling-3.0-flash-vl:free",
        "google/gemma-4-31b-it:free",
        "qwen/qwen3.6-plus:free",
        "openrouter/auto",
    ]
    for blocked in blocked_models:
        try:
            OpenRouterModelAdapter(api_key=api_key, model_name=blocked)
            logger.error("FAIL: OpenRouter adapter accepted unapproved model '%s'", blocked)
            sys.exit(1)
        except ValueError as e:
            logger.info("PASS: Blocked unapproved model '%s' as expected: %s", blocked, e)

    # 4. Exactly ONE live call using allowed model
    logger.info("--- Step 2: Executing single live request to %s ---", TARGET_MODEL_ID)
    adapter = OpenRouterModelAdapter(
        api_key=api_key,
        model_name=TARGET_MODEL_ID,
        timeout_seconds=45.0,
    )

    messages = [
        {"role": "system", "content": "You are IRIS, a concise assistant."},
        {"role": "user", "content": "Reply with: IRIS connection OK"},
    ]

    try:
        reply = adapter.generate_response(messages)
        logger.info("OpenRouter Response Received: %s", reply)
        assert len(reply.strip()) > 0, "Empty response from OpenRouter"
    except Exception as exc:
        # exc message contains HTTP status and sanitized response details, no secrets
        logger.error("FAIL: Live OpenRouter request failed: %s", exc)
        sys.exit(1)

    logger.info("==================================================================")
    logger.info("   PASS: OpenRouter Free Models Router Live Smoke Test Succeeded")
    logger.info("==================================================================")


if __name__ == "__main__":
    run_smoke_test()

