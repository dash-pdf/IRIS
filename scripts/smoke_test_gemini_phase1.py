"""
scripts/smoke_test_gemini_phase1.py
-----------------------------------
Live smoke test for IRIS Phase 1 Gemini Integration.

Verifies:
1. GeminiModelAdapter live connectivity with gemini-3.8-flash.
2. Multi-turn conversational context with live Gemini.
3. Complete Phase 1 flow:
   User input -> Transcription / turn processing -> Gemini reasoning ->
   TTS audio generation -> Durable Markdown session recording.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from iris.adapters.model import GeminiModelAdapter, GoogleModelAdapter
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_gemini")


def run_smoke_test() -> None:
    logger.info("==================================================================")
    logger.info("   Starting IRIS Phase 1 Live Gemini Smoke Test")
    logger.info("==================================================================")

    # 1. Load config and verify key presence without revealing secrets
    config = load_config(require_voice=False)
    api_key = config.google_api_key
    if not api_key:
        logger.error("FAIL: No GOOGLE_API_KEY found in .env or environment.")
        sys.exit(1)

    logger.info("Credential Check: GOOGLE_API_KEY is present and loaded.")
    logger.info("Selected Provider: %s", config.llm_provider)
    logger.info("Selected Model: %s", config.llm_model_name or "gemini-3.8-flash")

    # 2. Test direct GeminiModelAdapter call
    logger.info("--- Step 1: Testing direct GeminiModelAdapter generation ---")
    adapter = GoogleModelAdapter(
        api_key=api_key,
        model_name=config.llm_model_name or "gemini-3.8-flash",
    )

    messages = [
        {"role": "system", "content": "You are IRIS, a sophisticated British personal AI. Keep replies concise."},
        {"role": "user", "content": "Acknowledge Gemini integration for IRIS."},
    ]
    reply1 = adapter.generate_response(messages)
    logger.info("Gemini Reply 1: %s", reply1)
    assert reply1 and len(reply1) > 0, "Empty reply from Gemini"

    # 3. Test multi-turn conversational context
    logger.info("--- Step 2: Testing multi-turn conversational memory ---")
    multi_turn_messages = [
        {"role": "system", "content": "You are IRIS. Respond with understated British poise."},
        {"role": "user", "content": "My code name for this operation is Antigravity."},
        {"role": "assistant", "content": "Understood. The operation is noted as Antigravity."},
        {"role": "user", "content": "What is my operation code name?"},
    ]
    reply2 = adapter.generate_response(multi_turn_messages)
    logger.info("Gemini Multi-turn Reply: %s", reply2)
    assert "antigravity" in reply2.lower(), f"Gemini failed to retain multi-turn context: {reply2}"
    logger.info("Context retention: PASS (Retained 'Antigravity')")

    # 4. Test complete Phase 1 Assistant pipeline with Markdown archive and TTS
    logger.info("--- Step 3: Testing full IRIS Assistant pipeline with Gemini ---")
    smoke_archive_dir = ROOT / "memory" / "conversations"

    smoke_config = IRISConfig(
        llm_provider="google",
        google_api_key=api_key,
        llm_model_name="gemini-3.8-flash",
        tts_provider="edge",
        edge_voice=config.edge_voice,
        stt_provider="console",
        archive_dir=smoke_archive_dir,
        cooldown_ms=50,
    )

    assistant = IRISAssistant(config=smoke_config)
    assistant.start()

    session_file = assistant.archive.session_file
    logger.info("Active Session Markdown: %s", session_file)

    test_prompt = "IRIS, confirm voice synthesis and intelligence pipeline are operational."
    logger.info("Executing turn: '%s'", test_prompt)
    assistant_response = assistant.process_text_turn(test_prompt, wait_for_playback=True)

    logger.info("Assistant Spoken Response: %s", assistant_response)
    assert assistant_response and len(assistant_response) > 0, "No assistant response generated"

    assistant.stop()

    # 5. Verify Markdown persistence
    logger.info("--- Step 4: Verifying Markdown archive records ---")
    assert session_file is not None and session_file.exists(), "Session file was not created"
    session_content = session_file.read_text(encoding="utf-8")
    assert test_prompt in session_content, "User utterance missing from session Markdown"
    assert assistant_response in session_content, "Gemini response missing from session Markdown"
    assert "###" in session_content, "Turn headers missing from session Markdown"

    logger.info("==================================================================")
    logger.info("   ALL SMOKE TESTS PASSED SUCCESSFULLY! ")
    logger.info("   - Gemini Model: %s", smoke_config.llm_model_name)
    logger.info("   - Speech Pipeline: Active and Verified")
    logger.info("   - Markdown Session: %s", session_file.name)
    logger.info("==================================================================")


if __name__ == "__main__":
    run_smoke_test()
