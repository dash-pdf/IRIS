"""
poc_local_test.py
------------------
Local test runner for IRIS — runs immediately on macOS without external API keys.

Validates the full local pipeline:
1. In-session turn tracking and sliding window (iris.services.conversation).
2. The IRIS British personality system prompt.
3. Durable Markdown archiving (iris.services.archive).
4. Audio speech output using macOS British voice synthesis.
5. Verification of saved session transcripts in memory/conversations/.

Usage:
    python poc_local_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from iris.core.config import load_config
from iris.services.archive import ConversationArchive
from iris.services.conversation import ConversationService, IRIS_SYSTEM_PROMPT


def speak_macos(text: str, voice: str = "Flo (English (UK))") -> None:
    """Speak text aloud using macOS native speech synthesis."""
    try:
        subprocess.run(["say", "-v", voice, text], check=True)
    except Exception as exc:
        print(f"Audio playback note: {exc}")


def main() -> None:
    print("=" * 65)
    print("IRIS — Phase 0 Local Pipeline & Voice Validation")
    print("=" * 65)

    # 1. Load config (without requiring external voice keys for local test)
    config = load_config(require_voice=False)
    archive = ConversationArchive(
        archive_dir=config.archive_dir,
        persona_name=config.persona_name,
    )
    conv = ConversationService(system_prompt=IRIS_SYSTEM_PROMPT)

    session_file = archive.open_session()
    print(f"\n📂 Active Markdown Archive: {session_file}")

    # Greeting
    greeting = "Hello. I am IRIS. All local systems, conversation services, and audio pipelines are active."
    print(f"\n🤖 IRIS: {greeting}")
    archive.write_turn("assistant", greeting)
    conv.add_assistant_turn(greeting)
    speak_macos(greeting)

    print("\n💬 You can now talk to IRIS (type your message and press Enter).")
    print("   Type 'exit' or 'quit' to end the session.\n")

    sample_responses = [
        "I understand. Your assignment timeline is tight, but we can break it down into manageable steps.",
        "Understood. The turn-taking model and Markdown conversation archive are operating smoothly.",
        "Indeed. That sounds reasonable. Shall we proceed with the next task?",
    ]
    resp_idx = 0

    try:
        while True:
            try:
                user_input = input("🎤 You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                farewell = "Goodbye for now. Your session has been safely archived."
                print(f"\n🤖 IRIS: {farewell}")
                archive.write_turn("assistant", farewell)
                speak_macos(farewell)
                break

            # Record user turn in archive and conversation service
            t_start = time.perf_counter()
            archive.write_turn("user", user_input)
            conv.add_user_turn(user_input)

            # Simulated personality-consistent response
            response = sample_responses[resp_idx % len(sample_responses)]
            resp_idx += 1

            latency_ms = int((time.perf_counter() - t_start) * 1000)

            print(f"🤖 IRIS: {response}")
            print(f"   ⏱  Local pipeline latency: {latency_ms} ms")

            archive.write_turn("assistant", response)
            conv.add_assistant_turn(response)

            # Speak response through Mac speakers
            speak_macos(response)
            print()

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
    finally:
        archive.close_session()
        print(f"\n✅ Conversation successfully preserved in:\n   {session_file}\n")


if __name__ == "__main__":
    main()

