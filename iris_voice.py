"""
iris_voice.py
-------------
Main entry point for the decoupled IRIS Voice Assistant.

Features:
- Completely decoupled architecture (independent LLM, STT, TTS, Audio).
- Strict half-duplex state machine:
  IDLE -> LISTENING -> THINKING -> SPEAKING -> COOLDOWN -> IDLE
- Multi-layer acoustic feedback prevention (software mute gate + cooldown + echo filter).
- Markdown session journaling in memory/conversations/.
- Flexible CLI flags for choosing providers, audio devices, or falling back to text mode.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig, load_config
from iris.core.state import AudioState

# Configure terminal logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Reduce third-party library verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("iris.runner")


def print_banner(config: IRISConfig) -> None:
    if config.tts_provider == "kokoro":
        voice_desc = f"KOKORO ({config.kokoro_voice})"
    elif config.tts_provider == "edge":
        voice_desc = f"EDGE ({config.edge_voice})"
    elif config.tts_provider == "macos":
        voice_desc = f"MACOS ({config.macos_voice})"
    else:
        voice_desc = f"{config.tts_provider.upper()} ({config.elevenlabs_voice_id})"

    banner = f"""
========================================================================
   IRIS — Intelligent Reasoning & Interaction System
   Decoupled Half-Duplex Conversational Voice Assistant
========================================================================
   Intelligence (LLM) : {config.llm_provider.upper()} ({config.llm_model_name})
   Voice Output (TTS) : {voice_desc}
   Speech Input (STT) : {config.stt_provider.upper()}
   Audio Input Device : {"#" + str(config.audio_input_device) if config.audio_input_device is not None else "Default Microphone"}
   Acoustic Defense   : Half-Duplex Gate | {config.cooldown_ms}ms Cooldown | Echo Filter
   Session Markdown   : {config.archive_dir}
========================================================================
   Status: Ready. Speak into your microphone or type commands.
   Press Ctrl+C to terminate session.
========================================================================
"""
    print(banner)


def list_audio_devices() -> None:
    import sounddevice as sd

    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]

    print("\n--- Available Audio Input Devices ---")
    for idx, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            is_def = " [DEFAULT INPUT]" if idx == default_in else ""
            print(f"  [{idx}] {d['name']} (Channels: {d['max_input_channels']}, Default SR: {int(d['default_samplerate'])}Hz){is_def}")

    print("\n--- Available Audio Output Devices ---")
    for idx, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            is_def = " [DEFAULT OUTPUT]" if idx == default_out else ""
            print(f"  [{idx}] {d['name']} (Channels: {d['max_output_channels']}, Default SR: {int(d['default_samplerate'])}Hz){is_def}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="IRIS Decoupled Voice Assistant")
    parser.add_argument("--llm", choices=["ollama", "openrouter", "anthropic", "google", "gemini", "openai", "mock"], help="LLM provider")
    parser.add_argument("--model", help="Override LLM model name (e.g. llama3.2:1b, openrouter/free)")
    parser.add_argument("--ollama-host", help="Override Ollama host URL (default: http://localhost:11434)")
    parser.add_argument("--ollama-keep-alive", help="Override Ollama keep_alive duration (default: 30m)")
    parser.add_argument("--tts", choices=["kokoro", "edge", "macos", "elevenlabs", "mock"], help="TTS provider")
    parser.add_argument("--stt", choices=["elevenlabs", "console", "mock"], help="STT provider")
    parser.add_argument("--voice", help="Override voice name / ID (e.g. bf_emma, en-GB-SoniaNeural, Flo)")
    parser.add_argument("--device", type=int, help="Input audio device ID index")
    parser.add_argument("--list-devices", action="store_true", help="List audio input and output devices and exit")
    parser.add_argument("--text", action="store_true", help="Interactive terminal text mode (bypasses mic capture)")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Load base config from .env
    base_config = load_config(require_voice=False)

    # Determine providers with CLI flag overrides
    llm_provider = args.llm or base_config.llm_provider
    if llm_provider == "anthropic" and (not base_config.anthropic_api_key or "your_" in base_config.anthropic_api_key):
        logger.info("No valid ANTHROPIC_API_KEY found in environment; using built-in IRIS mock intelligence.")
        llm_provider = "mock"

    tts_provider = args.tts or base_config.tts_provider
    stt_provider = "console" if args.text else (args.stt or base_config.stt_provider)

    macos_voice = args.voice or base_config.macos_voice
    kokoro_voice = args.voice or base_config.kokoro_voice
    edge_voice = args.voice or base_config.edge_voice
    elevenlabs_voice = args.voice or base_config.elevenlabs_voice_id
    input_device = args.device if args.device is not None else base_config.audio_input_device
    llm_model = args.model or base_config.llm_model_name
    ollama_host = args.ollama_host or base_config.ollama_host
    ollama_keep_alive = args.ollama_keep_alive or base_config.ollama_keep_alive

    config = IRISConfig(
        elevenlabs_api_key=base_config.elevenlabs_api_key,
        elevenlabs_agent_id=base_config.elevenlabs_agent_id,
        elevenlabs_voice_id=elevenlabs_voice,
        llm_provider=llm_provider,
        ollama_host=ollama_host,
        ollama_keep_alive=ollama_keep_alive,
        anthropic_api_key=base_config.anthropic_api_key,
        google_api_key=base_config.google_api_key,
        openai_api_key=base_config.openai_api_key,
        openrouter_api_key=base_config.openrouter_api_key,
        llm_model_name=llm_model,
        tts_provider=tts_provider,
        macos_voice=macos_voice,
        kokoro_voice=kokoro_voice,
        edge_voice=edge_voice,
        tts_speaking_rate=base_config.tts_speaking_rate,
        stt_provider=stt_provider,
        sample_rate=base_config.sample_rate,
        audio_input_device=input_device,
        vad_energy_threshold=base_config.vad_energy_threshold,
        silence_timeout_ms=base_config.silence_timeout_ms,
        cooldown_ms=base_config.cooldown_ms,
        echo_similarity_threshold=base_config.echo_similarity_threshold,
        archive_dir=base_config.archive_dir,
        persona_name=base_config.persona_name,
    )

    print_banner(config)

    def on_state_change(old_state: AudioState, new_state: AudioState, turn_id: str) -> None:
        state_colors = {
            AudioState.IDLE: "\033[90m[IDLE - Listening]\033[0m",
            AudioState.LISTENING: "\033[1;94m[LISTENING... Speech detected, listening to you]\033[0m",
            AudioState.THINKING: "\033[1;93m[THINKING... Transcribing and reasoning]\033[0m",
            AudioState.SPEAKING: "\033[1;92m[SPEAKING... Spoken reply playing]\033[0m",
            AudioState.COOLDOWN: "\033[95m[COOLDOWN... Audio decaying]\033[0m",
            AudioState.ERROR: "\033[91m[ERROR]\033[0m",
            AudioState.CANCELLED: "\033[91m[CANCELLED]\033[0m",
        }
        tag = state_colors.get(new_state, f"[{new_state.value}]")
        print(f"Status: {tag}")

    def on_user_transcript(text: str) -> None:
        print(f"\n\033[1;36m👤 You:\033[0m {text}")

    def on_assistant_response(text: str) -> None:
        print(f"\033[1;35m🌸 IRIS:\033[0m {text}\n")

    def on_echo_detected(text: str, reason: str) -> None:
        print(f"\033[1;33m⚠️  [ECHO DEFENSE]\033[0m Filtered acoustic loopback: '{text}' ({reason})\n")

    def on_silence_warning() -> None:
        print("\n\033[1;33m" + "=" * 70)
        print("⚠️  [MICROPHONE ATTENTION NEEDED — ZERO AUDIO SIGNAL]")
        print("IRIS is capturing audio frames, but macOS is returning pure silence (0s).")
        print("  1. Grant Microphone access in:")
        print("     macOS System Settings -> Privacy & Security -> Microphone")
        print("     Ensure your Terminal / IDE (Terminal, iTerm, VS Code, Cursor) is ON.")
        print("  2. Check microphone input volume in:")
        print("     macOS System Settings -> Sound -> Input (raise volume to 75%+).")
        print("=" * 70 + "\033[0m\n")

    def on_error(exc: Exception) -> None:
        print(f"\n\033[1;31m❌ [ERROR]\033[0m {exc}\n")

    assistant = IRISAssistant(
        config=config,
        on_state_change=on_state_change,
        on_user_transcript=on_user_transcript,
        on_assistant_response=on_assistant_response,
        on_echo_detected=on_echo_detected,
        on_silence_warning=on_silence_warning,
        on_error=on_error,
    )

    try:
        assistant.start()

        if args.text or stt_provider == "console":
            print("Type your message and press Enter (or 'quit' / 'exit' to stop):")
            while assistant.is_running:
                try:
                    user_input = input("\nYou: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in {"quit", "exit", "q"}:
                        break
                    assistant.process_text_turn(user_input)
                except (KeyboardInterrupt, EOFError):
                    break
        else:
            print("🎙  Microphone listening loop active. Speak now. (Press Ctrl+C to end session)\n")
            while assistant.is_running:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nInterrupt received. Stopping IRIS cleanly...")
    finally:
        assistant.stop()
        print("IRIS session closed. Conversation saved to Markdown archive.\n")


if __name__ == "__main__":
    main()
