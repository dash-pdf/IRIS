# IRIS — Intelligent Reasoning & Interaction System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 172 passed](https://img.shields.io/badge/tests-172%20passed-brightgreen.svg)](tests/)

IRIS is a modular, half-duplex conversational voice assistant engineered for macOS (Apple Silicon & Intel) and designed for cross-platform portability. It combines articulate British female voice synthesis, local Voice Activity Detection (VAD), acoustic self-echo filtering, multi-turn conversation memory, and durable Markdown session archiving.

---

## Key Features

- **Half-Duplex Audio State Machine:** Strict transitions (`IDLE` &rarr; `LISTENING` &rarr; `THINKING` &rarr; `SPEAKING` &rarr; `COOLDOWN` &rarr; `IDLE`) with a software mute gate that completely eliminates speaker-to-microphone feedback loops.
- **Acoustic Self-Echo Cancellation:** Multi-stage `EchoFilter` with fuzzy string matching, content-word Jaccard similarity, and immediate override detection for spoken interruption commands (*"stop"*, *"cancel"*, *"wait"*, *"abort"*).
- **Speech Text Shaping:** Automatic filtering via `shape_speech_text` that strips Markdown asterisks, headings, list bullets, code fences, and raw URLs before text-to-speech synthesis while preserving unblemished dialogue in conversation archives.
- **In-Memory Turn History & Context Restoration:** Sliding window turn management (`ConversationService`) that automatically restores context from previous Markdown session archives upon startup.
- **Multi-Provider Intelligence:** Pluggable LLM adapters supporting local offline inference (Ollama), strictly free cloud routing (OpenRouter), and enterprise hosted APIs (Google Gemini, Anthropic Claude, OpenAI).
- **Pluggable Voice Synthesis (TTS):** Microsoft Edge Neural TTS (default British female voice `en-GB-SoniaNeural`, free without API keys), local offline Kokoro TTS (ONNX Runtime: `bf_emma`, `bf_alice`), native macOS speech synthesis (`say`), and ElevenLabs TTS.
- **Pluggable Speech Recognition (STT):** ElevenLabs Scribe STT, interactive terminal console mode (`--text`), and mock offline testing adapters.

---

## Architecture & Directory Structure

```text
IRIS/
├── iris/
│   ├── audio/                     # Audio capture, VAD, playback & echo prevention
│   │   ├── capture.py             # SoundDevice microphone stream with mute gate
│   │   ├── vad.py                 # Energy-based Voice Activity Detector
│   │   ├── player.py              # Non-blocking audio queue player with instant abort
│   │   └── echo_filter.py         # Transcript self-echo filter & interruption patterns
│   ├── core/                      # Core assistant coordinator & state
│   │   ├── assistant.py           # IRISAssistant main runtime orchestrator
│   │   ├── state.py               # Half-duplex AudioStateMachine
│   │   └── config.py              # Environment configuration loader (.env)
│   ├── adapters/                  # Pluggable hardware & vendor integrations
│   │   ├── model/                 # LLM Intelligence adapters
│   │   │   ├── ollama_adapter.py  # Local Ollama REST adapter (/api/chat, keep-alive, telemetry)
│   │   │   ├── openrouter_adapter.py # OpenRouter free router with strict allowlist
│   │   │   ├── gemini_adapter.py  # Google Gemini 3.8 Flash (google-genai SDK)
│   │   │   ├── anthropic_adapter.py # Anthropic Claude adapter
│   │   │   ├── openai_adapter.py  # OpenAI GPT-4o adapter
│   │   │   └── mock_adapter.py    # Offline mock adapter for unit tests
│   │   ├── tts/                   # Text-to-Speech adapters
│   │   │   ├── base.py            # Base adapter interface & shape_speech_text cleaner
│   │   │   ├── edge_tts.py        # Microsoft Edge TTS (cloud neural, free, low-latency)
│   │   │   ├── kokoro_tts.py      # Kokoro-82M ONNX local neural speech synthesizer
│   │   │   ├── macos_tts.py       # Native macOS 'say' command fallback
│   │   │   ├── elevenlabs_tts.py  # ElevenLabs cloud TTS
│   │   │   └── mock_tts.py        # Offline mock audio synthesizer
│   │   └── stt/                   # Speech-to-Text adapters
│   │       ├── elevenlabs_stt.py  # ElevenLabs Scribe STT
│   │       ├── console_stt.py     # Keyboard text input adapter
│   │       └── mock_stt.py        # Mock transcription feeder
│   └── services/                  # Persistent runtime services
│       ├── archive.py             # Date-partitioned Markdown session logger & loader
│       └── conversation.py        # In-session turn history & system prompts
├── scripts/                       # Diagnostics & integration smoke tests
│   ├── diagnose_audio.py          # Audio hardware & STT diagnostic suite
│   ├── smoke_test_openrouter.py   # Live OpenRouter free router smoke test
│   ├── smoke_test_gemini_phase1.py# Live Google Gemini smoke test
│   └── test_kokoro_tts.py         # Standalone Kokoro ONNX benchmark script
├── tests/                         # Comprehensive offline pytest test suite (172 tests)
├── iris_voice.py                  # Main CLI entry point
├── requirements.txt               # Production Python dependencies
├── .env.example                   # Environment variable template
└── .gitignore                     # Git ignore rules protecting private data
```

---

## Installation & Setup

### 1. Prerequisites
- **Operating System:** macOS Sonoma / Sequoia (Apple Silicon or Intel) or Linux / Windows (with WASAPI).
- **Python:** Python 3.12 or newer.
- **Audio Libraries:** [PortAudio](https://www.portaudio.com/) (installed automatically by `sounddevice` wheels, or via `brew install portaudio` on macOS).
- **Optional Local LLM:** [Ollama](https://ollama.com/) installed and running locally for completely offline execution.

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/IRIS.git
cd IRIS

# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy the example environment file to `.env`:

```bash
cp .env.example .env
```

Open `.env` and configure your preferred providers. For example:

```ini
# Choose your LLM provider: ollama | openrouter | google | anthropic | openai | mock
IRIS_LLM_PROVIDER=ollama
IRIS_LLM_MODEL=llama3.2:1b

# Local Ollama settings (used when IRIS_LLM_PROVIDER=ollama)
IRIS_OLLAMA_HOST=http://localhost:11434
IRIS_OLLAMA_KEEP_ALIVE=30m

# Microsoft Edge TTS is the free cloud neural default (no API key required)
IRIS_TTS_PROVIDER=edge
IRIS_EDGE_VOICE=en-GB-SoniaNeural

# If using hosted LLM providers, fill in only the key for the provider you use:
OPENROUTER_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# ElevenLabs credentials (only required if using ElevenLabs STT or TTS)
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=ThT5KcBeYPX3keUQqHPh
```

---

## Running IRIS

Start the assistant via the main CLI entrypoint:

```bash
python iris_voice.py
```

### Command-Line Options

| Argument | Choices / Type | Description |
|---|---|---|
| `--llm` | `ollama`, `openrouter`, `google`, `gemini`, `anthropic`, `openai`, `mock` | Override the active LLM provider |
| `--model` | `string` | Override LLM model name (e.g. `llama3.2:1b`, `openrouter/free`) |
| `--ollama-host` | `string` | Override Ollama server URL (default: `http://localhost:11434`) |
| `--ollama-keep-alive`| `string` | Override model memory retention duration (e.g. `30m`, `1h`) |
| `--tts` | `edge`, `kokoro`, `macos`, `elevenlabs`, `mock` | Override TTS synthesis engine |
| `--stt` | `elevenlabs`, `console`, `mock` | Override speech transcription engine |
| `--voice` | `string` | Override voice preset (e.g. `en-GB-SoniaNeural`, `bf_emma`, `Flo`) |
| `--device` | `int` | Specify input microphone device index |
| `--list-devices` | flag | List all detected audio input/output devices and exit |
| `--text` | flag | Terminal text chat mode (bypasses microphone capture) |

### Common Usage Examples

**1. Completely Free Local LLM + Free Neural Cloud TTS (Default):**
```bash
# Ensure Ollama is running with llama3.2:1b
ollama run llama3.2:1b
python iris_voice.py --llm ollama --tts edge
```

**2. Free Hosted Cloud LLM via OpenRouter:**
```bash
# Requires OPENROUTER_API_KEY in .env
python iris_voice.py --llm openrouter --tts edge
```

**3. Keyboard Text Mode (No Microphone Required):**
```bash
python iris_voice.py --text --llm ollama
```

**4. Google Gemini 3.8 Flash:**
```bash
python iris_voice.py --llm google --model gemini-3.8-flash --tts edge
```

---

## Supported Components

### Intelligence (LLM) Adapters

| Provider | Model ID | Connectivity | Notes |
|---|---|---|---|
| **Ollama** | `llama3.2:1b` (default) | Local REST | Fully offline. Uses compact `IRIS_LOCAL_SYSTEM_PROMPT`, configurable `keep_alive`, and parses nanosecond telemetry (tokens/sec, prompt eval). |
| **OpenRouter** | `openrouter/free` | Cloud REST | Strictly enforced free-tier allowlist. Hard no-fallback policy rejecting all paid models. |
| **Google** | `gemini-3.8-flash` | Cloud SDK | Official `google-genai` SDK with full multi-turn context preservation. |
| **Anthropic** | `claude-3-5-haiku` | Cloud SDK | Anthropic Messages API client. |
| **OpenAI** | `gpt-4o-mini` | Cloud REST | Standard OpenAI chat completions. |
| **Mock** | N/A | Offline memory | Deterministic canned responses for unit tests and local development. |

### Speech Synthesis (TTS) Adapters

| Provider | Default Voice | Requirements | Characteristics |
|---|---|---|---|
| **Edge TTS** | `en-GB-SoniaNeural` | Network connection | High-speed neural speech synthesis. Free, zero credentials required. |
| **Kokoro TTS** | `bf_emma` | `models/kokoro/` weights | Fully offline 82M parameter ONNX neural voice model. |
| **macOS Say** | `Flo (English (UK))` | macOS system | Offline fallback utilizing macOS native speech synthesis. |
| **ElevenLabs** | Custom Voice ID | API key | High-fidelity cloud voice synthesis. |

### Speech Recognition (STT) Adapters

| Provider | Requirements | Characteristics |
|---|---|---|
| **ElevenLabs Scribe** | API key & microphone | Cloud streaming speech-to-text with automatic punctuation. |
| **Console STT** | Terminal keyboard | Interactive console input for silent or headphoneless operation. |
| **Mock STT** | None | Queued string provider for automated pipeline testing. |

---

## Testing & Diagnostics

The project features a comprehensive test suite of 172 offline unit and integration tests.

### Running Automated Tests
```bash
# Run the complete test suite (no network or API keys required)
pytest tests/ -v
```

### Running Hardware & Integration Diagnostics
```bash
# Test audio input levels, RMS energy, and microphone permissions:
python scripts/diagnose_audio.py

# Verify live OpenRouter free router connectivity:
python scripts/smoke_test_openrouter.py

# Verify live Gemini connectivity:
python scripts/smoke_test_gemini_phase1.py
```

---

## Known Limitations

- **Non-Streaming Generation:** In the current phase, LLM text responses are fetched in full before being passed to TTS. Real-time token streaming pipelining is planned for a future milestone.
- **Hardware Memory for Local Models:** When using Ollama with `IRIS_OLLAMA_KEEP_ALIVE`, model weights remain in unified memory (~1.5–2.5 GB for 1B–3B parameter models). Ensure adequate free RAM on 8 GB devices.
- **Upstream Rate Limits:** The `openrouter/free` router endpoint relies on shared public provider capacity and may return HTTP 429 during peak traffic hours. IRIS halts cleanly without silently switching to paid models.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
