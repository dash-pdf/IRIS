# IRIS — Intelligent Reasoning & Interaction System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 198 passed](https://img.shields.io/badge/tests-198%20passed-brightgreen.svg)](tests/)

IRIS is a modular, half-duplex conversational voice assistant engineered for macOS (Apple Silicon & Intel) and designed for cross-platform portability. It combines articulate British female voice synthesis, local Voice Activity Detection (VAD), acoustic self-echo filtering, multi-turn conversation memory, deterministic model routing across free cloud intelligence providers, and durable Markdown session archiving.

---

## Key Features

- **Deterministic Cloud Intelligence Routing:** Automatic, lightweight Python classification routes queries to the optimal free-tier provider or deterministic tool without extra LLM overhead:
  - **Everyday Conversation:** Handled by OpenRouter free-tier models (`openrouter/free`).
  - **Complex Reasoning & Analysis:** Handled by Google Gemini Flash via the Gemini Developer API free tier (`gemini-3.8-flash`).
  - **Exact Calculations:** Deterministic AST-based math evaluator for arithmetic, percentages, square roots, and algebraic expansions.
  - **Date & Time Operations:** Instantaneous system utilities for local time, date, day of the week, and year.
  - **Session & Memory Retrieval:** Local conversation memory lookup from date-partitioned Markdown session archives.
- **Strict Free-Tier Enforcement:** Hard safeguards prevent accidental paid-model selection. Strict allowlists block paid endpoints, rate limits halt without paid fallback, and Gemini quota exhaustion prompts the user before degrading to a standard free model.
- **Half-Duplex Audio State Machine:** Strict transitions (`IDLE` &rarr; `LISTENING` &rarr; `THINKING` &rarr; `SPEAKING` &rarr; `COOLDOWN` &rarr; `IDLE`) with a software mute gate that completely eliminates speaker-to-microphone feedback loops.
- **Acoustic Self-Echo Cancellation:** Multi-stage `EchoFilter` with fuzzy string matching, content-word Jaccard similarity, and immediate override detection for spoken interruption commands (*"stop"*, *"cancel"*, *"wait"*, *"abort"*).
- **Speech Text Shaping:** Automatic filtering via `shape_speech_text` that strips Markdown asterisks, headings, list bullets, code fences, and raw URLs before text-to-speech synthesis while preserving unblemished dialogue in conversation archives.
- **In-Memory Turn History & Context Restoration:** Sliding window turn management (`ConversationService`) that automatically restores context from previous Markdown session archives upon startup.
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
│   ├── core/                      # Core assistant coordinator, router & state
│   │   ├── assistant.py           # IRISAssistant main runtime orchestrator
│   │   ├── router.py              # Deterministic classifier & safe math/datetime tools
│   │   ├── state.py               # Half-duplex AudioStateMachine
│   │   └── config.py              # Environment configuration loader (.env)
│   ├── adapters/                  # Pluggable hardware & vendor integrations
│   │   ├── model/                 # LLM Intelligence adapters
│   │   │   ├── cloud_router.py    # Cloud Intelligence Router (deterministic routing)
│   │   │   ├── openrouter_adapter.py # OpenRouter free router with strict allowlist
│   │   │   ├── gemini_adapter.py  # Google Gemini Flash (google-genai SDK, free allowlist)
│   │   │   ├── anthropic_adapter.py # Anthropic Claude adapter
│   │   │   ├── openai_adapter.py  # OpenAI GPT-4o adapter
│   │   │   ├── ollama_adapter.py  # Local Ollama REST adapter (/api/chat, legacy)
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
├── tests/                         # Comprehensive offline pytest test suite (198 tests)
│   ├── test_cloud_router.py       # Deterministic tools & routing verification
│   ├── test_model_adapters.py     # Model adapter unit & factory tests
│   ├── test_assistant.py          # Assistant turn & lifecycle orchestration
│   └── ...                        # Audio, state machine, VAD, STT, and TTS tests
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

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/dash-pdf/IRIS.git
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

Open `.env` and configure your API credentials:

```ini
# Cloud Intelligence Router (Default)
IRIS_LLM_PROVIDER=router

# OpenRouter API Key (for everyday conversation via openrouter/free)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Google Gemini API Key (for complex reasoning via free-tier Gemini Flash)
# Get a free key from Google AI Studio Developer API: https://aistudio.google.com/
GOOGLE_API_KEY=your_google_api_key_here

# Microsoft Edge TTS is the free cloud neural default (no API key required)
IRIS_TTS_PROVIDER=edge
IRIS_EDGE_VOICE=en-GB-SoniaNeural

# ElevenLabs credentials (only required if using ElevenLabs STT)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

---

## Running IRIS

Start the assistant with a single command:

```bash
./run.sh
```

*(Alternatively, `./iris.sh` or `python iris_voice.py`)*

For interactive keyboard chat mode (no microphone required):

```bash
./run.sh --text
```

### Command-Line Options

| Argument | Choices / Type | Description |
|---|---|---|
| `--llm` | `router`, `cloud`, `openrouter`, `google`, `gemini`, `anthropic`, `openai`, `mock`, `ollama` | Override the active intelligence provider (default: `router`) |
| `--model` | `string` | Override LLM model name (strict free allowlists enforced) |
| `--tts` | `edge`, `kokoro`, `macos`, `elevenlabs`, `mock` | Override TTS synthesis engine |
| `--stt` | `elevenlabs`, `console`, `mock` | Override speech transcription engine |
| `--voice` | `string` | Override voice preset (e.g. `en-GB-SoniaNeural`, `bf_emma`, `Flo`) |
| `--device` | `int` | Specify input microphone device index |
| `--list-devices` | flag | List all detected audio input/output devices and exit |
| `--text` | flag | Terminal text chat mode (bypasses microphone capture) |

### Common Usage Examples

**1. Default Cloud Intelligence Router + Free Neural Cloud TTS:**
```bash
python iris_voice.py
```

**2. Keyboard Text Mode (No Microphone Required):**
```bash
python iris_voice.py --text
```

**3. Direct OpenRouter Free Tier:**
```bash
python iris_voice.py --llm openrouter --tts edge
```

**4. Direct Google Gemini Flash Free Tier:**
```bash
python iris_voice.py --llm google --model gemini-3.8-flash --tts edge
```

---

## Intelligence Architecture: Deterministic Routing

IRIS employs deterministic request classification before making any external calls:

| Request Type | Execution Route | Notes |
|---|---|---|
| **Everyday Conversation** | OpenRouter (`openrouter/free`) | Strict free-only allowlist; halts on 429 without paid fallback. |
| **Complex Reasoning** | Gemini Flash (`gemini-3.8-flash`) | Free tier on Gemini Developer API; asks user before degrading if unavailable. |
| **Exact Calculations** | Deterministic Python Functions | Safe AST arithmetic, percentages, roots, powers, algebraic expansions. |
| **Date & Time Operations** | Deterministic System Utilities | Instant local time, date, day of week, and year formatting. |
| **Session Memory Retrieval** | Local Markdown Archives | Queries `memory/conversations/` session files without hallucinations. |
| **Future Tasks** | Deterministic Tools | Extensible interface for external tools and actions. |

---

## Testing & Diagnostics

The project features a comprehensive test suite of 198 offline unit and integration tests.

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
- **Upstream Rate Limits:** Free-tier cloud providers (`openrouter/free`, Gemini Developer API free tier) enforce rate limits (RPM/RPD). IRIS halts cleanly or prompts the user rather than spending money.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
