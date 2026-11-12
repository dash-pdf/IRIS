# IRIS — Intelligent Research & Information System

**IRIS** is a local-first personal intelligence assistant for macOS built around modular voice interaction, deterministic tool routing, environment awareness, and durable plain-text knowledge management.

---

## Current Implemented Architecture

### 1. Voice & Runtime Pipeline (`iris/core/`, `iris/audio/`, `iris/adapters/`)
- **Half-Duplex Orchestrator (`iris/core/assistant.py`, `iris/core/state.py`)**: Manages explicit state transitions (`IDLE -> LISTENING -> THINKING -> SPEAKING -> COOLDOWN -> IDLE`), thread-safe cancellation, and turn timing metrics.
- **Audio I/O & Echo Defense (`iris/audio/`)**: `sounddevice` microphone capture (`capture.py`), energy-based Voice Activity Detection (`vad.py`), transcript echo filtering (`echo_filter.py`), and interruptible playback (`player.py`).
- **Speech-to-Text (`iris/adapters/stt/`)**: ElevenLabs Scribe adapter (`elevenlabs_stt.py`), console text input (`console_stt.py`), and mock test adapter (`mock_stt.py`).
- **Text-to-Speech (`iris/adapters/tts/`)**: Edge TTS primary provider (`edge_tts.py`, `edge-tts>=7.3.1,<8.0.0`, default `en-GB-SoniaNeural`) with automatic fallback to native macOS `say` (`macos_tts.py`), plus optional ElevenLabs, Kokoro, and mock adapters.

### 2. Hybrid Deterministic & Cloud Intelligence (`iris/core/router.py`, `iris/adapters/model/`)
- **Deterministic Router (`classify_request`)**: Resolves offline/tool queries before invoking LLMs:
  - Exact arithmetic & unit expressions (`CALCULATION`)
  - Local date/time & timezone queries (`DATETIME`)
  - Live Open-Meteo weather queries (`WEATHER`)
  - Privacy-safe location queries without leaking raw coordinates (`LOCATION`)
  - Explicit Knowledge Vault saves & Inbox captures (`KNOWLEDGE_SAVE`)
  - Full-text & metadata Knowledge Vault searches (`KNOWLEDGE_SEARCH`)
  - Past conversation session recall (`MEMORY`)
- **Cloud Intelligence Router (`cloud_router.py`)**: Routes complex reasoning to Gemini (`gemini_adapter.py`) and everyday dialogue to OpenRouter free models (`openrouter_adapter.py`), with user-confirmed degradation instead of silent paid fallbacks. Also supports Ollama (`ollama_adapter.py`), OpenAI, Anthropic, and Mock adapters.

### 3. Context Engine & Native Location (`iris/services/context/`, `native/location_helper/`)
- **Background `ContextManager`**: Coordinates timezone resolution (`timezone.py`), live weather fetching via Open-Meteo (`weather.py`), and multi-tier location detection (`location.py`) without persisting synthetic fallbacks as true physical locations.
- **Native macOS CoreLocation Helper**: Signed Swift bundle (`bin/IrisLocationHelper.app` built from `native/location_helper/`) requesting CoreLocation When-In-Use permission with explicit manual location configuration supported as fallback.

### 4. Markdown Knowledge Vault & Search (`iris/services/knowledge/`, `vault/`)
IRIS keeps permanent user knowledge strictly separated from raw conversational session logs:

```text
vault/                       # Permanent Knowledge Vault (Markdown source of truth)
├── Inbox/                   # Crash-safe explicit captures awaiting organisation
├── Notes/                   # Evergreen notes & reference material
├── Projects/                # Project notes and plans
├── Research/                # Research briefs and reading notes
└── Archive/                 # Retired notes (status: archived)

memory/
├── conversations/           # Append-only Markdown session transcripts (YYYY/MM/DD/session-NNN.md)
├── context/                 # Cached weather / location snapshots
└── index/
    └── knowledge.db         # Disposable, rebuildable SQLite FTS5 index
```

- **Markdown + YAML Frontmatter (`frontmatter.py`, `models.py`)**: Notes store `note_id`, `title`, `created_at`, `updated_at`, `note_type`, `folder`, `tags`, `status`, and optional `source_url` cleanly above the Markdown body.
- **Atomic Persistence (`storage.py`)**: Writes go to a temporary file (`flush` + `fsync` -> `os.replace`) with collision-resistant filenames and path-traversal checks.
- **Explicit-Intent-Only Creation**: Ordinary dialogue turns and `VaultMemoryAdapter.store_turn()` **never** auto-create permanent notes. Notes are created only when the user explicitly requests saving or capturing (for example, `"Save this as a note: ..."`, `"Add this to my notes"`, or `"Capture this to my inbox: ..."`).
- **Crash-Safe Capture Inbox**: Items captured to `vault/Inbox/` remain intact on disk and in the index if downstream triage or classification raises an error.
- **Disposable SQLite FTS5 Index (`index.py`)**: Supports punctuation-safe full-text queries along with `tags`, `note_type`, and `folder` filters. Deleting `memory/index/knowledge.db` never loses notes; `KnowledgeService.rebuild_index()` reconstructs the index directly from `vault/**/*.md`.

### 5. Security & Credential Handling (`iris/core/security.py`, `iris/core/credentials.py`)
- Centralised secret resolution (environment variables and macOS Keychain via `keyring`) and error/log redaction so API keys and tokens never appear in terminal output, conversation transcripts, or Knowledge Vault files.
- `.env`, runtime caches (`memory/`), SQLite databases (`*.db`), and private Markdown notes (`vault/*/*.md`) are excluded from version control via `.gitignore`.

---

## Setup & Usage

### Prerequisites
- macOS (Apple Silicon / Intel) with Python 3.12+
- Virtual environment in `.venv/`

### Configuration
1. Copy the environment template and fill in your credentials locally:
   ```bash
   cp .env.example .env
   ```
2. Relevant Knowledge Vault & runtime variables in `.env.example`:
   - `IRIS_VAULT_DIR=vault`
   - `IRIS_KNOWLEDGE_INDEX_PATH=memory/index/knowledge.db`
   - `IRIS_ARCHIVE_DIR=memory/conversations`
   - Provider keys (`ELEVENLABS_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, etc.) and optional manual location settings (`IRIS_LOCATION_LAT`, `IRIS_LOCATION_LON`, `IRIS_LOCATION_LABEL`).

### Launching IRIS
- Full voice assistant launcher:
  ```bash
  ./run.sh
  ```
- Text mode or provider overrides via `iris_voice.py`:
  ```bash
  .venv/bin/python iris_voice.py --text
  .venv/bin/python iris_voice.py --text --llm cloud_router
  ```

### Running Tests
Run the full automated test suite (**269 tests** across 17 test modules):
```bash
.venv/bin/pytest tests/
```

---

## Roadmap / Upcoming Phases (Not Yet Implemented)
The following planned features remain outside the current Phase 0 / Phase 1 implementation:
- PySide6 desktop HUD and GUI
- Double-clap hands-free voice presence activation
- Embedded research browser
- Google Calendar & Spotify integrations
- Proactive background reminders and notifications
