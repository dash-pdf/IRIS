# IRIS — Intelligent Research & Information System

**IRIS** is a local-first personal intelligence assistant for macOS built around modular voice interaction, deterministic tool routing, environment awareness, and durable plain-text knowledge management.

---

## Current Implemented Architecture

### 1. Voice & Runtime Pipeline (`iris/core/`, `iris/audio/`, `iris/adapters/`)
- **Half-Duplex Orchestrator (`iris/core/assistant.py`, `iris/core/state.py`)**: Manages explicit state transitions (`IDLE -> LISTENING -> THINKING -> SPEAKING -> COOLDOWN -> IDLE`), thread-safe cancellation, and turn timing metrics.
- **Voice Presence & Double-Clap Activation (`iris/audio/presence.py`)**:
  - Dedicated state machine (`DISABLED`, `DORMANT`, `ARMED`, `PROCESSING`, `SPEAKING`, `ERROR`) coordinated with `AudioStateMachine`.
  - **Local `DoubleClapDetector`**: Sub-frame high-pass energy ratio, impulse duration limits, inter-clap timing bounds (`140ms–850ms`), noise-floor adaptation, and `1.5s` cooldown hysteresis so room reverb, desk thuds, typing, and continuous speech/music do not falsely trigger.
  - **Standby Privacy Guarantee**: While `DORMANT`, microphone audio goes strictly to the local transient clap detector — zero bytes are forwarded to VAD, ElevenLabs STT, or any LLM, and zero raw PCM buffers are retained across frames. Calling **Disable Mic** (`DISABLED`) halts microphone capture completely.
  - **Hands-Free Multi-Turn & Deactivation**: Double-clapping (or optional `"hey iris"` activation phrase) moves IRIS from `DORMANT` to `ARMED`, allowing consecutive voice commands without repeating a wake phrase until another double-clap (during quiet pauses), deterministic deactivation phrase (`"Stop listening."`, `"Go to standby."`, `"Deactivate listening."`, `"That's all."`), manual standby/disable button, shortcut (`Ctrl+Shift+I`), or configurable inactivity timeout returns IRIS to `DORMANT`.
  - **TTS & Media Suppression**: Clap detection is suppressed while `SPEAKING`, during post-TTS acoustic decay guard, and via external audio playback gating hooks.
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

### 5. PySide6 Desktop Application Shell (`iris/desktop/`, `iris_desktop.py`)
- **Three-Panel Dark Workspace (`iris/desktop/main_window.py`, `theme.py`)**: Uses `#09090D` / `#14131B` backgrounds with `#8057D9` / `#B69AFF` accents:
  - **Left Sidebar (`widgets/sidebar.py`)**: Navigation for `Home`, `Knowledge`, `Research` *(Planned)*, `Calendar` *(Planned)*, `Tasks` *(Planned)*, `Music` *(Planned)*, and `Settings`, alongside the live Markdown Vault folder tree (`Inbox`, `Notes`, `Projects`, `Research`, `Archive`).
  - **Centre Workspace (`widgets/home_page.py`, `knowledge_page.py`, `research_page.py`, `placeholders.py`)**: Live Home dashboard (weather/location context, Inbox count, recent notes, quick deterministic search & capture), tabbed Knowledge Vault Markdown editor/previewer & Inbox triage connected to `KnowledgeService`, modular Research & Video Intelligence Phase 4 container, clearly marked placeholders, and redacted non-secret Settings view.
  - **Right Assistant Panel (`widgets/assistant_panel.py`)**: Active session history, message composer, processing indicator, and secret-redacted error banner wired through `IRISDesktopBackend` (`QThreadPool` + Qt Signals) to `IRISAssistant`.
  - **Persistent Bottom Voice Control Bar (`widgets/voice_bar.py`)**: Explicit presence status (`Disabled`, `Dormant (Clap Standby)`, `Armed (Listening)`, `Processing`, `Speaking`, `Error`), honest listening indicator (never active when `DISABLED` or `DORMANT`), manual `Standby` / `Arm` / `Disable Mic` actions, `Ctrl+Shift+I` toggle shortcut, and quick text command input, paired with presence configuration (clap toggle, sensitivity `0.1–1.0`, active timeout, optional activation phrase) in `Settings`.

### 6. Application-Wide LLM Usage Policy & Research Compatibility (`iris/core/policy.py`, `iris/services/research/`)
- **Deterministic First (`LLMUsagePolicy`)**: Ordinary operations (Markdown note CRUD, folder moves, Inbox persistence, SQLite FTS5 search, math, date/time, context retrieval) run offline in Python/SQLite with zero LLM calls; AI reasoning flows exclusively through `IRISAssistant` and `ModelAdapter`.
- **Research & Video Intelligence Contracts (`iris/services/research/`)**: Modular Phase 4 interfaces (`ResearchWorkspaceBridge`, `ResearchSource`, `VideoResearchSummary`, `VideoTimestamp`) flagging external content untrusted, preserving URLs/timestamps, refusing to fabricate inaccessible video analyses, and requiring explicit user confirmation before writing to `vault/Research/`.

### 7. Security & Credential Handling (`iris/core/security.py`, `iris/core/credentials.py`)
- Centralised secret resolution (environment variables and macOS Keychain via `keyring`) and error/log redaction so API keys and tokens never appear in terminal output, conversation transcripts, Knowledge Vault files, or desktop Settings views.
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
   - Voice Presence controls (`IRIS_PRESENCE_ENABLED`, `IRIS_CLAP_ACTIVATION_ENABLED`, `IRIS_CLAP_SENSITIVITY`, `IRIS_ACTIVE_TIMEOUT_S`, `IRIS_ACTIVATION_PHRASE`, `IRIS_PRESENCE_SHORTCUT`)
   - Provider keys (`ELEVENLABS_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, etc.) and optional manual location settings (`IRIS_LOCATION_LAT`, `IRIS_LOCATION_LON`, `IRIS_LOCATION_LABEL`).

### Launching IRIS
- PySide6 Desktop Application (with Voice Presence & Knowledge Vault):
  ```bash
  ./run.sh --desktop
  ```
- Full CLI voice assistant launcher (add `--dormant` for local double-clap standby startup):
  ```bash
  ./run.sh
  .venv/bin/python iris_voice.py --dormant
  ```
- Text mode or provider overrides via `iris_voice.py`:
  ```bash
  .venv/bin/python iris_voice.py --text
  .venv/bin/python iris_voice.py --text --llm cloud_router
  ```

### Running Tests
Run the full automated test suite (**283 tests** across 19 test modules):
```bash
.venv/bin/pytest tests/
```

---

## Roadmap / Upcoming Phases (Not Yet Implemented)
The following planned features remain outside the current Phase 3 implementation:
- **Phase 4 — Research Engine & Mandatory Video Intelligence**: Embedded multi-tab research browser, webpage extraction, and full Video Intelligence (lecture/video discovery, caption/transcript retrieval, audio transcription when permitted, timestamped concept analysis, study notes, comparison, and revision flashcards)
- Google Calendar, Tasks, and Spotify integrations
- Proactive background reminders and notifications
