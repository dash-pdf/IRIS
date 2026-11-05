"""
iris/core/config.py
-------------------
Loads IRIS configuration from environment variables and .env file.

All configuration lives here. No other module reads os.environ directly.
Secrets are never logged or printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Load .env from the project root (two levels above this file)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # No .env file — rely on environment variables already set.
        pass


_load_env()


def _require(key: str) -> str:
    """Return env var value or raise a clear error if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your values."
        )
    return value


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _optional_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _optional_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _optional_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _optional_float_or_none(key: str) -> float | None:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


@dataclass(frozen=True)
class IRISConfig:
    """
    Immutable configuration object for the IRIS application.

    Instantiate once at startup; pass it into components that need it.
    Never store raw API keys in application state beyond this object.
    """

    # Voice / ElevenLabs credentials
    elevenlabs_api_key: str = ""
    elevenlabs_agent_id: str = ""
    elevenlabs_voice_id: str = "ThT5KcBeYPX3keUQqHPh"  # Default British female voice ID

    # LLM provider selection
    llm_provider: str = "router"  # "router" | "cloud" | "openrouter" | "google" | "anthropic" | "openai" | "ollama" | "mock"
    ollama_host: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"
    openrouter_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    llm_model_name: str = ""

    # TTS configuration
    tts_provider: str = "edge"  # "edge" | "kokoro" | "macos" | "elevenlabs" | "mock"
    macos_voice: str = "Flo (English (UK))"
    kokoro_voice: str = "bf_emma"
    edge_voice: str = "en-GB-SoniaNeural"
    tts_speaking_rate: float = 1.0

    # STT configuration
    stt_provider: str = "elevenlabs"  # "elevenlabs" | "console" | "mock"

    # Audio & Echo cancellation parameters
    sample_rate: int = 16000
    audio_input_device: int | None = None
    vad_energy_threshold: float = 200.0
    silence_timeout_ms: int = 750
    cooldown_ms: int = 400
    echo_similarity_threshold: float = 0.70

    # Conversation archive
    archive_dir: Path = field(default_factory=lambda: Path("memory/conversations"))

    # Context Engine configuration
    enable_location_tracking: bool = True
    fallback_location_name: str = ""
    fallback_latitude: float | None = None
    fallback_longitude: float | None = None
    weather_provider: str = "open-meteo"
    weather_api_key: str = ""
    context_refresh_interval_s: int = 1800
    weather_cache_ttl_s: int = 1800
    context_cache_dir: Path = field(default_factory=lambda: Path("memory/context"))

    # Personality
    persona_name: str = "IRIS"


def load_config(require_voice: bool = True) -> IRISConfig:
    """
    Build and return an IRISConfig from the current environment.

    Args:
        require_voice: If False, ElevenLabs keys are not required.
                       Useful for running offline tests.
    """
    if require_voice:
        elevenlabs_api_key = _require("ELEVENLABS_API_KEY")
        elevenlabs_agent_id = _require("ELEVENLABS_AGENT_ID")
    else:
        elevenlabs_api_key = _optional("ELEVENLABS_API_KEY")
        elevenlabs_agent_id = _optional("ELEVENLABS_AGENT_ID")

    llm_provider = _optional("IRIS_LLM_PROVIDER", "router").lower()
    valid_providers = {
        "router",
        "cloud",
        "cloud_router",
        "openrouter",
        "google",
        "gemini",
        "anthropic",
        "openai",
        "ollama",
        "mock",
    }
    if llm_provider not in valid_providers:
        raise ValueError(
            f"IRIS_LLM_PROVIDER must be one of {valid_providers}, got '{llm_provider}'"
        )

    archive_dir_str = _optional("IRIS_ARCHIVE_DIR", "memory/conversations")
    project_root = Path(__file__).resolve().parent.parent.parent
    archive_dir = (project_root / archive_dir_str).resolve()

    context_cache_dir_str = _optional("IRIS_CONTEXT_CACHE_DIR", "memory/context")
    context_cache_dir = (project_root / context_cache_dir_str).resolve()

    # Smart default for STT: if ElevenLabs key is present and valid, default to speech STT
    has_eleven_key = bool(elevenlabs_api_key and not elevenlabs_api_key.startswith("your_"))
    default_stt = "elevenlabs" if has_eleven_key else "console"

    dev_str = _optional("IRIS_INPUT_DEVICE")
    input_dev = int(dev_str) if dev_str.isdigit() else None

    openrouter_api_key = _optional("OPENROUTER_API_KEY")
    google_api_key = _optional("GOOGLE_API_KEY") or _optional("GEMINI_API_KEY")
    ollama_host = _optional("IRIS_OLLAMA_HOST") or _optional("OLLAMA_HOST", "http://localhost:11434")
    ollama_keep_alive = _optional("IRIS_OLLAMA_KEEP_ALIVE", "30m")

    default_model = ""
    if llm_provider in {"router", "cloud", "cloud_router"}:
        default_model = "router/free"
    elif llm_provider == "ollama":
        default_model = "llama3.2:1b"
    elif llm_provider == "openrouter":
        default_model = "openrouter/free"
    elif llm_provider in {"google", "gemini"}:
        default_model = "gemini-3.8-flash"
    llm_model = _optional("IRIS_LLM_MODEL") or default_model

    return IRISConfig(
        elevenlabs_api_key=elevenlabs_api_key,
        elevenlabs_agent_id=elevenlabs_agent_id,
        elevenlabs_voice_id=_optional("ELEVENLABS_VOICE_ID", "ThT5KcBeYPX3keUQqHPh"),
        llm_provider=llm_provider,
        ollama_host=ollama_host,
        ollama_keep_alive=ollama_keep_alive,
        openrouter_api_key=openrouter_api_key,
        anthropic_api_key=_optional("ANTHROPIC_API_KEY"),
        google_api_key=google_api_key,
        openai_api_key=_optional("OPENAI_API_KEY"),
        llm_model_name=llm_model,
        tts_provider=_optional("IRIS_TTS_PROVIDER", "edge").lower(),
        macos_voice=_optional("IRIS_MACOS_VOICE", "Flo (English (UK))"),
        kokoro_voice=_optional("IRIS_KOKORO_VOICE", "bf_emma").lower(),
        edge_voice=_optional("IRIS_EDGE_VOICE", "en-GB-SoniaNeural"),
        tts_speaking_rate=_optional_float("IRIS_TTS_RATE", 1.0),
        stt_provider=_optional("IRIS_STT_PROVIDER", default_stt).lower(),
        sample_rate=_optional_int("IRIS_SAMPLE_RATE", 16000),
        audio_input_device=input_dev,
        vad_energy_threshold=_optional_float("IRIS_VAD_THRESHOLD", 200.0),
        silence_timeout_ms=_optional_int("IRIS_SILENCE_TIMEOUT_MS", 750),
        cooldown_ms=_optional_int("IRIS_COOLDOWN_MS", 400),
        echo_similarity_threshold=_optional_float("IRIS_ECHO_THRESHOLD", 0.70),
        archive_dir=archive_dir,
        enable_location_tracking=_optional_bool("IRIS_LOCATION_TRACKING", True),
        fallback_location_name=_optional("IRIS_LOCATION_NAME", ""),
        fallback_latitude=_optional_float_or_none("IRIS_LOCATION_LATITUDE"),
        fallback_longitude=_optional_float_or_none("IRIS_LOCATION_LONGITUDE"),
        weather_provider=_optional("IRIS_WEATHER_PROVIDER", "open-meteo").lower(),
        weather_api_key=_optional("IRIS_WEATHER_API_KEY", ""),
        context_refresh_interval_s=_optional_int("IRIS_CONTEXT_REFRESH_INTERVAL", 1800),
        weather_cache_ttl_s=_optional_int("IRIS_WEATHER_CACHE_TTL", 1800),
        context_cache_dir=context_cache_dir,
        persona_name=_optional("IRIS_PERSONA_NAME", "IRIS"),
    )
