"""
iris/adapters/tts/__init__.py
-----------------------------
Text-to-speech adapters for IRIS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris.adapters.tts.base import TTSAdapter, shape_speech_text
from iris.adapters.tts.edge_tts import EdgeTTS
from iris.adapters.tts.elevenlabs_tts import ElevenLabsTTS
from iris.adapters.tts.kokoro_tts import KokoroTTS
from iris.adapters.tts.macos_tts import MacOSSayTTS
from iris.adapters.tts.mock_tts import MockTTS

if TYPE_CHECKING:
    from iris.core.config import IRISConfig


def create_tts_adapter(config: IRISConfig) -> TTSAdapter:
    """
    Instantiate the configured TTS adapter based on config.
    """
    provider = config.tts_provider.lower()

    if provider == "kokoro":
        return KokoroTTS(
            voice_name=getattr(config, "kokoro_voice", "bf_emma"),
            speed=config.tts_speaking_rate,
        )
    elif provider == "edge":
        return EdgeTTS(
            voice_name=getattr(config, "edge_voice", "en-GB-SoniaNeural"),
        )
    elif provider == "macos":
        # Calculate speaking rate (macOS say default rate ~175-185 wpm)
        wpm = int(185 * config.tts_speaking_rate)
        return MacOSSayTTS(voice_name=config.macos_voice, speaking_rate_wpm=wpm)
    elif provider == "elevenlabs":
        if not config.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY required for ElevenLabs TTS provider.")
        return ElevenLabsTTS(
            api_key=config.elevenlabs_api_key,
            voice_id=config.elevenlabs_voice_id,
        )
    elif provider == "mock":
        return MockTTS(sample_rate=config.sample_rate)
    else:
        raise ValueError(
            f"Unsupported TTS provider: '{provider}'. Supported: kokoro, edge, macos, elevenlabs, mock"
        )


__all__ = [
    "TTSAdapter",
    "shape_speech_text",
    "KokoroTTS",
    "EdgeTTS",
    "MacOSSayTTS",
    "ElevenLabsTTS",
    "MockTTS",
    "create_tts_adapter",
]
