"""
iris/adapters/stt/__init__.py
-----------------------------
Speech-to-text adapters for IRIS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iris.adapters.stt.base import STTAdapter
from iris.adapters.stt.console_stt import ConsoleSTT
from iris.adapters.stt.elevenlabs_stt import ElevenLabsSTT
from iris.adapters.stt.mock_stt import MockSTT

if TYPE_CHECKING:
    from iris.core.config import IRISConfig


def create_stt_adapter(config: IRISConfig) -> STTAdapter:
    """
    Instantiate the configured STT adapter.
    """
    provider = config.stt_provider.lower()

    if provider == "elevenlabs":
        if not config.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY required for ElevenLabs STT provider.")
        return ElevenLabsSTT(api_key=config.elevenlabs_api_key)
    elif provider == "console":
        return ConsoleSTT()
    elif provider == "mock":
        return MockSTT()
    else:
        raise ValueError(f"Unsupported STT provider: '{provider}'. Supported: elevenlabs, console, mock")


__all__ = [
    "STTAdapter",
    "ElevenLabsSTT",
    "ConsoleSTT",
    "MockSTT",
    "create_stt_adapter",
]

