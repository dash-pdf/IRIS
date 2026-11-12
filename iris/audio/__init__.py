"""
iris.audio package
------------------
Local audio capture, VAD, echo filtering, playback, and Voice Presence system.
"""

from iris.audio.presence import (
    ClapDetectorConfig,
    DoubleClapDetector,
    PresenceState,
    VoicePresenceService,
    is_activation_phrase,
    is_deactivation_phrase,
)

__all__ = [
    "ClapDetectorConfig",
    "DoubleClapDetector",
    "PresenceState",
    "VoicePresenceService",
    "is_activation_phrase",
    "is_deactivation_phrase",
]
