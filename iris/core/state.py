"""
iris/core/state.py
------------------
Explicit Audio State Machine for IRIS.

Coordinates the half-duplex conversational turn lifecycle:
    IDLE -> LISTENING -> THINKING -> SPEAKING -> COOLDOWN -> IDLE

Key Responsibilities:
- Enforce strict legal state transitions.
- Assign a unique UUID turn_id to each user utterance turn.
- Guard against microphone feedback by tracking active states.
- Support thread-safe state inspection and transition callbacks.
- Support instant cancellation and error reset.
"""

from __future__ import annotations

import enum
import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class AudioState(str, enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    COOLDOWN = "COOLDOWN"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


# Transition validity matrix
_LEGAL_TRANSITIONS: dict[AudioState, set[AudioState]] = {
    AudioState.IDLE: {AudioState.LISTENING, AudioState.THINKING, AudioState.ERROR, AudioState.CANCELLED},
    AudioState.LISTENING: {AudioState.THINKING, AudioState.IDLE, AudioState.ERROR, AudioState.CANCELLED},
    AudioState.THINKING: {AudioState.SPEAKING, AudioState.IDLE, AudioState.ERROR, AudioState.CANCELLED},
    AudioState.SPEAKING: {AudioState.COOLDOWN, AudioState.IDLE, AudioState.ERROR, AudioState.CANCELLED},
    AudioState.COOLDOWN: {AudioState.IDLE, AudioState.LISTENING, AudioState.ERROR, AudioState.CANCELLED},
    AudioState.ERROR: {AudioState.IDLE},
    AudioState.CANCELLED: {AudioState.IDLE},
}


StateCallback = Callable[[AudioState, AudioState, str], None]


class AudioStateMachine:
    """
    Thread-safe state machine governing the audio turn lifecycle.
    """

    def __init__(self, on_change: StateCallback | None = None) -> None:
        self._lock = threading.Lock()
        self._state: AudioState = AudioState.IDLE
        self._current_turn_id: str = str(uuid.uuid4())
        self._callbacks: list[StateCallback] = []
        if on_change:
            self._callbacks.append(on_change)

    @property
    def state(self) -> AudioState:
        with self._lock:
            return self._state

    @property
    def current_turn_id(self) -> str:
        with self._lock:
            return self._current_turn_id

    @property
    def is_mic_active(self) -> bool:
        """Microphone should ONLY record when IDLE or LISTENING."""
        with self._lock:
            return self._state in {AudioState.IDLE, AudioState.LISTENING}

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._state == AudioState.SPEAKING

    def add_listener(self, callback: StateCallback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def new_turn(self) -> str:
        """Generate and assign a new turn identifier."""
        with self._lock:
            self._current_turn_id = str(uuid.uuid4())
            return self._current_turn_id

    def transition_to(self, new_state: AudioState, expected_turn_id: str | None = None) -> bool:
        """
        Transition to a new state if legal and turn matches.

        Args:
            new_state: Target AudioState.
            expected_turn_id: If specified, transition only succeeds if
                              current_turn_id matches expected_turn_id.

        Returns:
            True if transition succeeded, False if rejected.
        """
        callbacks_to_fire = []
        old_state = AudioState.IDLE
        turn_id = ""

        with self._lock:
            if expected_turn_id and expected_turn_id != self._current_turn_id:
                logger.warning(
                    "Stale transition rejected: expected turn %s but state machine is on turn %s",
                    expected_turn_id,
                    self._current_turn_id,
                )
                return False

            if new_state not in _LEGAL_TRANSITIONS.get(self._state, set()):
                logger.warning(
                    "Illegal state transition attempted: %s -> %s",
                    self._state,
                    new_state,
                )
                return False

            old_state = self._state
            self._state = new_state
            turn_id = self._current_turn_id
            callbacks_to_fire = list(self._callbacks)

        logger.info("[State] %s -> %s (turn: %s)", old_state.value, new_state.value, turn_id[:8])

        for cb in callbacks_to_fire:
            try:
                cb(old_state, new_state, turn_id)
            except Exception as exc:
                logger.exception("Error in state change callback: %s", exc)

        return True

    def cancel(self, reason: str = "user_cancel") -> None:
        """Immediately cancel ongoing operations and reset towards IDLE."""
        logger.info("Cancelling state %s (reason: %s)", self.state.value, reason)
        self.transition_to(AudioState.CANCELLED)
        self.transition_to(AudioState.IDLE)
        self.new_turn()

    def reset_to_idle(self) -> None:
        """Safely force reset to IDLE."""
        with self._lock:
            self._state = AudioState.IDLE
            self._current_turn_id = str(uuid.uuid4())

    def reset(self) -> None:
        """Alias for reset_to_idle."""
        self.reset_to_idle()
