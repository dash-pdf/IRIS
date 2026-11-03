"""
tests/test_state.py
-------------------
Unit tests for iris/core/state.py (Audio State Machine).
"""

from __future__ import annotations

import pytest

from iris.core.state import AudioState, AudioStateMachine


class TestAudioStateMachine:
    def test_initial_state_is_idle(self) -> None:
        sm = AudioStateMachine()
        assert sm.state == AudioState.IDLE
        assert sm.is_mic_active is True
        assert sm.is_speaking is False

    def test_legal_lifecycle_progression(self) -> None:
        sm = AudioStateMachine()
        turn_id = sm.new_turn()

        # IDLE -> LISTENING
        assert sm.transition_to(AudioState.LISTENING, expected_turn_id=turn_id) is True
        assert sm.state == AudioState.LISTENING
        assert sm.is_mic_active is True

        # LISTENING -> THINKING
        assert sm.transition_to(AudioState.THINKING, expected_turn_id=turn_id) is True
        assert sm.state == AudioState.THINKING
        assert sm.is_mic_active is False

        # THINKING -> SPEAKING
        assert sm.transition_to(AudioState.SPEAKING, expected_turn_id=turn_id) is True
        assert sm.state == AudioState.SPEAKING
        assert sm.is_speaking is True
        assert sm.is_mic_active is False

        # SPEAKING -> COOLDOWN
        assert sm.transition_to(AudioState.COOLDOWN, expected_turn_id=turn_id) is True
        assert sm.state == AudioState.COOLDOWN
        assert sm.is_mic_active is False

        # COOLDOWN -> IDLE
        assert sm.transition_to(AudioState.IDLE, expected_turn_id=turn_id) is True
        assert sm.state == AudioState.IDLE
        assert sm.is_mic_active is True

    def test_direct_idle_to_thinking_for_text_turns(self) -> None:
        sm = AudioStateMachine()
        turn_id = sm.new_turn()
        assert sm.transition_to(AudioState.THINKING, expected_turn_id=turn_id) is True
        assert sm.state == AudioState.THINKING

    def test_illegal_transition_rejected(self) -> None:
        sm = AudioStateMachine()
        turn_id = sm.new_turn()

        # Cannot jump from IDLE to SPEAKING directly
        assert sm.transition_to(AudioState.SPEAKING, expected_turn_id=turn_id) is False
        assert sm.state == AudioState.IDLE

        # Move to SPEAKING legally
        sm.transition_to(AudioState.LISTENING, expected_turn_id=turn_id)
        sm.transition_to(AudioState.THINKING, expected_turn_id=turn_id)
        sm.transition_to(AudioState.SPEAKING, expected_turn_id=turn_id)

        # Cannot jump from SPEAKING to LISTENING (half-duplex violation)
        assert sm.transition_to(AudioState.LISTENING, expected_turn_id=turn_id) is False
        assert sm.state == AudioState.SPEAKING

    def test_stale_turn_transition_rejected(self) -> None:
        sm = AudioStateMachine()
        turn_1 = sm.new_turn()
        turn_2 = sm.new_turn()

        # Transition with old turn_1 must be rejected
        assert sm.transition_to(AudioState.LISTENING, expected_turn_id=turn_1) is False
        assert sm.state == AudioState.IDLE

        # Transition with current turn_2 must succeed
        assert sm.transition_to(AudioState.LISTENING, expected_turn_id=turn_2) is True
        assert sm.state == AudioState.LISTENING

    def test_cancel_resets_to_idle(self) -> None:
        sm = AudioStateMachine()
        turn_id = sm.new_turn()
        sm.transition_to(AudioState.LISTENING, expected_turn_id=turn_id)
        sm.cancel(reason="test_cancel")
        assert sm.state == AudioState.IDLE
        assert sm.is_mic_active is True

    def test_state_change_callback_fires(self) -> None:
        events = []
        sm = AudioStateMachine(on_change=lambda old, new, turn: events.append((old, new)))
        sm.transition_to(AudioState.LISTENING)
        sm.transition_to(AudioState.THINKING)

        assert len(events) == 2
        assert events[0] == (AudioState.IDLE, AudioState.LISTENING)
        assert events[1] == (AudioState.LISTENING, AudioState.THINKING)

