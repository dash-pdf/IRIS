"""
tests/test_conversation.py
---------------------------
Unit tests for iris/services/conversation.py — the conversation state manager.

These tests run entirely offline. No API keys or network access required.
"""

from __future__ import annotations

import pytest

from iris.services.conversation import ConversationService, IRIS_SYSTEM_PROMPT, Turn


class TestConversationServiceBasics:
    def test_starts_empty(self) -> None:
        """A new ConversationService must have zero turns."""
        conv = ConversationService()
        assert conv.turn_count == 0

    def test_system_prompt_in_messages(self) -> None:
        """get_messages() must always include the system prompt as first message."""
        conv = ConversationService()
        messages = conv.get_messages()
        assert messages[0]["role"] == "system"
        assert len(messages[0]["content"]) > 0

    def test_add_user_turn(self) -> None:
        """Adding a user turn increments turn_count and appears in messages."""
        conv = ConversationService()
        conv.add_user_turn("Hello IRIS.")
        assert conv.turn_count == 1
        messages = conv.get_messages()
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == "Hello IRIS."

    def test_add_assistant_turn(self) -> None:
        """Adding an assistant turn appears in messages with role 'assistant'."""
        conv = ConversationService()
        conv.add_user_turn("Hello.")
        conv.add_assistant_turn("Hello there.")
        assert conv.turn_count == 2
        messages = conv.get_messages()
        asst = [m for m in messages if m["role"] == "assistant"]
        assert asst[0]["content"] == "Hello there."

    def test_messages_order(self) -> None:
        """Messages must appear in system → user → assistant order."""
        conv = ConversationService()
        conv.add_user_turn("Hi.")
        conv.add_assistant_turn("Hey.")
        conv.add_user_turn("How are you?")

        messages = conv.get_messages()
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"

    def test_clear_resets_history(self) -> None:
        """clear() must reset turn_count to zero."""
        conv = ConversationService()
        conv.add_user_turn("Hi.")
        conv.add_assistant_turn("Hey.")
        conv.clear()
        assert conv.turn_count == 0

    def test_clear_preserves_system_prompt(self) -> None:
        """clear() must keep the system prompt in get_messages()."""
        conv = ConversationService()
        conv.add_user_turn("Hi.")
        conv.clear()
        messages = conv.get_messages()
        assert messages[0]["role"] == "system"
        assert len(messages) == 1  # only system prompt remains


class TestLastTurnHelpers:
    def test_get_last_user_text_none_when_empty(self) -> None:
        """get_last_user_text() returns None on empty history."""
        conv = ConversationService()
        assert conv.get_last_user_text() is None

    def test_get_last_user_text(self) -> None:
        """get_last_user_text() returns the most recent user message."""
        conv = ConversationService()
        conv.add_user_turn("First")
        conv.add_user_turn("Second")
        assert conv.get_last_user_text() == "Second"

    def test_get_last_assistant_text_none_when_empty(self) -> None:
        """get_last_assistant_text() returns None if no assistant turns yet."""
        conv = ConversationService()
        conv.add_user_turn("Hi.")
        assert conv.get_last_assistant_text() is None

    def test_get_last_assistant_text(self) -> None:
        """get_last_assistant_text() returns the most recent assistant reply."""
        conv = ConversationService()
        conv.add_user_turn("Hi.")
        conv.add_assistant_turn("Hello.")
        conv.add_user_turn("How are you?")
        conv.add_assistant_turn("Very well, thank you.")
        assert conv.get_last_assistant_text() == "Very well, thank you."


class TestSlidingWindow:
    def test_window_trims_oldest_turns(self) -> None:
        """With max_turns=4, adding 6 turns should keep only the 4 newest."""
        conv = ConversationService(max_turns=4)
        for i in range(3):
            conv.add_user_turn(f"User {i}")
            conv.add_assistant_turn(f"IRIS {i}")

        # 6 turns added, max is 4 — oldest 2 should be gone.
        assert conv.turn_count == 4

    def test_system_prompt_survives_trimming(self) -> None:
        """System prompt must remain in get_messages() even after heavy trimming."""
        conv = ConversationService(max_turns=2)
        for i in range(10):
            conv.add_user_turn(f"Message {i}")
            conv.add_assistant_turn(f"Reply {i}")

        messages = conv.get_messages()
        assert messages[0]["role"] == "system"

    def test_no_trimming_within_limit(self) -> None:
        """Turn count must not be reduced when under the limit."""
        conv = ConversationService(max_turns=10)
        for i in range(4):
            conv.add_user_turn(f"U{i}")
            conv.add_assistant_turn(f"A{i}")
        assert conv.turn_count == 8


class TestPersonalityPrompt:
    def test_default_system_prompt_mentions_iris(self) -> None:
        """The default system prompt must reference IRIS by name."""
        assert "IRIS" in IRIS_SYSTEM_PROMPT

    def test_custom_system_prompt(self) -> None:
        """A custom system prompt must appear as the first message."""
        custom_prompt = "You are a pirate."
        conv = ConversationService(system_prompt=custom_prompt)
        messages = conv.get_messages()
        assert messages[0]["content"] == custom_prompt

    def test_user_input_stripped(self) -> None:
        """Leading/trailing whitespace is stripped from user input."""
        conv = ConversationService()
        conv.add_user_turn("   hello   ")
        assert conv.get_last_user_text() == "hello"


class TestRestoreTurns:
    def test_restore_turns_sets_history(self) -> None:
        """Restoring turns updates turn_count and message contents."""
        conv = ConversationService()
        conv.restore_turns([
            Turn(role="user", content="Hello IRIS."),
            Turn(role="assistant", content="Hello, how can I help?"),
        ])
        assert conv.turn_count == 2
        assert conv.get_last_user_text() == "Hello IRIS."
        assert conv.get_last_assistant_text() == "Hello, how can I help?"

    def test_restore_turns_preserves_system_prompt(self) -> None:
        """Restoring turns retains the system prompt as the first message."""
        conv = ConversationService(system_prompt="Custom System Prompt")
        conv.restore_turns([
            Turn(role="user", content="Question"),
            Turn(role="assistant", content="Answer"),
        ])
        messages = conv.get_messages()
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "Custom System Prompt"}
        assert messages[1] == {"role": "user", "content": "Question"}
        assert messages[2] == {"role": "assistant", "content": "Answer"}

    def test_restore_turns_trims_sliding_window(self) -> None:
        """Restoring more turns than max_turns trims oldest pairs."""
        conv = ConversationService(max_turns=4)
        turns = [
            Turn(role="user", content=f"U{i}") if i % 2 == 0
            else Turn(role="assistant", content=f"A{i}")
            for i in range(10)
        ]
        conv.restore_turns(turns)
        assert conv.turn_count <= 4


