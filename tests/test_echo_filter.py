"""
tests/test_echo_filter.py
-------------------------
Unit tests for iris/audio/echo_filter.py (Acoustic Self-Echo Detection).
"""

from __future__ import annotations

import pytest

from iris.audio.echo_filter import EchoFilter


class TestEchoFilter:
    def test_clean_input_when_no_prior_response(self) -> None:
        ef = EchoFilter()
        is_echo, reason = ef.is_echo("Good morning IRIS")
        assert is_echo is False
        assert reason == "clean"

    def test_exact_match_detected_as_echo(self) -> None:
        ef = EchoFilter()
        ef.register_assistant_response("I have completed the analysis for you.")
        is_echo, reason = ef.is_echo("I have completed the analysis for you.")
        assert is_echo is True
        assert reason == "exact_match"

    def test_case_and_punctuation_insensitive_match(self) -> None:
        ef = EchoFilter()
        ef.register_assistant_response("All systems nominal, sir!")
        is_echo, reason = ef.is_echo("all systems nominal sir")
        assert is_echo is True
        assert "exact_match" in reason

    def test_substring_echo_detected(self) -> None:
        ef = EchoFilter()
        ef.register_assistant_response(
            "Certainly, the architecture decouples speech recognition from LLM reasoning."
        )
        # Mic caught partial phrase spoken by speakers
        is_echo, reason = ef.is_echo("decouples speech recognition from LLM reasoning")
        assert is_echo is True
        assert "substring_match" in reason

    def test_fuzzy_high_similarity_detected(self) -> None:
        ef = EchoFilter(similarity_threshold=0.70)
        ef.register_assistant_response("Everything is functioning within expected parameters.")
        # Non-substring fuzzy match (different suffix/spelling)
        is_echo, reason = ef.is_echo("Everything is functioning within expected parametres")
        assert is_echo is True
        assert "fuzzy_similarity" in reason

    def test_word_overlap_detected(self) -> None:
        ef = EchoFilter(similarity_threshold=0.85)  # Higher fuzzy threshold so word overlap takes over
        ef.register_assistant_response(
            "The quick brown fox jumps over the lazy dog in the garden."
        )
        is_echo, reason = ef.is_echo("quick brown fox jumps lazy dog garden")
        assert is_echo is True
        assert "word_overlap" in reason or "fuzzy_similarity" in reason

    def test_distinct_user_utterance_not_flagged(self) -> None:
        ef = EchoFilter()
        ef.register_assistant_response("I can help you review the project milestones.")
        is_echo, reason = ef.is_echo("What is the current time in London?")
        assert is_echo is False
        assert reason == "clean"

    def test_empty_utterance_handling(self) -> None:
        ef = EchoFilter()
        ef.register_assistant_response("Ready when you are.")
        is_echo, reason = ef.is_echo("")
        assert is_echo is False
        assert reason == "clean"

    def test_override_commands_bypass_echo_filter(self) -> None:
        ef = EchoFilter()
        ef.register_assistant_response("Please stop and wait while I process the request.")

        # Test various override commands
        overrides = [
            "stop",
            "please stop",
            "stop please",
            "stop now",
            "cancel",
            "cancel that",
            "please cancel",
            "abort",
            "hold on",
            "wait",
            "wait a second",
            "quiet",
            "be quiet",
            "silence",
            "iris stop",
            "hey iris stop",
        ]
        for cmd in overrides:
            is_echo, reason = ef.is_echo(cmd)
            assert is_echo is False, f"Command '{cmd}' was falsely flagged as echo"
            assert reason == "clean"

    def test_no_loophole_sentences_with_no_evaluated_normally(self) -> None:
        ef = EchoFilter()
        ef.register_assistant_response("There is no evidence of failure in the system.")

        # Exact match must still be caught despite containing 'no'
        is_echo, reason = ef.is_echo("There is no evidence of failure in the system.")
        assert is_echo is True
        assert reason == "exact_match"

        # Substring echo >= 25 chars must still be caught
        is_echo, reason = ef.is_echo("no evidence of failure in the system")
        assert is_echo is True
        assert "substring_match" in reason

    def test_short_followup_commands_not_flagged(self) -> None:
        ef = EchoFilter()
        # Assistant presents choices that contain the exact words user might say
        ef.register_assistant_response(
            "I found three options: the first one, the second one, and the third one."
        )
        is_echo, reason = ef.is_echo("the second one")
        assert is_echo is False
        assert reason == "clean"

        ef.register_assistant_response("I can provide a summary or detail each component.")
        is_echo, reason = ef.is_echo("read the summary")
        assert is_echo is False
        assert reason == "clean"

        ef.register_assistant_response("Would you like to run the tests now?")
        is_echo, reason = ef.is_echo("run the tests")
        assert is_echo is False
        assert reason == "clean"
