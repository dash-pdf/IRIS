"""
iris/audio/echo_filter.py
-------------------------
Acoustic Self-Echo Transcript Filter.

Detects if transcribed text is actually IRIS hearing its own speaker output
rather than genuine user input.

Criteria for echo detection:
1. Intentional override commands ("stop", "cancel", "wait", etc.) are never echoes.
2. Exact match with assistant's recent response.
3. Substring match (user transcript appears inside assistant text, >= 25 chars).
4. SequenceMatcher similarity ratio >= 0.70.
5. High content-word token overlap (>= 5 words, >= 3 content words, >= 80% overlap).
"""

from __future__ import annotations

import difflib
import logging
import re

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}

OVERRIDE_PATTERN = re.compile(
    r"^(?:(?:hey\s+)?iris\s+)?(?:please\s+)?"
    r"(?:"
    r"stop(?:\s+(?:it|now|please|talking))?"
    r"|cancel(?:\s+(?:that|it|please|this))?"
    r"|abort"
    r"|hold\s+on"
    r"|wait(?:\s+(?:a\s+(?:minute|second|moment)|up))?"
    r"|shut\s+up"
    r"|quiet"
    r"|be\s+quiet"
    r"|silence"
    r")(?:\s+please)?$"
)


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercased, stripped punctuation."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(cleaned.split())


class EchoFilter:
    """
    Evaluates whether an STT transcript is an acoustic echo of assistant speech.
    """

    def __init__(self, similarity_threshold: float = 0.70) -> None:
        self._similarity_threshold = similarity_threshold
        self._last_assistant_text: str = ""

    def register_assistant_response(self, text: str) -> None:
        """Register the text that IRIS is speaking aloud."""
        self._last_assistant_text = text.strip()

    def is_echo(self, transcript: str) -> tuple[bool, str]:
        """
        Check if transcript is an echo of the last assistant response.

        Returns:
            (is_echo, reason_string)
        """
        if not transcript.strip() or not self._last_assistant_text:
            return False, "clean"

        norm_user = _normalize(transcript)
        norm_asst = _normalize(self._last_assistant_text)

        if not norm_user or not norm_asst:
            return False, "clean"

        # 0. Override command check: explicit stop/cancel commands are intentional user inputs
        if OVERRIDE_PATTERN.match(norm_user):
            return False, "clean"

        # 1. Exact match
        if norm_user == norm_asst:
            return True, "exact_match"

        # 2. Substring match (user transcript appears inside assistant text)
        # Require >= 25 characters to prevent discarding short follow-up choices
        if len(norm_user) >= 25 and norm_user in norm_asst:
            return True, f"substring_match: user utterance '{norm_user}' found in assistant speech"

        # 3. SequenceMatcher fuzzy similarity ratio
        ratio = difflib.SequenceMatcher(None, norm_user, norm_asst).ratio()
        if ratio >= self._similarity_threshold:
            return True, f"fuzzy_similarity: ratio {ratio:.2f} >= {self._similarity_threshold}"

        # 4. Token overlap (Jaccard similarity on content words)
        user_words = set(norm_user.split())
        asst_words = set(norm_asst.split())
        if user_words and asst_words:
            content_words = user_words - STOP_WORDS
            if len(user_words) >= 5 and len(content_words) >= 3:
                intersection = user_words.intersection(asst_words)
                overlap = len(intersection) / len(user_words)
                content_intersection = content_words.intersection(asst_words)
                content_overlap = len(content_intersection) / len(content_words)
                if overlap >= 0.80 and content_overlap >= 0.75:
                    return True, f"word_overlap: {overlap:.2f} of words matched assistant output"

        return False, "clean"

