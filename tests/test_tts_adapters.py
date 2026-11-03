"""
tests/test_tts_adapters.py
--------------------------
Unit tests for iris/adapters/tts/ (Text-To-Speech Adapters).
"""

from __future__ import annotations

import numpy as np
import pytest

from iris.adapters.tts import (
    ElevenLabsTTS,
    MacOSSayTTS,
    MockTTS,
    create_tts_adapter,
    shape_speech_text,
)
from iris.core.config import IRISConfig


class TestTTSAdapters:
    def test_mock_tts_synthesis(self) -> None:
        tts = MockTTS(sample_rate=16000)
        assert tts.voice_name == "MockTTS"

        samples, sr = tts.synthesize("Hello world")
        assert sr == 16000
        assert isinstance(samples, np.ndarray)
        assert len(samples) > 0
        assert tts.last_synthesized_text == "Hello world"

    def test_mock_tts_empty_text(self) -> None:
        tts = MockTTS(sample_rate=16000)
        samples, sr = tts.synthesize("   ")
        assert len(samples) == 0

    def test_macos_tts_initialization(self) -> None:
        tts = MacOSSayTTS(voice_name="Flo (English (UK))", speaking_rate_wpm=185)
        assert tts.voice_name == "Flo (English (UK))"
        assert tts._rate == 185

    def test_elevenlabs_tts_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            ElevenLabsTTS(api_key="")

    def test_elevenlabs_tts_initialization(self) -> None:
        tts = ElevenLabsTTS(api_key="test-key", voice_id="ThT5KcBeYPX3keUQqHPh")
        assert "ThT5KcBeYPX3keUQqHPh" in tts.voice_name

    def test_create_tts_adapter_factory(self) -> None:
        cfg_mock = IRISConfig(tts_provider="mock")
        tts_mock = create_tts_adapter(cfg_mock)
        assert isinstance(tts_mock, MockTTS)

        cfg_macos = IRISConfig(tts_provider="macos", macos_voice="Shelley")
        tts_macos = create_tts_adapter(cfg_macos)
        assert isinstance(tts_macos, MacOSSayTTS)
        assert tts_macos.voice_name == "Shelley"

    def test_kokoro_tts_initialization_and_voices(self) -> None:
        from iris.adapters.tts.kokoro_tts import KokoroTTS

        tts_default = KokoroTTS()
        assert tts_default.voice_name == "bf_emma"

        tts_alice = KokoroTTS(voice_name="bf_alice")
        assert tts_alice.voice_name == "bf_alice"

        # Unknown voice fallback
        tts_fallback = KokoroTTS(voice_name="unknown_voice_xyz")
        assert tts_fallback.voice_name == "bf_emma"

    def test_kokoro_tts_factory(self) -> None:
        from iris.adapters.tts.kokoro_tts import KokoroTTS

        cfg = IRISConfig(tts_provider="kokoro", kokoro_voice="bf_isabella")
        tts = create_tts_adapter(cfg)
        assert isinstance(tts, KokoroTTS)
        assert tts.voice_name == "bf_isabella"

    def test_kokoro_tts_empty_text(self) -> None:
        from iris.adapters.tts.kokoro_tts import KokoroTTS

        tts = KokoroTTS()
        samples, sr = tts.generate_audio("   ")
        assert len(samples) == 0
        assert sr == 24000

    def test_kokoro_tts_synthesis(self) -> None:
        from iris.adapters.tts.kokoro_tts import KokoroTTS

        tts = KokoroTTS(voice_name="bf_emma")
        samples, sr = tts.synthesize("Hello world")
        assert sr == 24000
        assert isinstance(samples, np.ndarray)
        assert len(samples) > 0
        assert samples.dtype == np.float32

    def test_kokoro_sentence_splitting(self) -> None:
        from iris.adapters.tts.kokoro_tts import split_into_sentences

        text = "Good afternoon! I am IRIS. How can I help you today?"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Good afternoon!"
        assert sentences[1] == "I am IRIS."
        assert sentences[2] == "How can I help you today?"

    def test_edge_tts_initialization_and_voices(self) -> None:
        from iris.adapters.tts.edge_tts import EdgeTTS

        tts_default = EdgeTTS()
        assert tts_default.voice_name == "en-GB-SoniaNeural"

        tts_libby = EdgeTTS(voice_name="en-GB-LibbyNeural")
        assert tts_libby.voice_name == "en-GB-LibbyNeural"

        tts_alias = EdgeTTS(voice_name="maisie")
        assert tts_alias.voice_name == "en-GB-MaisieNeural"

        tts_fallback = EdgeTTS(voice_name="unknown-voice")
        assert tts_fallback.voice_name == "en-GB-SoniaNeural"

    def test_edge_tts_factory(self) -> None:
        from iris.adapters.tts.edge_tts import EdgeTTS

        cfg = IRISConfig(tts_provider="edge", edge_voice="en-GB-LibbyNeural")
        tts = create_tts_adapter(cfg)
        assert isinstance(tts, EdgeTTS)
        assert tts.voice_name == "en-GB-LibbyNeural"

    def test_edge_tts_empty_text(self) -> None:
        from iris.adapters.tts.edge_tts import EdgeTTS

        tts = EdgeTTS()
        samples, sr = tts.generate_audio("   ")
        assert len(samples) == 0
        assert sr == 24000

    def test_audio_player_queue_playback(self) -> None:
        import queue
        from iris.audio.player import AudioPlayer

        player = AudioPlayer()
        q: queue.Queue[np.ndarray | None] = queue.Queue()
        # Put empty chunk and sentinel
        q.put(np.zeros(100, dtype=np.float32))
        q.put(None)

        done_flag = False

        def on_done():
            nonlocal done_flag
            done_flag = True

        player.play_queue(q, sample_rate=24000, on_done=on_done)
        player.wait(timeout=1.0)
        player.stop()


class TestShapeSpeechText:
    def test_normal_prose_preserved(self) -> None:
        text = "Hello, how can I help you today?"
        assert shape_speech_text(text) == "Hello, how can I help you today?"

    def test_empty_and_whitespace_returns_empty(self) -> None:
        assert shape_speech_text("") == ""
        assert shape_speech_text("   \n\t  ") == ""
        assert shape_speech_text("```\n\n```") == ""

    def test_markdown_emphasis_stripped(self) -> None:
        raw = "This is **bold**, *italic*, ***bold-italic***, and ~~struck through~~."
        expected = "This is bold, italic, bold-italic, and struck through."
        assert shape_speech_text(raw) == expected

    def test_inline_code_and_backticks_stripped(self) -> None:
        raw = "Run the command `git status` or `pytest` now."
        expected = "Run the command git status or pytest now."
        assert shape_speech_text(raw) == expected

    def test_underscores_and_identifiers(self) -> None:
        # Emphasis underscores should be stripped, but snake_case preserved
        raw = "This is _emphasized_ text with model_name and api_key variables."
        expected = "This is emphasized text with model_name and api_key variables."
        assert shape_speech_text(raw) == expected

    def test_headings_stripped(self) -> None:
        raw = "# Main Heading\n## Subheading\n### Level 3\nSystem is online."
        expected = "Main Heading Subheading Level 3 System is online."
        assert shape_speech_text(raw) == expected

    def test_list_bullets_and_blockquotes_stripped(self) -> None:
        raw = "- First item\n* Second item\n+ Third item\n• Fourth item\n> A blockquote."
        expected = "First item Second item Third item Fourth item A blockquote."
        assert shape_speech_text(raw) == expected

    def test_markdown_links_preserve_label_only(self) -> None:
        raw = "Check [OpenRouter documentation](https://openrouter.ai/docs) for setup instructions."
        expected = "Check OpenRouter documentation for setup instructions."
        assert shape_speech_text(raw) == expected

    def test_raw_urls_stripped_preserving_sentence_punctuation(self) -> None:
        raw = "Visit https://openrouter.ai for details. Also see https://github.com."
        expected = "Visit for details. Also see."
        assert shape_speech_text(raw) == expected

    def test_raw_urls_with_trailing_exclamation(self) -> None:
        raw = "Check this out https://example.com/demo!"
        expected = "Check this out!"
        assert shape_speech_text(raw) == expected

    def test_numbers_and_conversational_punctuation_preserved(self) -> None:
        raw = "Version 3.14 was released with a 99.9% pass rate in 2026. What's next—ready?"
        expected = "Version 3.14 was released with a 99.9% pass rate in 2026. What's next—ready?"
        assert shape_speech_text(raw) == expected

    def test_code_blocks_remove_fences(self) -> None:
        raw = "Here is the snippet:\n```python\nprint('hello')\n```\nAll done."
        expected = "Here is the snippet: print('hello') All done."
        assert shape_speech_text(raw) == expected


