"""
iris/adapters/tts/kokoro_tts.py
-------------------------------
Standalone Kokoro Text-To-Speech Adapter for IRIS on macOS.

Features:
- Completely offline, local neural speech synthesis using Kokoro-82M ONNX Runtime.
- First-class support for British female voice presets:
    - bf_emma (default) - articulate, modern British female voice
    - bf_alice - refined, calm British female voice
    - bf_isabella - poised, clear British female voice
- Simple standalone interface:
    speak(text, voice=None, speed=1.0, blocking=False)
    generate_audio(text, voice=None, speed=1.0) -> tuple[np.ndarray, int]
    stop()
- Model is loaded once and reused across requests.
- Decoupled from LLM, STT, and agent runtime.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from iris.adapters.tts.base import TTSAdapter
from iris.audio.player import AudioPlayer

logger = logging.getLogger(__name__)

# Allowed British female voices
BRITISH_FEMALE_VOICES = {
    "bf_emma": "British Female - Emma (Default)",
    "bf_alice": "British Female - Alice",
    "bf_isabella": "British Female - Isabella",
}
DEFAULT_VOICE = "bf_emma"


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting common sentence boundaries."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences if sentences else [text.strip()]


class KokoroTTS(TTSAdapter):
    """
    Local Kokoro TTS adapter powered by ONNX Runtime with sentence pipelined streaming.
    """

    # Shared class-level engine cache so multiple instances do not reload weights
    _cached_engine: Any = None
    _engine_lock = threading.Lock()

    def __init__(
        self,
        model_path: str | Path | None = None,
        voices_path: str | Path | None = None,
        voice_name: str = DEFAULT_VOICE,
        speed: float = 1.0,
        player: AudioPlayer | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.model_path = Path(model_path) if model_path else (project_root / "models/kokoro/kokoro-v1.0.onnx")
        self.voices_path = Path(voices_path) if voices_path else (project_root / "models/kokoro/voices-v1.0.bin")

        self._voice_name = self._validate_voice(voice_name)
        self.speed = float(speed)
        self._player = player or AudioPlayer()
        self._engine: Any = None
        self._stop_requested = threading.Event()
        self._synthesis_thread: threading.Thread | None = None

    def _validate_voice(self, voice: str) -> str:
        clean = voice.strip().lower()
        if clean in BRITISH_FEMALE_VOICES:
            return clean
        logger.warning(
            "Requested voice '%s' is not in British female presets (%s). Defaulting to '%s'.",
            voice,
            list(BRITISH_FEMALE_VOICES.keys()),
            DEFAULT_VOICE,
        )
        return DEFAULT_VOICE

    @property
    def voice_name(self) -> str:
        return self._voice_name

    @voice_name.setter
    def voice_name(self, value: str) -> None:
        self._voice_name = self._validate_voice(value)

    def _get_engine(self) -> Any:
        """Load the Kokoro model once and cache it in memory."""
        if KokoroTTS._cached_engine is not None:
            return KokoroTTS._cached_engine

        with KokoroTTS._engine_lock:
            if KokoroTTS._cached_engine is not None:
                return KokoroTTS._cached_engine

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Kokoro model file not found at: {self.model_path}. "
                    "Ensure models/kokoro/kokoro-v1.0.onnx has been downloaded."
                )
            if not self.voices_path.exists():
                raise FileNotFoundError(
                    f"Kokoro voices file not found at: {self.voices_path}. "
                    "Ensure models/kokoro/voices-v1.0.bin has been downloaded."
                )

            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise ImportError(
                    "kokoro-onnx package is required for KokoroTTS. "
                    "Run: pip install kokoro-onnx soundfile"
                ) from exc

            logger.info("Loading Kokoro ONNX model into memory from %s...", self.model_path.name)
            engine = Kokoro(str(self.model_path), str(self.voices_path))
            KokoroTTS._cached_engine = engine
            return engine

    def generate_audio(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate raw audio samples from text using the cached Kokoro model.

        Args:
            text: Text string to synthesize.
            voice: Voice name override (e.g. 'bf_emma', 'bf_alice', 'bf_isabella').
            speed: Speaking speed multiplier (default: 1.0).

        Returns:
            Tuple of (samples, sample_rate), where samples is a 1D float32 numpy array
            and sample_rate is 24000.
        """
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(0, dtype=np.float32), 24000

        target_voice = self._validate_voice(voice) if voice else self._voice_name
        target_speed = float(speed) if speed is not None else self.speed

        engine = self._get_engine()
        logger.info(
            "Generating speech via Kokoro TTS (voice=%s, speed=%.2f, lang=en-gb)",
            target_voice,
            target_speed,
        )

        try:
            samples, sample_rate = engine.create(
                clean_text,
                voice=target_voice,
                speed=target_speed,
                lang="en-gb",
            )
            # Ensure float32 array
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples, dtype=np.float32)
            return samples, sample_rate
        except Exception as exc:
            logger.error("Kokoro TTS generation failed: %s", exc)
            raise RuntimeError(f"Kokoro speech synthesis error: {exc}") from exc

    def speak(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        blocking: bool = False,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """
        Synthesize text and play audio through speakers.

        Uses sentence-pipelined streaming for multi-sentence responses:
        Synthesizes Sentence 1 immediately and starts speaker playback, while
        subsequent sentences are synthesized in the background concurrently.
        """
        clean_text = text.strip()
        if not clean_text:
            if on_done:
                on_done()
            return

        sentences = split_into_sentences(clean_text)

        # Single sentence: synthesize and play directly
        if len(sentences) <= 1:
            samples, sample_rate = self.generate_audio(sentences[0], voice=voice, speed=speed)
            if len(samples) == 0:
                if on_done:
                    on_done()
                return
            self._player.play(samples, sample_rate=sample_rate, on_done=on_done)
            if blocking:
                self._player.wait()
            return

        # Multi-sentence: pipelined streaming synthesis & playback
        chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop_requested.clear()

        def _producer() -> None:
            try:
                for sent in sentences:
                    if self._stop_requested.is_set():
                        break
                    samples, _ = self.generate_audio(sent, voice=voice, speed=speed)
                    if self._stop_requested.is_set():
                        break
                    if len(samples) > 0:
                        chunk_queue.put(samples)
            except Exception as exc:
                logger.error("Error in Kokoro streaming sentence synthesis: %s", exc)
            finally:
                chunk_queue.put(None)

        self._synthesis_thread = threading.Thread(
            target=_producer,
            name="kokoro-pipelined-synth",
            daemon=True,
        )
        self._synthesis_thread.start()

        self._player.play_queue(chunk_queue, sample_rate=24000, on_done=on_done)
        if blocking:
            self._player.wait()
            if self._synthesis_thread and self._synthesis_thread.is_alive():
                self._synthesis_thread.join(timeout=1.0)

    def stop(self) -> None:
        """Immediately abort any active audio playback and background synthesis."""
        self._stop_requested.set()
        self._player.stop()
        if self._synthesis_thread and self._synthesis_thread.is_alive():
            if threading.current_thread() != self._synthesis_thread:
                self._synthesis_thread.join(timeout=0.5)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Satisfies the TTSAdapter base contract for the core IRIS assistant orchestrator.
        """
        return self.generate_audio(text)

