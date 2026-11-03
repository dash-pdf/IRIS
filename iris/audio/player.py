"""
iris/audio/player.py
--------------------
Half-Duplex Audio Player using sounddevice.

Features:
- Thread-safe audio playback for numpy arrays / PCM raw bytes.
- Instantaneous cancellation / abort via stop().
- Callback on completion or cancellation.
- Status inspection via is_playing property.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    Manages audio playback through speakers with instant cancellation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_playing = False
        self._playback_thread: threading.Thread | None = None
        self._abort_event = threading.Event()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int = 24000,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """
        Play audio samples asynchronously.

        Args:
            samples: 1D float32 or int16 numpy array.
            sample_rate: Audio sampling frequency (e.g. 24000 for Kokoro).
            on_done: Optional callback fired when playback finishes normally.
        """
        self.stop()  # Stop any previous in-flight audio first

        with self._lock:
            self._abort_event.clear()
            self._is_playing = True
            self._playback_thread = threading.Thread(
                target=self._run_playback,
                args=(samples, sample_rate, on_done),
                name="iris-audio-player",
                daemon=True,
            )
            self._playback_thread.start()

    def play_bytes(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        dtype: str = "int16",
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """Play raw PCM audio bytes."""
        samples = np.frombuffer(pcm_bytes, dtype=dtype)
        self.play(samples, sample_rate=sample_rate, on_done=on_done)

    def play_queue(
        self,
        chunk_queue: queue.Queue[np.ndarray | None],
        sample_rate: int = 24000,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """
        Play a stream of audio chunks from a Queue until a None sentinel is received.

        Args:
            chunk_queue: Queue of 1D float32/int16 arrays, terminating with None.
            sample_rate: Sampling frequency.
            on_done: Callback fired when entire queue completes normally.
        """
        self.stop()  # Stop any previous in-flight audio first

        with self._lock:
            self._abort_event.clear()
            self._is_playing = True
            self._playback_thread = threading.Thread(
                target=self._run_playback_queue,
                args=(chunk_queue, sample_rate, on_done),
                name="iris-audio-stream-player",
                daemon=True,
            )
            self._playback_thread.start()

    def stop(self) -> None:
        """Immediately abort active playback."""
        self._abort_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        with self._lock:
            self._is_playing = False

    def wait(self, timeout: float | None = None) -> None:
        """Block until current playback finishes."""
        thread = self._playback_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def _run_playback(
        self,
        samples: np.ndarray,
        sample_rate: int,
        on_done: Callable[[], None] | None,
    ) -> None:
        import sounddevice as sd

        try:
            # Check abort before starting
            if self._abort_event.is_set():
                return

            sd.play(samples, samplerate=sample_rate)
            sd.wait()

            if not self._abort_event.is_set() and on_done:
                on_done()
        except Exception as exc:
            if not self._abort_event.is_set():
                logger.error("Audio playback error: %s", exc)
        finally:
            with self._lock:
                self._is_playing = False

    def _run_playback_queue(
        self,
        chunk_queue: queue.Queue[np.ndarray | None],
        sample_rate: int,
        on_done: Callable[[], None] | None,
    ) -> None:
        import sounddevice as sd

        try:
            while not self._abort_event.is_set():
                try:
                    chunk = chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if chunk is None or self._abort_event.is_set():
                    break

                if len(chunk) > 0:
                    sd.play(chunk, samplerate=sample_rate)
                    sd.wait()

            if not self._abort_event.is_set() and on_done:
                on_done()
        except Exception as exc:
            if not self._abort_event.is_set():
                logger.error("Audio stream queue playback error: %s", exc)
        finally:
            with self._lock:
                self._is_playing = False

