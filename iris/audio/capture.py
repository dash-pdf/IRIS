"""
iris/audio/capture.py
---------------------
Microphone Audio Capture with Half-Duplex Mute Gating and Device Selection.

Features:
- Continuous 16kHz mono 16-bit PCM stream from selected/default input device.
- Half-duplex software mute gate: immediately drops incoming frames when muted.
- Protects against acoustic feedback while assistant is speaking.
- Detects persistent digital silence (e.g. macOS microphone permission missing or muted).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Microphone stream capture with mute control and device selection.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCK_SIZE = 800  # 50ms per frame at 16kHz

    def __init__(
        self,
        on_frame: Callable[[bytes], None] | None = None,
        device: int | str | None = None,
        on_silence_warning: Callable[[], None] | None = None,
    ) -> None:
        self._on_frame = on_frame
        self._device = device
        self.on_silence_warning = on_silence_warning
        self._is_muted = False
        self._is_running = False
        self._stream: Any = None
        self._lock = threading.Lock()

        # Silence monitoring
        self._consecutive_silent_frames = 0
        self._silence_warned = False

    @property
    def is_muted(self) -> bool:
        with self._lock:
            return self._is_muted

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    @property
    def device(self) -> int | str | None:
        return self._device

    def mute(self) -> None:
        """Mute microphone audio forwarding (used during SPEAKING and COOLDOWN)."""
        with self._lock:
            self._is_muted = True

    def unmute(self) -> None:
        """Unmute microphone audio forwarding (used when IDLE or LISTENING)."""
        with self._lock:
            self._is_muted = False

    def start(self, on_frame: Callable[[bytes], None] | None = None) -> None:
        """Start microphone capture stream."""
        if on_frame:
            self._on_frame = on_frame

        with self._lock:
            if self._is_running:
                return

            import sounddevice as sd

            kwargs: dict[str, Any] = {
                "samplerate": self.SAMPLE_RATE,
                "channels": self.CHANNELS,
                "dtype": "int16",
                "blocksize": self.BLOCK_SIZE,
                "callback": self._in_callback,
            }
            if self._device is not None:
                kwargs["device"] = self._device

            try:
                self._stream = sd.RawInputStream(**kwargs)
                self._stream.start()
                self._is_running = True
                self._consecutive_silent_frames = 0
                self._silence_warned = False
                dev_name = str(self._device) if self._device is not None else "default"
                logger.info("Microphone capture started (device: %s, 16kHz, mono, 50ms blocks)", dev_name)
            except Exception as exc:
                logger.error("Failed to start audio input stream: %s", exc)
                raise RuntimeError(f"Could not open microphone (device={self._device}): {exc}") from exc

    def stop(self) -> None:
        """Stop and close microphone stream."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False
            stream = self._stream
            self._stream = None

        if stream:
            try:
                if hasattr(stream, "active") and stream.active:
                    stream.stop()
                if hasattr(stream, "closed") and not stream.closed:
                    stream.close()
            except Exception:
                pass
        logger.info("Microphone capture stopped")

    def _in_callback(self, in_data: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            logger.debug("Audio input status flag: %s", status)

        with self._lock:
            if self._is_muted or not self._is_running or not self._on_frame:
                return  # Dropped when muted — half-duplex acoustic protection

        raw_bytes = bytes(in_data)

        # Digital silence check (detects missing macOS mic permissions or muted mic)
        # Check if entire frame consists of null bytes
        if not self._silence_warned:
            if not any(raw_bytes):
                self._consecutive_silent_frames += 1
                # 60 frames * 50ms = 3.0 seconds of pure silence
                if self._consecutive_silent_frames >= 60:
                    self._silence_warned = True
                    logger.warning(
                        "Microphone input has received 3 seconds of all-zero audio. "
                        "Check macOS System Settings -> Privacy & Security -> Microphone."
                    )
                    if self.on_silence_warning:
                        try:
                            self.on_silence_warning()
                        except Exception:
                            pass
            else:
                self._consecutive_silent_frames = 0

        # Forward audio chunk to registered consumer (VAD)
        try:
            self._on_frame(raw_bytes)
        except Exception as exc:
            logger.error("Error in audio capture callback: %s", exc)
