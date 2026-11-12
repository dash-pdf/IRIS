"""
iris/audio/presence.py
----------------------
Local Voice Presence System and Double-Clap Acoustic Gesture Detector for IRIS.

Responsibilities:
- Coordinate hands-free presence states (`DISABLED`, `DORMANT`, `ARMED`,
  `PROCESSING`, `SPEAKING`, `ERROR`) with `AudioStateMachine`.
- Provide local transient double-clap detection with background noise tracking,
  high-frequency snap weighting, decay/brevity checks, inter-clap timing bounds,
  cooldown hysteresis, and zero cloud/remote transmission while dormant.
- Detect deterministic activation/deactivation phrases without invoking an LLM.
- Enforce configurable active-listening inactivity timeout and manual overrides.
"""

from __future__ import annotations

import enum
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from iris.core.state import AudioState

logger = logging.getLogger(__name__)


class PresenceState(str, enum.Enum):
    """Operational state of the IRIS Voice Presence controller."""

    DISABLED = "DISABLED"      # Mic hardware capture stopped completely
    DORMANT = "DORMANT"        # Mic active locally for clap standby ONLY (zero cloud/STT)
    ARMED = "ARMED"            # Active command listening (VAD + STT pipeline ready)
    PROCESSING = "PROCESSING"  # Captured utterance / command being transcribed or answered
    SPEAKING = "SPEAKING"      # IRIS speaking aloud (clap detection gated/suppressed)
    ERROR = "ERROR"            # Recoverable hardware/permission error


_DEACTIVATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:(?:hey\s+)?iris[,\s]+)?(?:please\s+)?"
        r"(?:stop\s+listening|go\s+to\s+standby|return\s+to\s+standby|"
        r"deactivate\s+listening|disable\s+listening|standby\s+mode|"
        r"go\s+dormant|that(?:'s|s|\s+is)\s+all)"
        r"(?:\s+(?:for\s+now|please|iris))?[.!?,\s]*$",
        re.IGNORECASE,
    ),
)


def _normalize_phrase(text: str) -> str:
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(cleaned.split())


def is_deactivation_phrase(transcript: str) -> bool:
    """
    Deterministically detect natural voice deactivation phrases without an LLM.
    Examples: "Stop listening.", "Go to standby.", "Deactivate listening.", "That's all."
    """
    raw = transcript.strip()
    if not raw:
        return False
    return any(pat.match(raw) for pat in _DEACTIVATION_PATTERNS)


def is_activation_phrase(transcript: str, configured_phrase: str = "hey iris") -> bool:
    """
    Deterministically check whether `transcript` matches an optional configured
    activation phrase (empty `configured_phrase` disables phrase activation).
    """
    phrase_norm = _normalize_phrase(configured_phrase)
    text_norm = _normalize_phrase(transcript)
    if not phrase_norm or not text_norm:
        return False
    return text_norm == phrase_norm or text_norm in {
        f"activate {phrase_norm}",
        f"wake up {phrase_norm}",
    }


@dataclass
class ClapDetectorConfig:
    """Tunable parameters for local acoustic double-clap recognition."""

    sample_rate: int = 16000
    sensitivity: float = 0.6             # [0.1 .. 1.0], higher detects quieter claps
    min_peak_amplitude: float = 2600.0   # Base 16-bit PCM peak amplitude threshold
    snr_threshold: float = 4.2           # Required RMS spike ratio above adaptive noise floor
    min_high_freq_ratio: float = 0.28    # Rejects low-frequency thuds (desk bumps, doors)
    max_impulse_duration_ms: float = 135.0  # Rejects continuous speech/music sustains
    min_gap_s: float = 0.14              # Ignore room echo / bounce closer than 140ms
    max_gap_s: float = 0.85              # Max interval allowed between clap 1 and clap 2
    cooldown_s: float = 1.50             # Refractory cooldown after triggering double-clap
    quiet_gap_decay_ratio: float = 0.35  # Decay level marking clean impulse termination


class DoubleClapDetector:
    """
    Local, privacy-safe acoustic double-clap transient detector.

    Privacy invariant:
    - Converts each incoming PCM chunk strictly into transient scalar energy metrics
      and retains ZERO PCM audio buffers beyond the synchronous `process_frame()` call.
    """

    def __init__(self, config: ClapDetectorConfig | None = None) -> None:
        self.config = config or ClapDetectorConfig()
        self._lock = threading.Lock()
        self._noise_floor_rms: float = 60.0
        self._in_impulse: bool = False
        self._impulse_start_t: float = 0.0
        self._impulse_peak_t: float = 0.0
        self._impulse_peak_rms: float = 0.0
        self._impulse_qualified: bool = False
        self._quiet_subframes: int = 0
        self._first_clap_t: float | None = None
        self._cooldown_until: float = 0.0
        self._last_event_reason: str = "ready"

    @property
    def background_noise_rms(self) -> float:
        with self._lock:
            return self._noise_floor_rms

    @property
    def last_event_reason(self) -> str:
        with self._lock:
            return self._last_event_reason

    def set_sensitivity(self, sensitivity: float) -> None:
        with self._lock:
            self.config.sensitivity = max(0.1, min(1.0, float(sensitivity)))

    def reset(self, preserve_noise_floor: bool = False) -> None:
        with self._lock:
            if not preserve_noise_floor:
                self._noise_floor_rms = 60.0
            self._in_impulse = False
            self._impulse_start_t = 0.0
            self._impulse_peak_t = 0.0
            self._impulse_peak_rms = 0.0
            self._impulse_qualified = False
            self._quiet_subframes = 0
            self._first_clap_t = None
            self._cooldown_until = 0.0

    def process_frame(self, frame_bytes: bytes, timestamp: float | None = None) -> bool:
        """
        Inspect a chunk of 16-bit mono PCM audio synchronously.
        Returns True when a verified double-clap gesture concludes.
        """
        if not frame_bytes or len(frame_bytes) < 4:
            return False

        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
        if samples.size < 16:
            return False

        sr = max(8000, int(self.config.sample_rate))
        sub_len = max(32, int(sr * 0.010))  # 10ms transient sub-windows
        sub_dur_s = sub_len / float(sr)
        base_t = time.monotonic() if timestamp is None else float(timestamp)

        with self._lock:
            sens = max(0.1, min(1.0, float(self.config.sensitivity)))
            gain = 1.6 - sens
            eff_peak = self.config.min_peak_amplitude * gain
            eff_snr = self.config.snr_threshold * (0.70 + 0.50 * (1.0 - sens))
            min_hf_ratio = self.config.min_high_freq_ratio

            triggered = False
            n_chunks = max(1, samples.size // sub_len)

            for idx in range(n_chunks):
                start = idx * sub_len
                end = start + sub_len if idx < n_chunks - 1 else samples.size
                chunk = samples[start:end]
                if chunk.size < 8:
                    continue

                t_now = base_t + idx * sub_dur_s
                peak = float(np.max(np.abs(chunk)))
                full_rms = float(np.sqrt(np.mean(chunk * chunk)))
                hp = chunk[1:] - 0.85 * chunk[:-1]
                hp_rms = float(np.sqrt(np.mean(hp * hp)))
                hp_ratio = hp_rms / max(full_rms, 1.0)

                elevated_thresh = max(self._noise_floor_rms * 2.2, 320.0)
                is_elevated = full_rms >= elevated_thresh
                is_clap_spike = (
                    peak >= eff_peak
                    and full_rms >= max(self._noise_floor_rms, 35.0) * eff_snr
                    and hp_ratio >= min_hf_ratio
                )

                if not self._in_impulse:
                    if is_clap_spike:
                        self._in_impulse = True
                        self._impulse_start_t = t_now
                        self._impulse_peak_t = t_now
                        self._impulse_peak_rms = full_rms
                        self._impulse_qualified = True
                        self._quiet_subframes = 0
                    elif not is_elevated:
                        # Smoothly track quiet room noise baseline
                        self._noise_floor_rms = min(
                            4000.0, max(15.0, 0.94 * self._noise_floor_rms + 0.06 * full_rms)
                        )
                        if (
                            self._first_clap_t is not None
                            and (t_now - self._first_clap_t) > self.config.max_gap_s
                        ):
                            self._first_clap_t = None
                else:
                    if full_rms > self._impulse_peak_rms:
                        self._impulse_peak_rms = full_rms
                        self._impulse_peak_t = t_now

                    dur_ms = (t_now - self._impulse_start_t) * 1000.0
                    decay_floor = max(
                        self._noise_floor_rms * 2.0,
                        self._impulse_peak_rms * self.config.quiet_gap_decay_ratio,
                    )

                    if full_rms <= decay_floor:
                        self._quiet_subframes += 1
                        if self._quiet_subframes >= 2 or full_rms <= self._impulse_peak_rms * 0.25:
                            was_valid = self._impulse_qualified and (
                                dur_ms <= self.config.max_impulse_duration_ms
                            )
                            imp_t = self._impulse_peak_t
                            self._in_impulse = False
                            self._quiet_subframes = 0

                            if was_valid and self._register_impulse(imp_t):
                                triggered = True
                    else:
                        self._quiet_subframes = 0
                        if dur_ms > self.config.max_impulse_duration_ms:
                            # Sustained signal (speech / music / constant tone) — reject and adapt floor
                            self._impulse_qualified = False
                            self._first_clap_t = None
                            self._noise_floor_rms = min(
                                5000.0, 0.90 * self._noise_floor_rms + 0.10 * full_rms
                            )
                            self._last_event_reason = "rejected_sustained_audio"

            return triggered

    def _register_impulse(self, impulse_time: float) -> bool:
        """Evaluate valid sharp impulse against inter-clap gap bounds and cooldown."""
        if impulse_time < self._cooldown_until:
            self._last_event_reason = "ignored_cooldown"
            return False

        if self._first_clap_t is None:
            self._first_clap_t = impulse_time
            self._last_event_reason = "first_clap_candidate"
            return False

        gap = impulse_time - self._first_clap_t
        if gap < self.config.min_gap_s:
            # Room reverb bounce or sub-clap artifact; keep waiting
            self._last_event_reason = f"ignored_too_close ({gap:.3f}s)"
            return False

        if gap > self.config.max_gap_s:
            # Previous clap timed out; promote this impulse to clap #1
            self._first_clap_t = impulse_time
            self._last_event_reason = "first_clap_window_restarted"
            return False

        # Confirmed valid double-clap!
        self._first_clap_t = None
        self._cooldown_until = impulse_time + self.config.cooldown_s
        self._last_event_reason = f"double_clap_confirmed (gap={gap:.3f}s)"
        return True


PresenceCallback = Callable[[PresenceState, PresenceState, str], None]


class VoicePresenceService:
    """
    Coordinates hands-free Voice Presence lifecycle across `DORMANT`, `ARMED`,
    `PROCESSING`, `SPEAKING`, `DISABLED`, and `ERROR`.

    Thread-safe, deterministic, and independent of any LLM or cloud service.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        clap_enabled: bool = True,
        clap_sensitivity: float = 0.6,
        active_timeout_s: float = 30.0,
        activation_phrase: str = "hey iris",
        post_speech_guard_s: float = 0.75,
        on_presence_change: PresenceCallback | None = None,
        detector: DoubleClapDetector | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._state: PresenceState = PresenceState.DISABLED
        self._last_reason: str = "initialized"
        self._clap_enabled = bool(clap_enabled)
        self._active_timeout_s = max(5.0, float(active_timeout_s))
        self._activation_phrase = activation_phrase.strip()
        self._post_speech_guard_s = max(0.0, float(post_speech_guard_s))
        self._guard_until: float = 0.0
        self._external_audio_playing: bool = False  # Future Spotify/media playback awareness hook

        self._detector = detector or DoubleClapDetector(
            ClapDetectorConfig(sample_rate=sample_rate, sensitivity=clap_sensitivity)
        )
        self._inactivity_timer: threading.Timer | None = None
        self._callbacks: list[PresenceCallback] = []
        if on_presence_change:
            self._callbacks.append(on_presence_change)

    # ── Read-Only Properties ──────────────────────────────────────────────────

    @property
    def state(self) -> PresenceState:
        with self._lock:
            return self._state

    @property
    def last_reason(self) -> str:
        with self._lock:
            return self._last_reason

    @property
    def clap_enabled(self) -> bool:
        with self._lock:
            return self._clap_enabled

    @property
    def clap_sensitivity(self) -> float:
        return self._detector.config.sensitivity

    @property
    def active_timeout_s(self) -> float:
        with self._lock:
            return self._active_timeout_s

    @property
    def activation_phrase(self) -> str:
        with self._lock:
            return self._activation_phrase

    @property
    def external_audio_playing(self) -> bool:
        with self._lock:
            return self._external_audio_playing

    @property
    def is_stt_allowed(self) -> bool:
        """True ONLY when active speech capture is permitted (ARMED)."""
        with self._lock:
            return self._state == PresenceState.ARMED

    @property
    def is_mic_capture_expected(self) -> bool:
        """True in all states except DISABLED and ERROR."""
        with self._lock:
            return self._state not in {PresenceState.DISABLED, PresenceState.ERROR}

    # ── Listener & Configuration Setters ──────────────────────────────────────

    def add_listener(self, callback: PresenceCallback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def set_clap_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._clap_enabled = bool(enabled)

    def set_sensitivity(self, sensitivity: float) -> None:
        self._detector.set_sensitivity(sensitivity)

    def set_active_timeout(self, timeout_s: float) -> None:
        restart = False
        with self._lock:
            self._active_timeout_s = max(5.0, float(timeout_s))
            if self._state == PresenceState.ARMED:
                restart = True
        if restart:
            self._schedule_inactivity_timer()

    def set_activation_phrase(self, phrase: str) -> None:
        with self._lock:
            self._activation_phrase = phrase.strip()

    def set_external_audio_playing(self, playing: bool) -> None:
        """Hook for future media players (e.g. Spotify) to tighten clap suppression."""
        with self._lock:
            self._external_audio_playing = bool(playing)
            if playing:
                self._detector.reset(preserve_noise_floor=True)

    # ── State Transitions (Deterministic & Thread-Safe) ───────────────────────

    def _transition(self, target: PresenceState, reason: str) -> bool:
        callbacks: list[PresenceCallback] = []
        old_state = self._state
        with self._lock:
            old_state = self._state
            if old_state == target and self._last_reason == reason:
                return False
            self._state = target
            self._last_reason = reason
            callbacks = list(self._callbacks)
            self._cancel_timer_locked()

        if target == PresenceState.ARMED:
            self._schedule_inactivity_timer()

        logger.info("[Presence] %s -> %s (%s)", old_state.value, target.value, reason)
        for cb in callbacks:
            try:
                cb(old_state, target, reason)
            except Exception as exc:
                logger.exception("Error in VoicePresenceService listener: %s", exc)
        return True

    def enable_dormant(self, reason: str = "dormant_standby") -> None:
        """Enter DORMANT standby (local clap detection only, zero cloud/STT)."""
        self._detector.reset(preserve_noise_floor=True)
        self._transition(PresenceState.DORMANT, reason)

    def activate(self, source: str = "manual") -> None:
        """Transition into ARMED active command listening."""
        self._detector.reset(preserve_noise_floor=True)
        self._transition(PresenceState.ARMED, f"activated:{source}")

    def deactivate(self, reason: str = "manual") -> None:
        """Return cleanly to DORMANT standby while keeping local clap listening ready."""
        self._detector.reset(preserve_noise_floor=True)
        self._transition(PresenceState.DORMANT, f"deactivated:{reason}")

    def disable(self, reason: str = "mic_disabled") -> None:
        """Disable microphone capture completely (stops clap detection & active listening)."""
        self._detector.reset(preserve_noise_floor=True)
        self._transition(PresenceState.DISABLED, reason)

    def set_error(self, reason: str = "hardware_error") -> None:
        self._detector.reset(preserve_noise_floor=True)
        self._transition(PresenceState.ERROR, reason)

    def on_user_speech_onset(self) -> None:
        """Suspend inactivity timeout while user is actively speaking a command."""
        with self._lock:
            if self._state == PresenceState.ARMED:
                self._cancel_timer_locked()

    def on_processing_started(self) -> None:
        with self._lock:
            if self._state not in {PresenceState.ARMED, PresenceState.PROCESSING}:
                return
        self._transition(PresenceState.PROCESSING, "processing_command")

    def on_speaking_started(self) -> None:
        with self._lock:
            if self._state in {PresenceState.DISABLED, PresenceState.ERROR, PresenceState.DORMANT}:
                return
            self._detector.reset(preserve_noise_floor=True)
        self._transition(PresenceState.SPEAKING, "tts_speaking")

    def on_speaking_finished(self, stay_armed: bool = True) -> None:
        """
        Resume ARMED multi-turn listening after TTS playback completes unless
        the user deactivated IRIS during/before playback.
        """
        with self._lock:
            self._guard_until = time.monotonic() + self._post_speech_guard_s
            self._detector.reset(preserve_noise_floor=True)
            if self._state in {PresenceState.DISABLED, PresenceState.ERROR, PresenceState.DORMANT}:
                return
        if stay_armed:
            self._transition(PresenceState.ARMED, "ready_for_followup")
        else:
            self._transition(PresenceState.DORMANT, "turn_completed_dormant")

    def recover_from_turn(self, return_to_dormant: bool = False) -> None:
        """Recover safely after turn error, empty transcript, or echo discard."""
        with self._lock:
            self._detector.reset(preserve_noise_floor=True)
            if self._state in {PresenceState.DISABLED, PresenceState.ERROR, PresenceState.DORMANT}:
                return
        target = PresenceState.DORMANT if return_to_dormant else PresenceState.ARMED
        self._transition(target, "recovered")

    # ── Audio Frame Processing & Acoustic Feedback Gating ─────────────────────

    def process_audio_frame(
        self,
        frame_bytes: bytes,
        audio_state: AudioState | str = "idle",
        vad_active: bool = False,
        timestamp: float | None = None,
    ) -> str | None:
        """
        Evaluate incoming local audio frame for double-clap transitions.

        Strictly gates out clap detection during `PROCESSING`, `SPEAKING`, `COOLDOWN`,
        active speech (`vad_active`), post-TTS acoustic decay guard, and external
        media playback.

        Returns `"activated"`, `"deactivated"`, or `None`.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        audio_state_val = (
            str(audio_state.value).lower()
            if hasattr(audio_state, "value")
            else str(audio_state).lower()
        )

        with self._lock:
            state = self._state
            clap_on = self._clap_enabled
            guarded = (now < self._guard_until) or self._external_audio_playing

        if not clap_on or guarded:
            return None
        if state in {PresenceState.DISABLED, PresenceState.ERROR, PresenceState.PROCESSING, PresenceState.SPEAKING}:
            return None
        if audio_state_val != "idle":
            return None
        if vad_active:
            return None

        if self._detector.process_frame(frame_bytes, timestamp=now):
            if state == PresenceState.DORMANT:
                self.activate(source="double_clap")
                return "activated"
            if state == PresenceState.ARMED:
                self.deactivate(reason="double_clap")
                return "deactivated"
        return None

    def try_handle_text_presence_command(self, text: str) -> str | None:
        """
        Handle deterministic activation/deactivation text commands without LLM.
        Returns action `"activated"`, `"deactivated"`, or `None`.
        """
        if is_deactivation_phrase(text):
            if self.state != PresenceState.DISABLED:
                self.deactivate(reason="voice_command")
            return "deactivated"

        if self.state == PresenceState.DORMANT and is_activation_phrase(text, self.activation_phrase):
            self.activate(source="activation_phrase")
            return "activated"

        return None

    # ── Inactivity Timer Helpers ──────────────────────────────────────────────

    def _cancel_timer_locked(self) -> None:
        if self._inactivity_timer is not None:
            self._inactivity_timer.cancel()
            self._inactivity_timer = None

    def _schedule_inactivity_timer(self) -> None:
        with self._lock:
            self._cancel_timer_locked()
            if self._state != PresenceState.ARMED:
                return
            timeout = self._active_timeout_s
            timer = threading.Timer(timeout, self._on_inactivity_timeout)
            timer.daemon = True
            timer.name = "iris-presence-timeout"
            self._inactivity_timer = timer
            timer.start()

    def _on_inactivity_timeout(self) -> None:
        with self._lock:
            if self._state != PresenceState.ARMED:
                return
        self.deactivate(reason="inactivity_timeout")

    def shutdown(self) -> None:
        with self._lock:
            self._cancel_timer_locked()
            self._detector.reset(preserve_noise_floor=False)
            self._state = PresenceState.DISABLED
            self._last_reason = "shutdown"
