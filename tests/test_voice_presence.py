"""
tests/test_voice_presence.py
----------------------------
Comprehensive unit and integration tests for IRIS Phase 3 Voice Presence System:
- Local DoubleClapDetector DSP (synthetic 16kHz PCM transients, noise, thuds, timing, cooldown)
- VoicePresenceService state machine & deterministic phrases
- Strict standby privacy invariant (zero STT, zero LLM, zero VAD, zero retained audio in DORMANT)
- Multi-turn ARMED conversation, TTS echo suppression, voice deactivation, manual overrides
- Desktop VoiceControlBar, SettingsPage, IRISMainWindow shortcut & presence UI honesty
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iris.audio.presence import (
    ClapDetectorConfig,
    DoubleClapDetector,
    PresenceState,
    VoicePresenceService,
    is_activation_phrase,
    is_deactivation_phrase,
)
from iris.core.assistant import IRISAssistant
from iris.core.config import IRISConfig
from iris.core.policy import DEFAULT_LLM_POLICY
from iris.core.state import AudioState
from iris.desktop.backend import IRISDesktopBackend
from iris.desktop.main_window import IRISMainWindow
from iris_voice import build_arg_parser

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ── Synthetic 16kHz Mono Int16 Audio Helpers ─────────────────────────────────


def _silence_pcm(duration_ms: int, sr: int = 16000) -> bytes:
    n = int(sr * duration_ms / 1000)
    return np.zeros(n, dtype=np.int16).tobytes()


def _clap_impulse_pcm(
    duration_ms: int = 35,
    peak: float = 18000.0,
    sr: int = 16000,
    seed: int = 42,
) -> bytes:
    """Synthesize a crisp broadband hand-clap transient with rapid exponential decay."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration_ms / 1000)
    t = np.arange(n, dtype=np.float32) / float(sr)
    # Sharp attack (~2ms) + fast 7ms exponential decay + high-frequency snap noise
    envelope = np.exp(-t / 0.007)
    envelope[: max(1, int(0.0015 * sr))] = np.linspace(0.2, 1.0, max(1, int(0.0015 * sr)))
    noise = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    snap = np.sin(2.0 * np.pi * 2400.0 * t).astype(np.float32)
    wave = (0.75 * noise + 0.25 * snap) * envelope
    wave = wave / max(1e-6, float(np.max(np.abs(wave)))) * peak
    return np.clip(wave, -32767, 32767).astype(np.int16).tobytes()


def _low_freq_thud_pcm(duration_ms: int = 45, peak: float = 20000.0, sr: int = 16000) -> bytes:
    """Synthesize an 80Hz low-frequency desk bump or thump."""
    n = int(sr * duration_ms / 1000)
    t = np.arange(n, dtype=np.float32) / float(sr)
    wave = peak * np.sin(2.0 * np.pi * 80.0 * t) * np.exp(-t / 0.015)
    return np.clip(wave, -32767, 32767).astype(np.int16).tobytes()


def _sustained_speech_or_tone_pcm(
    duration_ms: int = 450,
    amplitude: float = 12000.0,
    freq_hz: float = 1800.0,
    sr: int = 16000,
) -> bytes:
    """Synthesize sustained harmonic/speech-like audio exceeding max_impulse_duration_ms."""
    n = int(sr * duration_ms / 1000)
    t = np.arange(n, dtype=np.float32) / float(sr)
    wave = amplitude * (0.7 * np.sin(2.0 * np.pi * freq_hz * t) + 0.3 * np.sin(2.0 * np.pi * 420.0 * t))
    return np.clip(wave, -32767, 32767).astype(np.int16).tobytes()


def _feed_pcm_chunks(
    detector: DoubleClapDetector,
    pcm: bytes,
    start_t: float = 10.0,
    chunk_ms: int = 30,
    sr: int = 16000,
) -> tuple[bool, float]:
    """Feed raw PCM in 30ms frames (matching SoundDevice capture) with deterministic timestamps."""
    chunk_bytes = int(sr * chunk_ms / 1000) * 2
    t = start_t
    triggered = False
    for offset in range(0, len(pcm), chunk_bytes):
        piece = pcm[offset : offset + chunk_bytes]
        if detector.process_frame(piece, timestamp=t):
            triggered = True
        t += len(piece) / float(sr * 2)
    return triggered, t


def _make_offline_config(tmp_path: Path, **overrides) -> IRISConfig:
    base = dict(
        openrouter_api_key="",
        google_api_key="",
        elevenlabs_api_key="",
        openai_api_key="",
        anthropic_api_key="",
        llm_provider="openrouter",
        stt_provider="elevenlabs",
        tts_provider="macos",
        sample_rate=16000,
        archive_dir=tmp_path / "memory" / "sessions",
        vault_dir=tmp_path / "vault",
        knowledge_index_path=tmp_path / "memory" / "index" / "knowledge.db",
        active_listening_timeout_s=30.0,
    )
    base.update(overrides)
    return IRISConfig(**base)


# ── DoubleClapDetector DSP Tests ─────────────────────────────────────────────


def test_double_clap_detector_triggers_on_crisp_double_clap() -> None:
    detector = DoubleClapDetector(ClapDetectorConfig(sensitivity=0.6))
    waveform = (
        _silence_pcm(120)
        + _clap_impulse_pcm(35, peak=16000, seed=1)
        + _silence_pcm(320)
        + _clap_impulse_pcm(35, peak=16000, seed=2)
        + _silence_pcm(120)
    )
    triggered, _ = _feed_pcm_chunks(detector, waveform, start_t=10.0)
    assert triggered is True
    assert "double_clap_confirmed" in detector.last_event_reason
    # Privacy check: detector holds no raw PCM arrays or waveform history attributes
    for val in vars(detector).values():
        assert not isinstance(val, (bytes, bytearray, np.ndarray, list))


def test_double_clap_sensitivity_tuning() -> None:
    quieter_claps = (
        _silence_pcm(90)
        + _clap_impulse_pcm(30, peak=3100, seed=3)
        + _silence_pcm(280)
        + _clap_impulse_pcm(30, peak=3100, seed=4)
        + _silence_pcm(90)
    )
    strict = DoubleClapDetector(ClapDetectorConfig(sensitivity=0.1))
    assert _feed_pcm_chunks(strict, quieter_claps, start_t=5.0)[0] is False

    sensitive = DoubleClapDetector(ClapDetectorConfig(sensitivity=0.95))
    assert _feed_pcm_chunks(sensitive, quieter_claps, start_t=5.0)[0] is True


def test_double_clap_rejects_echo_too_close_and_window_expired() -> None:
    detector = DoubleClapDetector()
    # Too close (<140ms room echo bounce at 60ms)
    echo_pair = (
        _silence_pcm(90)
        + _clap_impulse_pcm(25, peak=18000, seed=5)
        + _silence_pcm(55)
        + _clap_impulse_pcm(25, peak=14000, seed=6)
        + _silence_pcm(120)
    )
    assert _feed_pcm_chunks(detector, echo_pair, start_t=10.0)[0] is False

    # Window expired (>850ms between claps): 2nd clap resets candidate; 3rd completes pair
    detector.reset()
    slow_then_pair = (
        _silence_pcm(90)
        + _clap_impulse_pcm(30, peak=18000, seed=7)
        + _silence_pcm(960)
        + _clap_impulse_pcm(30, peak=18000, seed=8)
        + _silence_pcm(300)
        + _clap_impulse_pcm(30, peak=18000, seed=9)
        + _silence_pcm(90)
    )
    triggered, _ = _feed_pcm_chunks(detector, slow_then_pair, start_t=20.0)
    assert triggered is True


def test_double_clap_rejects_thuds_sustained_audio_and_respects_cooldown() -> None:
    detector = DoubleClapDetector()

    # 1. Low-frequency desk thuds (80 Hz) must be rejected cleanly
    thuds = (
        _silence_pcm(90)
        + _low_freq_thud_pcm(45, peak=24000)
        + _silence_pcm(300)
        + _low_freq_thud_pcm(45, peak=24000)
        + _silence_pcm(90)
    )
    assert _feed_pcm_chunks(detector, thuds, start_t=5.0)[0] is False

    # 2. Sustained speech / tone (>135ms) must be rejected
    sustained = (
        _silence_pcm(90)
        + _sustained_speech_or_tone_pcm(420, amplitude=15000)
        + _silence_pcm(260)
        + _sustained_speech_or_tone_pcm(420, amplitude=15000)
        + _silence_pcm(120)
    )
    assert _feed_pcm_chunks(detector, sustained, start_t=10.0)[0] is False
    assert detector.last_event_reason == "rejected_sustained_audio"

    # 3. Cooldown prevents immediate re-trigger inside 1.5s refractory period
    detector.reset()
    double_pair = (
        _silence_pcm(90)
        + _clap_impulse_pcm(30, peak=18000, seed=10)
        + _silence_pcm(300)
        + _clap_impulse_pcm(30, peak=18000, seed=11)
        + _silence_pcm(120)
    )
    ok1, end_t = _feed_pcm_chunks(detector, double_pair, start_t=30.0)
    assert ok1 is True
    # Immediate replay inside cooldown (~0.6s later) is ignored
    ok2, _ = _feed_pcm_chunks(detector, double_pair, start_t=end_t + 0.1)
    assert ok2 is False
    assert detector.last_event_reason == "ignored_cooldown"


def test_deterministic_activation_and_deactivation_phrases() -> None:
    assert is_deactivation_phrase("Stop listening.") is True
    assert is_deactivation_phrase("Go to standby") is True
    assert is_deactivation_phrase("Deactivate listening please.") is True
    assert is_deactivation_phrase("That's all.") is True
    assert is_deactivation_phrase("Hey IRIS, stop listening now") is False
    assert is_deactivation_phrase("Explain what a standby circuit is") is False

    assert is_activation_phrase("Hey IRIS", "hey iris") is True
    assert is_activation_phrase("Wake up hey iris!", "hey iris") is True
    assert is_activation_phrase("Hey IRIS", "") is False


# ── VoicePresenceService Lifecycle Tests ─────────────────────────────────────


def test_presence_service_gates_external_audio_and_inactivity_timeout() -> None:
    events: list[tuple[str, str, str]] = []
    svc = VoicePresenceService(
        clap_enabled=True,
        active_timeout_s=10.0,
        on_presence_change=lambda old, new, reason: events.append((old.value, new.value, reason)),
    )
    svc.enable_dormant("startup")
    assert svc.state == PresenceState.DORMANT
    assert svc.is_stt_allowed is False

    double_clap = (
        _silence_pcm(90)
        + _clap_impulse_pcm(30, peak=18000, seed=21)
        + _silence_pcm(300)
        + _clap_impulse_pcm(30, peak=18000, seed=22)
        + _silence_pcm(90)
    )

    # External audio hook (e.g. future Spotify playback) suppresses clap activation safely
    svc.set_external_audio_playing(True)
    chunk_bytes = int(16000 * 0.03) * 2
    t = 100.0
    activated = False
    for i in range(0, len(double_clap), chunk_bytes):
        if svc.process_audio_frame(double_clap[i : i + chunk_bytes], timestamp=t) == "activated":
            activated = True
        t += 0.03
    assert activated is False
    assert svc.state == PresenceState.DORMANT

    # Disable external audio -> double clap arms presence cleanly
    svc.set_external_audio_playing(False)
    for i in range(0, len(double_clap), chunk_bytes):
        svc.process_audio_frame(double_clap[i : i + chunk_bytes], timestamp=t + 2.0)
        t += 0.03
    assert svc.state == PresenceState.ARMED
    assert svc.is_stt_allowed is True

    # Simulate inactivity timeout callback -> cleanly drops back to DORMANT
    svc._on_inactivity_timeout()
    assert svc.state == PresenceState.DORMANT
    svc.shutdown()


# ── Assistant Privacy, Multi-Turn, Echo Gate, Override Integration ───────────


def test_dormant_standby_enforces_zero_stt_zero_llm_and_double_clap_toggles(
    tmp_path: Path,
) -> None:
    cfg = _make_offline_config(tmp_path, cooldown_ms=0)
    mock_model = MagicMock()
    mock_model.generate_response.return_value = "First reply acknowledged."
    mock_tts = MagicMock()
    mock_tts.speak.side_effect = lambda _text, on_done=None: on_done() if on_done else None
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "What is the time?"

    with patch("iris.core.assistant.AudioCapture") as MockCapture:
        cap_instance = MockCapture.return_value
        cap_instance.is_running = True
        cap_instance.is_muted = False

        assistant = IRISAssistant(
            config=cfg,
            model_adapter=mock_model,
            tts_adapter=mock_tts,
            stt_adapter=mock_stt,
        )
        assistant.vad = MagicMock()
        assistant.vad.is_speaking = False
        assistant.vad.process_frame.return_value = None

        assistant.start(capture_audio=True, start_dormant=True, initial_context_refresh=False)
        assert assistant.presence.state == PresenceState.DORMANT
        cap_instance.start.assert_called_once()

        base_t = time.monotonic() + 100.0
        # 1. Feed sustained speech & ambient noise while DORMANT:
        # Strict privacy guarantee: ZERO STT, ZERO LLM, ZERO VAD calls!
        speech_chunk = _sustained_speech_or_tone_pcm(120, amplitude=9000)
        t_speech = iter([base_t + idx * 0.12 for idx in range(20)])
        with patch("iris.audio.presence.time.monotonic", side_effect=lambda: next(t_speech)):
            for _ in range(5):
                assistant._on_audio_frame(speech_chunk)
            # Let noise/impulse decay across 150ms of silence before testing double clap
            assistant._on_audio_frame(_silence_pcm(150))

        assistant.vad.process_frame.assert_not_called()
        mock_stt.transcribe.assert_not_called()
        mock_model.generate_response.assert_not_called()
        assert assistant.presence.state == PresenceState.DORMANT
        assert list(cfg.vault_dir.rglob("*.md")) == []

        # 2. Perform double clap while DORMANT -> arms active listening without calling LLM
        double_clap = (
            _silence_pcm(90)
            + _clap_impulse_pcm(30, peak=18000, seed=31)
            + _silence_pcm(300)
            + _clap_impulse_pcm(30, peak=18000, seed=32)
            + _silence_pcm(90)
        )
        t_vals = iter([base_t + 50.0 + idx * 0.03 for idx in range(200)])
        with patch("iris.audio.presence.time.monotonic", side_effect=lambda: next(t_vals)):
            chunk_bytes = int(16000 * 0.03) * 2
            for i in range(0, len(double_clap), chunk_bytes):
                assistant._on_audio_frame(double_clap[i : i + chunk_bytes])

        assert assistant.presence.state == PresenceState.ARMED
        mock_model.generate_response.assert_not_called()

        def _run_simulated_voice_turn() -> None:
            assistant.state_machine.transition_to(AudioState.LISTENING)
            tid = assistant.state_machine.current_turn_id
            assistant.state_machine.transition_to(AudioState.THINKING, expected_turn_id=tid)
            assistant.presence.on_processing_started()
            assistant._process_utterance(b"\x01\x02" * 500, turn_id=tid)
            # Wait briefly for cooldown_ms=0 timer thread to return state_machine to IDLE
            deadline = time.monotonic() + 1.0
            while assistant.state_machine.state != AudioState.IDLE and time.monotonic() < deadline:
                time.sleep(0.01)

        # 3. Issue two consecutive voice turns without repeating wake word
        _run_simulated_voice_turn()
        assert assistant.presence.state == PresenceState.ARMED
        _run_simulated_voice_turn()
        assert assistant.presence.state == PresenceState.ARMED
        assert mock_stt.transcribe.call_count == 2
        assert mock_model.generate_response.call_count == 2

        # 4. Deactivation phrase returns deterministically to DORMANT without invoking LLM
        mock_stt.transcribe.return_value = "Stop listening."
        _run_simulated_voice_turn()
        assert assistant.presence.state == PresenceState.DORMANT
        assert mock_model.generate_response.call_count == 2

        # 5. Re-arm, test TTS speaking suppression & post-speech guard, then double-clap deactivate
        assistant.activate_presence(source="manual")
        assert assistant.presence.state == PresenceState.ARMED
        assistant.presence.on_speaking_started()
        assert assistant.presence.state == PresenceState.SPEAKING

        # Double-clap fed while SPEAKING must be suppressed
        t_vals_speak = iter([base_t + 90.0 + idx * 0.03 for idx in range(200)])
        with patch("iris.audio.presence.time.monotonic", side_effect=lambda: next(t_vals_speak)):
            for i in range(0, len(double_clap), chunk_bytes):
                assistant._on_audio_frame(double_clap[i : i + chunk_bytes])
        assert assistant.presence.state == PresenceState.SPEAKING

        # Finish speaking (with short post-speech guard disabled for instant test of armed double-clap)
        assistant.presence.set_active_timeout(30.0)
        assistant.presence._post_speech_guard_s = 0.0
        assistant.presence.on_speaking_finished(stay_armed=True)
        assert assistant.presence.state == PresenceState.ARMED

        # Double clap while ARMED returns to DORMANT
        assistant.presence._detector.reset()
        t_vals_arm = iter([base_t + 140.0 + idx * 0.03 for idx in range(200)])
        with patch("iris.audio.presence.time.monotonic", side_effect=lambda: next(t_vals_arm)):
            for i in range(0, len(double_clap), chunk_bytes):
                assistant._on_audio_frame(double_clap[i : i + chunk_bytes])
        assert assistant.presence.state == PresenceState.DORMANT

        # 6. Disable microphone stops capture completely
        assistant.disable_microphone()
        assert assistant.presence.state == PresenceState.DISABLED
        cap_instance.stop.assert_called()
        assistant.stop()


def test_text_turn_activation_deactivation_and_cli_dormant_flag(tmp_path: Path) -> None:
    cfg = _make_offline_config(tmp_path)
    mock_model = MagicMock()
    mock_tts = MagicMock()
    mock_stt = MagicMock()
    with patch("iris.core.assistant.AudioCapture") as MockCapture:
        MockCapture.return_value.is_running = False
        MockCapture.return_value.is_muted = False
        assistant = IRISAssistant(
            config=cfg,
            model_adapter=mock_model,
            tts_adapter=mock_tts,
            stt_adapter=mock_stt,
        )
        assistant.enable_dormant_standby()
        assert assistant.presence.state == PresenceState.DORMANT

        # Activation phrase works without calling LLM
        reply_arm = assistant.process_text_turn("Hey IRIS", speak=False)
        assert assistant.presence.state == PresenceState.ARMED
        assert "listening" in reply_arm.lower()

        # Deactivation phrase works without calling LLM
        reply_sleep = assistant.process_text_turn("Go to standby", speak=False)
        assert assistant.presence.state == PresenceState.DORMANT
        assert "standby" in reply_sleep.lower()
        mock_model.generate_response.assert_not_called()
        assistant.stop()

    # Verify CLI --dormant flag and LLM policy capabilities
    args = build_arg_parser().parse_args(["--dormant"])
    assert args.dormant is True
    assert DEFAULT_LLM_POLICY.requires_llm("double_clap_detector") is False
    assert DEFAULT_LLM_POLICY.requires_llm("voice_presence_state") is False


# ── Desktop Presence UI Honesty & Settings Integration ───────────────────────


def test_desktop_presence_ui_states_and_settings_controls(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    cfg = _make_offline_config(tmp_path, clap_sensitivity=0.6, active_listening_timeout_s=25.0)
    mock_model = MagicMock()
    mock_tts = MagicMock()
    mock_stt = MagicMock()

    with patch("iris.core.assistant.AudioCapture") as MockCapture:
        cap = MockCapture.return_value
        cap.is_running = False
        cap.is_muted = False

        def _fake_start(*_args, **_kwargs) -> None:
            cap.is_running = True

        def _fake_stop() -> None:
            cap.is_running = False

        cap.start.side_effect = _fake_start
        cap.stop.side_effect = _fake_stop

        assistant = IRISAssistant(
            config=cfg,
            model_adapter=mock_model,
            tts_adapter=mock_tts,
            stt_adapter=mock_stt,
        )
        backend = IRISDesktopBackend(config=cfg, assistant=assistant)
        window = IRISMainWindow(backend=backend, auto_start_backend=False)
        backend.start(refresh_context_async=False)
        window.refresh_all_vault_views()
        try:
            # 1. Startup keeps presence DISABLED and Listening strictly Off
            assert "Disabled" in window.voice_bar.presence_pill.text()
            assert "Off" in window.voice_bar.listening_pill.text()

            # 2. DORMANT standby keeps Mic Active (Clap Standby) but Listening strictly Off
            assistant.enable_dormant_standby()
            app.processEvents()
            assert "Dormant" in window.voice_bar.presence_pill.text()
            assert "Clap Only" in window.voice_bar.listening_pill.text()
            assert "Clap Standby" in window.voice_bar.mic_status_pill.text()

            # 3. ARMED enables active listening pill honestly
            assistant.activate_presence(source="manual")
            app.processEvents()
            assert "Armed" in window.voice_bar.presence_pill.text()
            assert "Ready" in window.voice_bar.listening_pill.text()
            assert "Pause to Standby" in window.voice_bar.arm_btn.text()

            # 4. Settings page controls adjust presence settings cleanly
            window.navigate_to("settings")
            window.settings_page.clap_enabled_checkbox.setChecked(False)
            assert assistant.presence.clap_enabled is False
            window.settings_page.sensitivity_spin.setValue(0.85)
            assert abs(assistant.presence.clap_sensitivity - 0.85) < 1e-6
            window.settings_page.timeout_spin.setValue(45.0)
            assert abs(assistant.presence.active_timeout_s - 45.0) < 1e-6
            window.settings_page.activation_phrase_input.setText("computer wake")
            window.settings_page.apply_phrase_btn.click()
            assert assistant.presence.activation_phrase == "computer wake"
        finally:
            window.close()
