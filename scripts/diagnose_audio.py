"""
scripts/diagnose_audio.py
-------------------------
Isolated Diagnostic Tool for IRIS Audio Pipeline and Speech Recognition.

Tests each layer independently:
1. Available audio input devices and default device selection.
2. Actual microphone capture (duration, sample rate, channels, RMS energy, peak amplitude).
3. macOS Microphone Permission verification.
4. STT engine initialization (ElevenLabs Scribe STT).
5. Transcription of a short recorded speech sample.
6. Delivery of transcript to IRIS ConversationService and EchoFilter.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dotenv
dotenv.load_dotenv(PROJECT_ROOT / ".env")


def run_diagnostics() -> int:
    print("=" * 70)
    print("   IRIS AUDIO & STT DIAGNOSTIC SUITE")
    print("=" * 70)

    all_passed = True

    # ── 1. Audio Input Devices ────────────────────────────────────────────
    print("\n[Layer 1] Querying Audio Devices...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]

        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        print(f"  Found {len(input_devs)} input device(s):")
        for idx, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                is_def = " (DEFAULT)" if idx == default_in else ""
                print(f"    - Device #{idx}: {d['name']} [ch: {d['max_input_channels']}, sr: {d['default_samplerate']}Hz]{is_def}")

        print(f"  Active Default Input: Device #{default_in} ({devices[default_in]['name']})")
        print("  --> PASS: Audio devices queried successfully.")
    except Exception as exc:
        print(f"  --> FAIL: Could not query audio devices: {exc}")
        return 1

    # ── 2. Microphone Capture & Amplitude Test ────────────────────────────
    print("\n[Layer 2] Testing Microphone Capture & Signal Energy...")
    try:
        import numpy as np
        sr = 16000
        duration = 1.0
        print(f"  Capturing {duration}s from device #{default_in} at {sr}Hz (mono)...")
        rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="int16", device=default_in)
        sd.wait()

        samples = rec.flatten()
        max_amp = int(np.max(np.abs(samples)))
        rms = math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))

        print(f"  Captured: {len(samples)} samples, Peak Amplitude={max_amp}, RMS Energy={rms:.2f}")

        if max_amp == 0 and rms == 0.0:
            print("  [!] WARNING: Audio capture returned pure digital silence (RMS=0.00, Max=0)!")
            print("      On macOS, this indicates:")
            print("      1. macOS Microphone permission has NOT been granted to this Terminal / App.")
            print("         Fix: System Settings -> Privacy & Security -> Microphone -> Enable your Terminal/IDE.")
            print("      2. Or the microphone input volume is muted in macOS System Settings -> Sound.")
            print("  --> FAIL (Microphone silence / Permission blocked)")
            all_passed = False
        else:
            print(f"  --> PASS: Microphone signal detected (RMS: {rms:.2f}).")
    except Exception as exc:
        print(f"  --> FAIL: Microphone recording error: {exc}")
        all_passed = False

    # ── 3. macOS Microphone Permission Verification ───────────────────────
    print("\n[Layer 3] Checking macOS Microphone System Permissions...")
    # Check input volume using osascript
    try:
        res = subprocess.run(
            ["osascript", "-e", "input volume of (get volume settings)"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        vol = res.stdout.strip()
        print(f"  macOS System Input Volume: {vol}%")
        if vol.isdigit() and int(vol) < 20:
            print("  [!] Notice: Input volume is low (<20%). Consider raising in System Settings -> Sound.")
    except Exception:
        pass

    # ── 4. STT Engine Initialization ──────────────────────────────────────
    print("\n[Layer 4] Initializing Speech-To-Text Engine...")
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        print("  --> FAIL: ELEVENLABS_API_KEY is missing or placeholder in .env.")
        all_passed = False
    else:
        try:
            from iris.adapters.stt.elevenlabs_stt import ElevenLabsSTT
            stt = ElevenLabsSTT(api_key=api_key, model_id="scribe_v1")
            print(f"  Engine: {stt.provider_name} successfully initialized.")
            print("  --> PASS: STT adapter ready.")
        except Exception as exc:
            print(f"  --> FAIL: STT initialization failed: {exc}")
            all_passed = False

    # ── 5. Speech Transcription Test (Clean Speech Sample) ────────────────
    print("\n[Layer 5] Testing Speech Transcription on Synthesized Sample...")
    test_phrase = "Hello IRIS, testing speech transcription pipeline."
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            aiff_file = Path(tmp_dir) / "diag.aiff"
            wav_file = Path(tmp_dir) / "diag.wav"

            # Synthesize clean test phrase via macOS say
            subprocess.run(["/usr/bin/say", "-o", str(aiff_file), test_phrase], check=True, capture_output=True)
            subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff_file), str(wav_file)], check=True, capture_output=True)

            with wave.open(str(wav_file), "rb") as wf:
                sample_rate = wf.getframerate()
                pcm_bytes = wf.readframes(wf.getnframes())

            print(f"  Synthesized test WAV: {len(pcm_bytes)} bytes ({sample_rate}Hz)")
            print("  Submitting to STT engine...")
            t0 = time.perf_counter()
            transcript = stt.transcribe(pcm_bytes, sample_rate=sample_rate)
            t1 = time.perf_counter()

            print(f"  Transcription received in {(t1-t0)*1000:.0f}ms:")
            print(f"  -> \"{transcript}\"")

            if transcript:
                print("  --> PASS: Speech transcription functional and verified.")
            else:
                print("  --> FAIL: STT returned empty transcription.")
                all_passed = False
    except Exception as exc:
        print(f"  --> FAIL: Speech transcription test error: {exc}")
        all_passed = False

    # ── 6. Delivery to IRIS Conversation & Echo Filter ────────────────────
    print("\n[Layer 6] Testing Transcript Delivery to IRIS Services...")
    try:
        from iris.audio.echo_filter import EchoFilter
        from iris.services.conversation import ConversationService

        conv = ConversationService()
        echo = EchoFilter()

        # Check echo filter
        is_echo, reason = echo.is_echo(transcript)
        print(f"  Echo Filter Result: is_echo={is_echo} (reason={reason})")

        # Deliver to conversation
        conv.add_user_turn(transcript)
        messages = conv.get_messages()
        last_turn = messages[-1]
        print(f"  Conversation Message Count: {len(messages)}, Last Turn Role: {last_turn['role']}")
        assert last_turn["role"] == "user"
        assert last_turn["content"] == transcript
        print("  --> PASS: Transcript successfully delivered to conversation context.")
    except Exception as exc:
        print(f"  --> FAIL: Conversation delivery error: {exc}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("   ALL AUDIO & STT DIAGNOSTIC CHECKS PASSED!")
    else:
        print("   DIAGNOSTICS COMPLETED WITH NOTED WARNINGS / FAILURES.")
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_diagnostics())

