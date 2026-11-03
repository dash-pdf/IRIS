"""
scripts/test_kokoro_tts.py
--------------------------
Standalone test for the Kokoro TTS Adapter.

Tests:
1. Model loading (single load verification).
2. Audio synthesis across British female voices (bf_emma, bf_alice, bf_isabella).
3. Latency and Real-Time Factor (RTF) measurement.
4. Audio playback via speak() and instant abort via stop().
5. Saving generated audio to WAV.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf

from iris.adapters.tts.kokoro_tts import (
    BRITISH_FEMALE_VOICES,
    DEFAULT_VOICE,
    KokoroTTS,
)


def run_standalone_test(play_audio: bool = True, selected_voice: str = DEFAULT_VOICE) -> int:
    print("=" * 70)
    print("   STANDALONE KOKORO TTS ADAPTER TEST")
    print("=" * 70)

    # 1. Model Initialization
    print("\n[Step 1] Initializing KokoroTTS adapter...")
    t0 = time.perf_counter()
    try:
        tts = KokoroTTS(voice_name=selected_voice)
        # Trigger model load
        engine = tts._get_engine()
        t1 = time.perf_counter()
        print(f"  --> PASS: Model loaded successfully in {(t1 - t0) * 1000:.1f} ms")
        print(f"      Model Path:  {tts.model_path}")
        print(f"      Voices Path: {tts.voices_path}")
        print(f"      Active Voice: {tts.voice_name}")
    except Exception as exc:
        print(f"  --> FAIL: Model loading error: {exc}")
        return 1

    # 2. Benchmark synthesis across British female presets
    print("\n[Step 2] Testing synthesis across British female voice presets...")
    test_phrase = "Good afternoon. I am IRIS, your personal intelligent companion."

    for v in ["bf_emma", "bf_alice", "bf_isabella"]:
        try:
            t_start = time.perf_counter()
            samples, sr = tts.generate_audio(test_phrase, voice=v)
            t_end = time.perf_counter()

            latency_ms = (t_end - t_start) * 1000
            duration_s = len(samples) / sr
            rtf = latency_ms / (duration_s * 1000)

            print(f"  Voice '{v}':")
            print(f"    - Output: {len(samples)} samples @ {sr}Hz (Audio Duration: {duration_s:.2f}s)")
            print(f"    - Synthesis Latency: {latency_ms:.1f}ms (RTF: {rtf:.2f})")
            assert len(samples) > 0
            assert sr == 24000
        except Exception as exc:
            print(f"  --> FAIL: Synthesis failed for voice {v}: {exc}")
            return 1

    print("  --> PASS: All British female presets synthesized successfully.")

    # 3. Save sample WAV
    out_wav = PROJECT_ROOT / "test_kokoro_output.wav"
    print(f"\n[Step 3] Saving sample audio to {out_wav.name}...")
    try:
        samples, sr = tts.generate_audio(
            f"Hello, this is IRIS speaking with the British {selected_voice} voice preset.",
            voice=selected_voice,
        )
        sf.write(str(out_wav), samples, sr)
        print(f"  --> PASS: Saved {out_wav} ({out_wav.stat().st_size} bytes)")
    except Exception as exc:
        print(f"  --> FAIL: Could not save WAV: {exc}")
        return 1

    # 4. Test speak() playback
    if play_audio:
        print(f"\n[Step 4] Testing audio playback via speak() using voice '{selected_voice}'...")
        try:
            print("  Playing sample through speakers (blocking=True)...")
            tts.speak(
                "All systems nominal. Speech synthesis is fully operational.",
                voice=selected_voice,
                blocking=True,
            )
            print("  --> PASS: Audio playback finished cleanly.")
        except Exception as exc:
            print(f"  --> FAIL: Audio playback error: {exc}")
            return 1
    else:
        print("\n[Step 4] Skipping speaker playback (--no-play specified).")

    # 5. Test stop() abort
    print("\n[Step 5] Testing stop() abort functionality...")
    try:
        long_phrase = "This is a long sentence meant to test immediate playback cancellation and thread safety."
        # Start speaking asynchronously
        tts.speak(long_phrase, voice=selected_voice, blocking=False)
        time.sleep(0.1)
        # Abort immediately
        tts.stop()
        print("  --> PASS: stop() executed cleanly without raising errors.")
    except Exception as exc:
        print(f"  --> FAIL: stop() failed: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("   ALL KOKORO TTS STANDALONE TESTS PASSED!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Kokoro TTS Test")
    parser.add_argument("--voice", choices=list(BRITISH_FEMALE_VOICES.keys()), default=DEFAULT_VOICE)
    parser.add_argument("--no-play", action="store_true", help="Skip speaker playback")
    args = parser.parse_args()

    sys.exit(run_standalone_test(play_audio=not args.no_play, selected_voice=args.voice))

