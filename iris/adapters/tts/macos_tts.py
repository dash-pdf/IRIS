"""
iris/adapters/tts/macos_tts.py
------------------------------
Local macOS Native TTS Adapter using /usr/bin/say and afconvert.

Zero cloud dependency, zero network roundtrip, runs 100% locally on macOS.
Defaults to natural British English voices: 'Flo (English (UK))', 'Daniel (English (UK))',
'Shelley (English (UK))', or 'Sandy (English (UK))'.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from iris.adapters.tts.base import TTSAdapter
from iris.audio.player import AudioPlayer

logger = logging.getLogger(__name__)

# Preferred British English voice hierarchy on macOS
PREFERRED_BRITISH_VOICES = [
    "Flo (English (UK))",
    "Flo",
    "Daniel (English (UK))",
    "Daniel",
    "Shelley (English (UK))",
    "Shelley",
    "Sandy (English (UK))",
    "Sandy",
]


def get_installed_macos_voices() -> dict[str, str]:
    """
    Query /usr/bin/say -v '?' and return a dict mapping voice name to locale code.
    E.g. {'Flo (English (UK))': 'en_GB', 'Daniel (English (UK))': 'en_GB', ...}
    """
    if sys.platform != "darwin" or not os.path.exists("/usr/bin/say"):
        return {}

    try:
        res = subprocess.run(
            ["/usr/bin/say", "-v", "?"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        logger.debug("Failed to list macOS voices: %s", exc)
        return {}

    voices: dict[str, str] = {}
    pattern = re.compile(r"^(.*?)\s+([a-z]{2}_[A-Z]{2})\s+#")
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if m:
            v_name = m.group(1).strip()
            locale = m.group(2).strip()
            voices[v_name] = locale
    return voices


def resolve_macos_voice(preferred_voice: str = "Flo (English (UK))") -> str:
    """
    Verify voice existence on macOS.
    If preferred_voice is installed, returns it.
    If not, searches for installed British English (en_GB) voices in priority order.
    Returns the selected voice name.
    """
    installed = get_installed_macos_voices()
    if not installed:
        return preferred_voice

    clean_pref = preferred_voice.strip()

    # 1. Exact or prefix match for preferred_voice among installed voices
    for name in installed:
        if name.lower() == clean_pref.lower():
            return preferred_voice
        # Prefix or substring match (e.g. "Shelley" matches "Shelley (English (UK))")
        if clean_pref.lower() in name.lower():
            return preferred_voice

    # 2. Check prioritized British English alternatives
    for candidate in PREFERRED_BRITISH_VOICES:
        for name in installed:
            if candidate.lower() == name.lower() or (candidate.lower() in name.lower() and "en_GB" in installed[name]):
                logger.info(
                    "Preferred macOS voice '%s' not found. Selected British English voice '%s'.",
                    preferred_voice,
                    name,
                )
                return name

    # 3. Check any other en_GB voice
    for name, locale in installed.items():
        if locale == "en_GB":
            logger.info(
                "Preferred macOS voice '%s' not found. Selected alternative British voice '%s'.",
                preferred_voice,
                name,
            )
            return name

    # 4. Fallback to preferred_voice as requested
    return preferred_voice


class MacOSSayTTS(TTSAdapter):
    """
    Synthesizes speech using the macOS built-in speech synthesis system.
    """

    def __init__(
        self,
        voice_name: str = "Flo (English (UK))",
        speaking_rate_wpm: int = 185,
        player: AudioPlayer | None = None,
    ) -> None:
        self._voice_name = resolve_macos_voice(voice_name)
        self._rate = speaking_rate_wpm
        self._player = player or AudioPlayer()

    @property
    def voice_name(self) -> str:
        return self._voice_name

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(0, dtype=np.float32), 22050

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            aiff_file = tmp_path / "speech.aiff"
            wav_file = tmp_path / "speech.wav"

            cmd_say = [
                "/usr/bin/say",
                "-v",
                self._voice_name,
                "-r",
                str(self._rate),
                "-o",
                str(aiff_file),
                clean_text,
            ]
            res_say = subprocess.run(
                cmd_say,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res_say.returncode != 0:
                logger.warning(
                    "say command failed with voice '%s': %s. Retrying with default voice.",
                    self._voice_name,
                    res_say.stderr,
                )
                # Fallback to default say without -v
                fallback_cmd = ["/usr/bin/say", "-o", str(aiff_file), clean_text]
                res_fallback = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res_fallback.returncode != 0:
                    raise RuntimeError(f"macOS say failed: {res_fallback.stderr}")

            # Convert AIFF to 16-bit PCM WAV
            cmd_convert = [
                "/usr/bin/afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16",
                str(aiff_file),
                str(wav_file),
            ]
            res_convert = subprocess.run(
                cmd_convert,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res_convert.returncode != 0:
                raise RuntimeError(f"afconvert failed: {res_convert.stderr}")

            with wave.open(str(wav_file), "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            return samples, sample_rate

    def generate_audio(
        self,
        text: str,
        voice: str | None = None,
        rate: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text into a numpy array (float32) and sample rate."""
        return self.synthesize(text)

    def speak(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        blocking: bool = False,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """Synthesize text and play audio through speakers."""
        samples, sample_rate = self.generate_audio(text, voice=voice)
        if len(samples) == 0:
            if on_done:
                on_done()
            return

        self._player.play(samples, sample_rate=sample_rate, on_done=on_done)
        if blocking:
            self._player.wait()

    def stop(self) -> None:
        """Immediately abort any active audio playback."""
        self._player.stop()


# Alias for backward and forward compatibility
MacOSTTS = MacOSSayTTS
