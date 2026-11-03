"""
iris/adapters/tts/macos_tts.py
------------------------------
Local macOS Native TTS Adapter using /usr/bin/say and afconvert.

Zero cloud dependency, zero network roundtrip, runs 100% locally on macOS.
Defaults to natural British English voices: 'Flo (English (UK))', 'Shelley', or 'Sandy'.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from iris.adapters.tts.base import TTSAdapter

logger = logging.getLogger(__name__)


class MacOSSayTTS(TTSAdapter):
    """
    Synthesizes speech using the macOS built-in speech synthesis system.
    """

    def __init__(
        self,
        voice_name: str = "Flo (English (UK))",
        speaking_rate_wpm: int = 185,
    ) -> None:
        self._voice_name = voice_name
        self._rate = speaking_rate_wpm

    @property
    def voice_name(self) -> str:
        return self._voice_name

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        clean_text = text.strip()
        if not clean_text:
            return np.zeros(0, dtype=np.int16), 22050

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
                samples = np.frombuffer(raw_bytes, dtype=np.int16)

            return samples, sample_rate

