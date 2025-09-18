"""Text-to-speech helper that shells out to Piper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


class TTS:
    """Run Piper to synthesise spoken audio for assistant replies."""

    def __init__(
        self,
        voice_path: str | Path,
        piper_executable: str = "piper",
        extra_args: Iterable[str] | None = None,
    ) -> None:
        self._voice_path = Path(voice_path)
        self._piper_executable = piper_executable
        self._extra_args = list(extra_args or [])

    def synth(self, text: str, out_path: str | Path) -> Path:
        """Synthesise ``text`` into ``out_path`` and return the resulting path."""

        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self._piper_executable,
            "--model",
            str(self._voice_path),
            "--output_file",
            str(destination),
            *self._extra_args,
        ]
        subprocess.run(command, check=True, input=text.encode("utf-8"))
        return destination
