"""Speech-to-text helper built on Faster-Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


class STT:
    """Light wrapper around the Faster-Whisper transcription model."""

    def __init__(
        self,
        language: str = "en",
        model_size: str = "medium",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, compute_type=compute_type)
        self._language = language

    def transcribe(self, wav_path: str | Path) -> str:
        """Transcribe an audio file into text."""

        segments, _ = self._model.transcribe(str(Path(wav_path)), language=self._language)
        texts: List[str] = [segment.text.strip() for segment in segments]
        return " ".join(text for text in texts if text).strip()

    def transcribe_segments(self, wav_path: str | Path) -> List[str]:
        """Return individual segments for downstream processing."""

        segments, _ = self._model.transcribe(str(Path(wav_path)), language=self._language)
        return [segment.text.strip() for segment in segments if segment.text]

    def available_languages(self) -> Iterable[str]:
        """Expose languages supported by the underlying model."""

        return self._model.available_languages
