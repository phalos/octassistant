"""Entry-point that wires up the companion assistant components."""

from __future__ import annotations

import atexit
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console

from agent import ActionExecutor, Brain, IntentRouter, MemoryManager, STT, TTS
from agent.brain import Message


SYSTEM_PROMPT = (
    "You are a friendly South African assistant named Kyle. Keep replies under 70 words. "
    "Use any provided memories to keep context."
)


class Recorder:
    """Simple WAV recorder using sounddevice."""

    def __init__(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        self._sample_rate = sample_rate
        self._channels = channels

    def record_wav(self, seconds: int, destination: Path) -> Path:
        import numpy as np
        import sounddevice as sd
        import wave

        destination.parent.mkdir(parents=True, exist_ok=True)
        frames = int(seconds * self._sample_rate)
        audio = sd.rec(frames, samplerate=self._sample_rate, channels=self._channels, dtype="float32")
        sd.wait()
        pcm_audio = np.int16(audio * 32767)
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(pcm_audio.tobytes())
        return destination


class VoiceAssistantApp:
    """Coordinates STT, LLM, TTS, memory, and actions."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        load_dotenv(base_dir / ".env", override=True)
        self._console = Console()
        self._ensure_data_dirs()

        backend = os.getenv("LLM_BACKEND", "ollama")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        voice_path = os.getenv("PIPER_VOICE")
        stt_language = os.getenv("STT_LANGUAGE", "en")

        self.memory = MemoryManager()
        self.actions = ActionExecutor(self._notify)
        self.actions.start()
        self.router = IntentRouter(self.memory, self.actions)
        self.brain = Brain(backend=backend, model=model)
        self.stt = STT(language=stt_language)
        self.recorder = Recorder()
        self.tts: Optional[TTS]
        if voice_path:
            self.tts = TTS(voice_path)
        else:
            self.tts = None

        atexit.register(self._cleanup)

    def _ensure_data_dirs(self) -> None:
        for path in (Path("data/audio_tmp"), Path("data/logs")):
            path.mkdir(parents=True, exist_ok=True)

    def _cleanup(self) -> None:
        self.actions.stop()
        self.memory.close()

    def _notify(self, message: str) -> None:
        self._console.print(f"[yellow][Reminder][/yellow] {message}")

    def handle_text(self, user_text: str) -> str:
        routed = self.router.route(user_text)
        if routed:
            reply = routed
        else:
            memories = self.memory.search_facts(user_text)
            memory_context = "\n".join(f"• {item}" for item in memories)
            system_prompt = SYSTEM_PROMPT
            if memory_context:
                system_prompt += "\nKnown facts:\n" + memory_context
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_text),
            ]
            reply = self.brain.chat(messages)
        self.actions.append_journal({"user": user_text, "assistant": reply})
        if self.tts:
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            out_path = Path("data/audio_tmp") / f"reply-{timestamp}.wav"
            try:
                self.tts.synth(reply, out_path)
                self._console.print(f"[green]Spoken reply saved to[/green] {out_path}")
            except Exception as exc:  # noqa: BLE001
                self._console.print(f"[red]Failed to run Piper:[/red] {exc}")
        return reply

    def run_cli(self) -> None:
        self._console.print("[bold green]Companion AI ready. Type ':quit' to exit.[/bold green]")
        while True:
            try:
                user_text = input("You › ")
            except (EOFError, KeyboardInterrupt):
                self._console.print("\nGoodbye!")
                break
            if user_text.strip().lower() in {":quit", "exit", "quit"}:
                self._console.print("Take care!")
                break
            if user_text.strip().lower() == ":voice":
                if not sys.platform.startswith("win"):
                    self._console.print("Voice capture is optimised for Windows; text mode is recommended here.")
                    continue
                wav_path = Path("data/audio_tmp") / "input.wav"
                self.recorder.record_wav(seconds=10, destination=wav_path)
                user_text = self.stt.transcribe(wav_path)
                self._console.print(f"[cyan]Transcribed:[/cyan] {user_text}")
            if not user_text.strip():
                continue
            reply = self.handle_text(user_text)
            self._console.print(f"[magenta]Kyle:[/magenta] {reply}")


def main() -> None:
    app = VoiceAssistantApp()
    app.run_cli()


if __name__ == "__main__":
    main()
