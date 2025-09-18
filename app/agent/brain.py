"""Language model abstraction for chat-based reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(slots=True)
class Message:
    """Message structure compatible with Ollama and OpenAI-style APIs."""

    role: str
    content: str


class Brain:
    """Thin wrapper over an Ollama chat model."""

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.1:8b",
        temperature: float = 0.6,
    ) -> None:
        self._backend = backend
        self._model = model
        self._temperature = temperature
        if backend == "ollama":
            import ollama

            self._client = ollama.Client()
        else:
            raise NotImplementedError(f"Backend '{backend}' is not implemented yet")

    def chat(self, messages: Iterable[Message | Dict[str, str]], **options: object) -> str:
        """Send messages to the configured model and return the response text."""

        payload: List[Dict[str, str]] = [
            message if isinstance(message, dict) else {"role": message.role, "content": message.content}
            for message in messages
        ]
        if self._backend == "ollama":
            response = self._client.chat(
                model=self._model,
                messages=payload,
                options={"temperature": self._temperature, **options},
            )
            return response["message"]["content"].strip()
        raise RuntimeError("Unsupported backend configured")

    @property
    def model(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return self._backend
