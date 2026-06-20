"""
Abstract base class for LLM clients.
All concrete clients must implement `chat()` and `complete()`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Optional[dict] = field(default=None, repr=False)


class BaseLLMClient(ABC):
    """
    Uniform interface for all LLM backends.

    Usage
    -----
    response = client.chat(
        messages=[Message("user", "What is QUIC?")],
        temperature=0.2,
        max_tokens=512,
    )
    print(response.content)
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self._init(**kwargs)

    def _init(self, **kwargs):
        """Optional hook for subclass initialisation."""

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
        **kwargs,
    ) -> LLMResponse:
        """Send a multi-turn chat request and return a structured response."""

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 5000,
        **kwargs,
    ) -> LLMResponse:
        """Convenience wrapper — builds a messages list and calls chat()."""
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
