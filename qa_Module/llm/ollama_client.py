"""
Ollama client — local LLM inference via Ollama server.

Ollama exposes an OpenAI-compatible Chat Completions endpoint at
http://localhost:11434/v1, so this client reuses the openai SDK.

Requires: ollama running locally (`ollama serve`)
          pip install openai
"""
from __future__ import annotations

from openai import OpenAI

from .base import BaseLLMClient, LLMResponse, Message

_DEFAULT_BASE = "http://localhost:11434/v1"


class OllamaClient(BaseLLMClient):
    """
    Parameters
    ----------
    model : str
        Model tag pulled in Ollama, e.g. "mistral", "llama3.2", "phi3.5".
    base_url : str
        Ollama server URL. Defaults to http://localhost:11434/v1.
    """

    def _init(self, base_url: str = _DEFAULT_BASE, **_):
        # Ollama doesn't require a real key, but the SDK needs a non-empty value
        self._client = OpenAI(api_key="ollama", base_url=base_url)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        raw_messages = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=raw_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = resp.choices[0].message
        usage = resp.usage or {}
        return LLMResponse(
            content=choice.content or "",
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            raw=resp.model_dump(),
        )
