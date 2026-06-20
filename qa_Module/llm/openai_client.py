"""
OpenAI (and OpenAI-compatible) client.

Works with:
  - OpenAI cloud  (api_base=None, api_key="sk-...")
  - Azure OpenAI  (api_base="https://<resource>.openai.azure.com/", api_version="2024-02-01")
  - Any proxy / local server that exposes the OpenAI Chat Completions API
    (e.g. LM Studio, vLLM, Mistral.ai cloud)
"""
from __future__ import annotations

from openai import OpenAI

from .base import BaseLLMClient, LLMResponse, Message


class OpenAIClient(BaseLLMClient):
    """
    Parameters
    ----------
    model : str
        Model name, e.g. "gpt-4o", "gpt-3.5-turbo".
    api_key : str
        OpenAI API key.
    api_base : str | None
        Override the default endpoint (useful for Azure or local proxies).
    api_version : str | None
        Required for Azure deployments.
    """

    def _init(
        self,
        api_key: str,
        api_base: str | None = None,
        api_version: str | None = None,
        **_,
    ):
        kwargs: dict = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        if api_version:
            kwargs["default_query"] = {"api-version": api_version}
        self._client = OpenAI(**kwargs)

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
            model=resp.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            raw=resp.model_dump(),
        )
