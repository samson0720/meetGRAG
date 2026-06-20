"""
LLM factory — single entry point for creating LLM clients.

Usage
-----
from qa_Module.llm import create_llm

# Ollama (local)
llm = create_llm("ollama", model="mistral")

# Groq
llm = create_llm("groq", model="llama-3.3-70b-versatile", api_key="gsk_...")

# OpenAI
llm = create_llm("openai", model="gpt-4o", api_key="sk-...")

# OpenAI-compatible local server (e.g. LM Studio)
llm = create_llm("openai", model="local-model", api_key="any", api_base="http://localhost:1234/v1")

# From settings dict (e.g. loaded from settings.yaml)
llm = create_llm_from_config({"provider": "ollama", "model": "mistral"})
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from .base import BaseLLMClient
from .groq_client import GroqClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient

_REGISTRY: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "groq": GroqClient,
    "ollama": OllamaClient,
}


def create_llm(provider: str, model: str, **kwargs) -> BaseLLMClient:
    """
    Instantiate an LLM client by provider name.

    Parameters
    ----------
    provider : str
        One of: "openai", "groq", "ollama".
    model : str
        Model identifier understood by the chosen provider.
    **kwargs
        Extra arguments forwarded to the client constructor
        (e.g. api_key, api_base, base_url).
        kwargs["api_key"] = api_key

    Raises
    ------
    ValueError
        If `provider` is not registered.
    """
    key = provider.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown LLM provider '{provider}'. Available: {available}")
    
    load_dotenv()

    if provider == 'groq':
        # 收集所有非空的 Groq API Key（GROQ_API_KEY 為第一組，GROQ_API_KEY_2~10 為額外組）
        raw_keys = [os.getenv("GROQ_API_KEY", "")]
        for i in range(2, 11):
            raw_keys.append(os.getenv(f"GROQ_API_KEY_{i}", ""))
        api_keys = [k for k in raw_keys if k]
        if not api_keys:
            api_keys = ["default_key_if_not_found"]
        return _REGISTRY[key](model=model, api_keys=api_keys, **kwargs)
    elif provider == 'openai':
        api_key = os.getenv("OPENAPI_API_KEY", "default_key_if_not_found")
    else:
        api_key = kwargs.get("api_key", "")

    return _REGISTRY[key](model=model, api_key=api_key, **kwargs)


def create_llm_from_config(config: dict[str, Any]) -> BaseLLMClient:
    """
    Build a client from a config dict (e.g. parsed from settings.yaml).

    Expected keys:
        provider  (required) — "openai" | "groq" | "ollama"
        model     (required)
        api_key   (optional)
        api_base  (optional)
        base_url  (optional)
    """
    config = dict(config)  
    provider = config.pop("provider")
    model = config.pop("model")
    return create_llm(provider, model, **config)


def register_provider(name: str, client_class: type[BaseLLMClient]) -> None:
    """Register a custom provider at runtime."""
    _REGISTRY[name.lower().strip()] = client_class
