"""
Groq client — ultra-fast inference via Groq Cloud.

Supported models (as of 2025):
  llama-3.3-70b-versatile, llama-3.1-8b-instant,
  mixtral-8x7b-32768, gemma2-9b-it, ...

Requires: pip install groq

API Key 輪換：若提供多組 api_keys，遇到非 200 回應時自動換下一組。
"""
from __future__ import annotations

import logging
import time

import groq as groq_module
from groq import Groq

from .base import BaseLLMClient, LLMResponse, Message

logger = logging.getLogger(__name__)


class GroqClient(BaseLLMClient):
    """
    Parameters
    ----------
    model : str
        Groq model ID, e.g. "llama-3.3-70b-versatile".
    api_key : str
        單一 Groq API key（與 api_keys 二擇一）。
    api_keys : list[str]
        多組 Groq API key；呼叫失敗時自動輪換到下一組。
    """

    def _init(self, api_key: str = "", api_keys: list[str] | None = None, **_):
        if api_keys:
            self._api_keys = [k for k in api_keys if k]
        elif api_key:
            self._api_keys = [api_key]
        else:
            self._api_keys = []

        if not self._api_keys:
            raise ValueError("GroqClient: 至少需要一個有效的 API Key")

        self._key_idx = 0
        self._client = Groq(api_key=self._api_keys[0], max_retries=0)
        logger.info("GroqClient 初始化：共 %d 組 API Key", len(self._api_keys))

    def _rotate_key(self) -> bool:
        """輪換到下一組 API Key。回傳 False 表示只有一組、無法輪換。"""
        if len(self._api_keys) <= 1:
            return False
        self._key_idx = (self._key_idx + 1) % len(self._api_keys)
        self._client = Groq(api_key=self._api_keys[self._key_idx], max_retries=0)
        logger.warning("Groq API Key 已輪換至第 %d 組", self._key_idx + 1)
        return True

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
        **kwargs,
    ) -> LLMResponse:
        _WAIT = 60       # 所有 key 均 429 時等待秒數
        _MAX_ROUNDS = 2  # 最多重試幾輪（每輪輪過全部 key 一次）

        raw_messages = [{"role": m.role, "content": m.content} for m in messages]
        n_keys = len(self._api_keys)
        resp = None

        for round_ in range(_MAX_ROUNDS):
            for attempt in range(n_keys):
                try:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=raw_messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        **kwargs,
                    )
                    break  # 成功，跳出內層
                except groq_module.RateLimitError:
                    if attempt < n_keys - 1:
                        self._rotate_key()
                        logger.info("429 → 已換至 Key #%d，立即重試", self._key_idx + 1)
                        time.sleep(5)
                except groq_module.APITimeoutError:
                    # 逾時：換 key 後重試；若已是最後一組則往上拋
                    logger.warning("Groq Key #%d 請求逾時，換 key 重試", self._key_idx + 1)
                    self._rotate_key()
                    if attempt >= n_keys - 1:
                        raise
                    time.sleep(2)
                except groq_module.APIStatusError as e:
                    logger.warning(
                        "Groq API Key #%d 回應錯誤 (HTTP %d)：%s",
                        self._key_idx + 1, e.status_code, e.message,
                    )
                    self._rotate_key()
                    time.sleep(5)
                    raise

            if resp is not None:
                break  # 成功，跳出外層

            # 本輪所有 key 均 429
            if round_ < _MAX_ROUNDS - 1:
                logger.warning(
                    "所有 %d 組 Key 均觸發速率限制，等待 %ds 後重試（第 %d/%d 輪）…",
                    n_keys, _WAIT, round_ + 1, _MAX_ROUNDS,
                )
                time.sleep(_WAIT)
                self._rotate_key()
        else:
            raise RuntimeError(f"Groq chat: 所有 Key 已重試 {_MAX_ROUNDS} 輪，請求失敗")

        choice = resp.choices[0].message
        usage = resp.usage or {}
        content = choice.content
        if not content:
            content = getattr(choice, "reasoning_content", None)
        if not content:
            raw_dict = choice.model_dump() if hasattr(choice, "model_dump") else {}
            content = raw_dict.get("reasoning_content") or raw_dict.get("content") or ""
        return LLMResponse(
            content=content,
            model=resp.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            raw=resp.model_dump(),
        )
