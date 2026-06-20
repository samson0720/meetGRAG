"""
query_processor.py
==================
查詢語意解析：透過 LLM 將使用者原始查詢分類為 local 或 global，
並抽取核心概念關鍵字，供 retriever 路由至對應搜尋策略。

判斷規則
--------
  local  — 詢問特定技術、人物、RFC、組織、具體事件或特定時間點
  global — 詢問整體趨勢、比較、演進歷史、「哪些」「概述」「有什麼」

公開 API
--------
  process_query(query: str, llm) -> QueryResult
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]

# ══════════════════════════════════════════════════════════════════════════════
# 資料類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    """LLM 分析後的查詢結果。"""
    query: str
    query_type: Literal["local", "global"]
    entities: list[str] = field(default_factory=list)   # 查詢中明確提及的實體名稱
    expanded_query: str = ""                             # 檢索最佳化後的查詢字串
    reasoning: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════════════════════

from prompts.query_processor import SYSTEM_PROMPT as _SYSTEM_PROMPT
from prompts.query_processor import USER_PROMPT as _USER_PROMPT


# ══════════════════════════════════════════════════════════════════════════════
# 內部輔助
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_raw(raw: str) -> str:
    """清理 LLM 回覆中常見的格式問題。"""
    # 去除 BOM
    text = raw.lstrip("\ufeff")
    # 全形引號 → ASCII 引號
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text.strip()


def _parse_json(raw: str) -> dict:
    """
    從 LLM 回覆中提取 JSON 物件，依序嘗試三種策略：
    1. 直接解析（LLM 輸出純 JSON）
    2. 去除 markdown fence（```json ... ```）後解析
    3. 從文字中抽取第一個 {...} 區塊後解析
    """
    text = _normalize_raw(raw)

    # 策略 1：直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略 2：去除 markdown fence
    cleaned = re.sub(r"^```(?:json)?\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略 3：抽取第一個 {...} 區塊（LLM 在 JSON 前後加說明文字時）
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        return json.loads(m.group())

    raise ValueError(f"No valid JSON found in LLM output: {repr(text[:300])}")


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API
# ══════════════════════════════════════════════════════════════════════════════

def process_query(query: str, llm) -> QueryResult:
    """
    呼叫 LLM 分析查詢，回傳 QueryResult。

    Parameters
    ----------
    query   使用者原始查詢字串
    llm     已初始化的 LLM client（BaseLLMClient）

    Returns
    -------
    QueryResult，若 LLM 解析失敗則以關鍵字規則降級判斷。
    """
    from qa_Module.llm import Message

    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", _USER_PROMPT.format(query=query)),
    ]

    raw = ""
    try:
        resp = llm.chat(messages, temperature=0.0, max_tokens=1024)
        raw = resp.content.strip()
        print(f"LLM raw output repr: {repr(raw[:300])}")
        data = _parse_json(raw)

        query_type = data.get("query_type", "local")
        if query_type not in ("local", "global"):
            query_type = "local"

        return QueryResult(
            query          = query,
            query_type     = query_type,
            entities       = data.get("entities", []),
            expanded_query = data.get("expanded_query", query),
            reasoning      = data.get("reasoning", ""),
        )
    except Exception as exc:
        logger.warning("LLM 查詢分類失敗（%s），使用關鍵字規則降級", exc)
        return _fallback_classify(query)


def _fallback_classify(query: str) -> QueryResult:
    """關鍵字規則降級分類，不需要 LLM。"""
    q = query.lower()
    global_keywords = (
        "overview", "trend", "history", "compare", "evolution",
        "what are all", "which groups", "overall", "summary",
        "概述", "趨勢", "整體", "比較", "歷史", "哪些",
    )
    query_type: Literal["local", "global"] = (
        "global" if any(kw in q for kw in global_keywords) else "local"
    )
    return QueryResult(
        query      = query,
        query_type = query_type,
        reasoning  = "fallback keyword rule",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 命令列快速測試
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from qa_Module.graphrag.vectorizer import _load_settings
    from qa_Module.llm import create_llm_from_config

    SETTINGS_PATH = _ROOT / "qa_Module/graphrag/settings.yaml"
    settings = _load_settings(SETTINGS_PATH, SETTINGS_PATH.parent)
    llm = create_llm_from_config(dict(settings.get("llm", {})))

    TEST_QUERIES = [
        "Who chairs the IETF ADD working group?",
        "What is the overall direction of IETF ADD working group discussions?",
        "Which RFC defines DNS over HTTPS?",
        "What are all the protocols discussed in these sessions?",
        "IETF 118 ADD session 中有哪些主要議題？",
        "誰主持了這次會議？",
    ]

    for q in TEST_QUERIES:
        result = process_query(q, llm)
        print(f"[{result.query_type:6}] {q}")
        print(f"         entities: {result.entities}")
        print(f"         expanded: {result.expanded_query}")
        print(f"         reason  : {result.reasoning}\n")
