"""
generator.py
============
接收 OrganizedContext，呼叫 LLM 生成回覆，並附帶結構化的引用來源資訊。

設計重點
--------
  - LLM 僅可使用 context 中的資訊回答（grounded generation）
  - 回答中以 [REF:chunk_id] 標注引用來源，事後解析為結構化 Citation
  - 回傳 GeneratedAnswer，包含回答文字與 Citation 列表，供 UI 呈現跳轉連結

Citation 結構
-------------
  每筆 Citation 對應一個 OrganizedChunk，包含：
    - chunk_id / chunk_type / title
    - source_refs：影片名稱 + 起止時間戳（UI 可用來生成跳轉連結）

公開 API
--------
  generate(organized_context, query, llm,
           language, temperature, max_tokens) -> GeneratedAnswer
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


# ══════════════════════════════════════════════════════════════════════════════
# 資料類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Citation:
    """
    單筆引用來源，對應一個 OrganizedChunk。

    Attributes
    ----------
    chunk_id    對應 OrganizedChunk.chunk_id（也是 LLM 回答中 [REF:xxx] 的 xxx）
    chunk_type  "text_unit" 或 "community"
    title       可讀標題（community title 或 "video start-end"）
    source_refs 影片/時間戳列表，UI 可用來生成跳轉連結
                每筆包含：source_video, start_time, end_time, slide_image
    """
    chunk_id:    str
    chunk_type:  str
    title:       str
    source_refs: list = field(default_factory=list)   # list[SourceRef]


@dataclass
class GeneratedAnswer:
    """
    LLM 生成結果的完整容器。

    Attributes
    ----------
    answer      回覆文字，含行內 [REF:chunk_id] 標注
    citations   已解析的引用來源列表（依回答中出現順序排列）
    query       原始查詢字串
    query_type  "local" 或 "global"
    is_fallback 若 context 為空而使用 fallback 回覆則為 True
    usage       LLM token 用量 {"prompt": N, "completion": N}
    """
    answer:      str
    citations:   list[Citation] = field(default_factory=list)
    query:       str = ""
    query_type:  str = ""
    is_fallback: bool = False
    usage:       dict = field(default_factory=dict)

    def cited_chunk_ids(self) -> list[str]:
        """回傳所有被引用的 chunk_id 列表（依出現順序）。"""
        return [c.chunk_id for c in self.citations]

    def as_plain_text(self) -> str:
        """
        去除 [REF:xxx] 標記的純文字版本，供純文字輸出使用。
        """
        return re.sub(r'\[REF:[^\]]+\]', '', self.answer).strip()

    def format_citations(self) -> str:
        """
        格式化引用來源為可讀文字，附在回答末尾。

        範例輸出
        --------
        References:
          [1] IETF118_ADD_session.mp4  0.0s-1800.0s
          [2] Adaptive DNS Discovery Working Group Overview
        """
        if not self.citations:
            return ""
        lines = ["References:"]
        for i, c in enumerate(self.citations, 1):
            if c.source_refs:
                for ref in c.source_refs[:2]:
                    ts = f"{ref.start_time:.1f}s-{ref.end_time:.1f}s"
                    lines.append(f"  [{i}] {ref.source_video}  {ts}")
            else:
                lines.append(f"  [{i}] {c.title}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ══════════════════════════════════════════════════════════════════════════════

from prompts.generator import SYSTEM_PROMPT as _SYSTEM_PROMPT
from prompts.generator import USER_PROMPT as _USER_PROMPT
from prompts.generator import FALLBACK_ANSWER as _FALLBACK_ANSWER


# ══════════════════════════════════════════════════════════════════════════════
# 內部工具
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_citations(answer: str) -> str:
    """
    將 LLM 可能輸出的非標準引用格式統一轉換為 [REF:id]。

    常見錯誤：全形括號【REF:id】、圓括號(REF:id)、角括號<REF:id>、
             全形冒號【REF：id】。
    """
    # 全形冒號 → 半形
    answer = answer.replace('\uff1a', ':')
    # 【REF:id】→ [REF:id]
    answer = re.sub(r'[\u3010【]REF:([^\u3011】]+)[\u3011】]', r'[REF:\1]', answer)
    # (REF:id) → [REF:id]
    answer = re.sub(r'\(REF:([^)]+)\)', r'[REF:\1]', answer)
    # <REF:id> → [REF:id]
    answer = re.sub(r'<REF:([^>]+)>', r'[REF:\1]', answer)
    return answer


def _extract_citations(
    answer: str,
    chunk_map: dict,   # chunk_id → OrganizedChunk
) -> list[Citation]:
    """
    從 LLM 回答中解析所有 [REF:chunk_id] 標記，
    依出現順序建立 Citation 列表（同一 id 只取第一次出現）。
    解析前先呼叫 _normalize_citations 統一格式。
    """
    answer = _normalize_citations(answer)
    pattern = re.compile(r'\[REF:([^\]]+)\]')
    seen: set[str] = set()
    citations: list[Citation] = []

    for m in pattern.finditer(answer):
        cid = m.group(1).strip()
        if cid in seen:
            continue
        seen.add(cid)

        chunk = chunk_map.get(cid)
        if chunk is None:
            logger.warning("LLM 引用了未知的 chunk_id：%r，略過", cid)
            continue

        citations.append(Citation(
            chunk_id    = cid,
            chunk_type  = chunk.chunk_type,
            title       = chunk.title,
            source_refs = list(chunk.source_refs),
        ))

    return citations


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API
# ══════════════════════════════════════════════════════════════════════════════

def generate(
    organized_context,           # OrganizedContext（避免循環 import）
    query: str,
    llm,                         # BaseLLMClient
    temperature: float = 0.2,
    max_tokens:  int   = 30000,
) -> GeneratedAnswer:
    """
    以 OrganizedContext 為參考，呼叫 LLM 生成有來源標注的回覆。

    Parameters
    ----------
    organized_context   organizer.organize() 的回傳值（OrganizedContext）
    query               原始查詢字串
    llm                 已初始化的 LLM client（BaseLLMClient）
    temperature         生成溫度（預設 0.2，偏確定性）
    max_tokens          最大生成 token 數

    Returns
    -------
    GeneratedAnswer
      .answer      含 [REF:chunk_id] 標注的回覆文字
      .citations   已解析的引用來源列表（依出現順序）
      .is_fallback 若 context 為空則為 True
    """
    from qa_Module.llm import Message

    # ── Context 為空 → fallback ────────────────────────────────────────────────
    if organized_context.is_empty():
        logger.warning("generate：OrganizedContext 為空（無 chunks 且無 entity_graph），回傳 fallback 回覆")
        return GeneratedAnswer(
            answer      = _FALLBACK_ANSWER,
            query       = query,
            query_type  = organized_context.query_type,
            is_fallback = True,
        )

    # ── 建立 chunk_map 供事後解析引用 ────────────────────────────────────────
    chunk_map = {c.chunk_id: c for c in organized_context.chunks}

    # ── 組裝 prompt ───────────────────────────────────────────────────────────
    context_text = organized_context.as_prompt_text()
    eg = organized_context.entity_graph
    if eg and not eg.is_empty():
        logger.info(
            "generate：context 含 entity_graph（seeds=%d / neighbors=%d / rels=%d）",
            len(eg.seed_entities), len(eg.neighbor_entities), len(eg.relationships),
        )
    user_content = _USER_PROMPT.format(context=context_text, query=query)

    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", user_content),
    ]

    # ── 呼叫 LLM ──────────────────────────────────────────────────────────────
    try:
        resp = llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
        answer = _normalize_citations(resp.content.strip())
        usage = {
            "prompt":     resp.prompt_tokens,
            "completion": resp.completion_tokens,
        }
        logger.info(
            "generate 完成：%d prompt tokens / %d completion tokens",
            resp.prompt_tokens, resp.completion_tokens,
        )
    except Exception as exc:
        logger.error("LLM 呼叫失敗：%s", exc)
        raise

    # ── 解析 [REF:xxx] 引用 ───────────────────────────────────────────────────
    citations = _extract_citations(answer, chunk_map)
    logger.info(
        "引用解析：回答中共 %d 筆引用，對應 %d 個唯一 chunk",
        len(re.findall(r'\[REF:', answer)),
        len(citations),
    )

    return GeneratedAnswer(
        answer     = answer,
        citations  = citations,
        query      = query,
        query_type = organized_context.query_type,
        usage      = usage,
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

    from qa_Module.graphrag.searcher import GraphRAGSearcher
    from qa_Module.graphrag.vectorizer import _load_settings
    from qa_Module.llm import create_llm_from_config
    from qa_Module.query_processor import process_query
    from qa_Module.retriever import retrieve
    from qa_Module.organizer import organize

    OUTPUT_DIR    = _ROOT / "qa_Module/graphrag/output"
    LANCEDB_PATH  = _ROOT / "qa_Module/graphrag/lancedb"
    SETTINGS_PATH = _ROOT / "qa_Module/graphrag/settings.yaml"

    settings = _load_settings(SETTINGS_PATH, SETTINGS_PATH.parent)
    llm      = create_llm_from_config(dict(settings.get("llm", {})))
    searcher = GraphRAGSearcher(OUTPUT_DIR, LANCEDB_PATH)

    TEST_QUERIES = [
        "誰主持了這次會議？",
        "IETF 118 ADD session 中有哪些主要議題？",
    ]

    for q in TEST_QUERIES:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        print('='*70)

        qr  = process_query(q, llm)
        ctx = retrieve(qr, searcher, top_k=5)
        org = organize(ctx, query=q)
        ans = generate(org, query=q, llm=llm)

        print(f"\n[query_type: {ans.query_type}  fallback: {ans.is_fallback}]")
        print(f"[tokens: prompt={ans.usage.get('prompt',0)}  "
              f"completion={ans.usage.get('completion',0)}]")

        print("\n--- Answer ---")
        print(ans.answer)

        if ans.citations:
            print(f"\n--- Citations ({len(ans.citations)}) ---")
            for i, c in enumerate(ans.citations, 1):
                print(f"  [{i}] chunk_id={c.chunk_id}  type={c.chunk_type}")
                print(f"       title: {c.title}")
                for ref in c.source_refs[:3]:
                    ts = f"{ref.start_time:.1f}s-{ref.end_time:.1f}s"
                    img = f"  slide={ref.slide_image}" if ref.slide_image else ""
                    print(f"       source: {ref.source_video}  {ts}{img}")

        print(f"\n--- References block ---")
        print(ans.format_citations())
