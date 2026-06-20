"""
organizer.py
============
整理 RetrievalContext，過濾雜訊與重複資料，
產出結構化的 OrganizedContext 供 generator 生成回覆時使用。

處理流程
--------
  1. 格式轉換   — TextUnitResult / CommunityResult → OrganizedChunk（統一格式）
  2. 分數過濾   — 移除低於 score_cutoff 的低相關片段
  3. 內容去重   — Jaccard 相似度去重，保留最高分版本
  4. 截斷       — 限制 max_chunks 數量
  5. 實體整合   — 跨所有片段去重，按出現頻率排序
  6. 來源去重   — 以 (video, start_time) 為 key 合併重複來源

公開 API
--------
  organize(retrieval_context, query,
           score_cutoff, max_chunks, dedup_threshold) -> OrganizedContext
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


# ══════════════════════════════════════════════════════════════════════════════
# 資料類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrganizedChunk:
    """
    整理後的單筆上下文片段，統一表示 local text_unit 或 global community。

    Attributes
    ----------
    chunk_id    唯一識別碼（繼承自 TextUnitResult.id 或 CommunityResult.id）
    chunk_type  "text_unit"（local）或 "community"（global）
    title       可讀標題（community title 或 "video start-end" 格式）
    content     完整文字（text_unit 的 text 或 community 的 summary）
    score       相關性分數（來自 searcher，cosine similarity）
    entities    關聯實體列表（EntitySnippet）
    source_refs 可溯源的影片/時間戳參考（SourceRef）
    """
    chunk_id:   str
    chunk_type: Literal["text_unit", "community"]
    title:      str
    content:    str
    score:      float
    entities:   list = field(default_factory=list)    # list[EntitySnippet]
    source_refs: list = field(default_factory=list)   # list[SourceRef]


@dataclass
class OrganizedContext:
    """
    整理後的查詢上下文，供 generator 生成回覆。

    Attributes
    ----------
    query           原始查詢字串
    query_type      "local" 或 "global"
    chunks          過濾 + 去重後的片段列表（按相關性分數降序）
    key_entities    跨所有片段整合後的實體列表（按出現頻率降序）
    all_source_refs 去重後的所有來源（按 video + start_time 排序）
    entity_graph    以查詢實體為種子的圖上下文（EntityGraphContext | None）
    """
    query:          str
    query_type:     str
    chunks:         list[OrganizedChunk] = field(default_factory=list)
    key_entities:   list = field(default_factory=list)   # list[EntitySnippet]
    all_source_refs: list = field(default_factory=list)  # list[SourceRef]
    entity_graph:   object = None                        # EntityGraphContext | None

    def is_empty(self) -> bool:
        eg = self.entity_graph
        has_graph = eg is not None and not eg.is_empty()
        return not self.chunks and not has_graph

    def as_prompt_text(self, max_content_chars: int = 800) -> str:
        """
        將整理後的上下文格式化為可直接傳入 LLM 的純文字。

        格式
        ----
        每個 chunk 以 [CHUNK_ID: xxx] 標記，方便 generator 做來源引用。
        community chunk 附 Topic 標題；text_unit chunk 省略。
        每個 chunk 最多顯示前 3 個實體與前 2 個來源時間戳。
        末尾附關鍵實體索引（最多 10 個）。
        """
        sections: list[str] = []

        for chunk in self.chunks:
            lines: list[str] = []
            lines.append(f"[CHUNK_ID: {chunk.chunk_id}]  ({chunk.chunk_type})")

            if chunk.chunk_type == "community" and chunk.title:
                lines.append(f"Topic: {chunk.title}")

            # 截斷過長內容
            content = chunk.content
            if len(content) > max_content_chars:
                content = content[:max_content_chars] + "..."
            lines.append(content)

            if chunk.entities:
                ent_str = ", ".join(
                    f"{e.name}({e.type})" for e in chunk.entities[:3]
                )
                lines.append(f"Entities: {ent_str}")

            for ref in chunk.source_refs[:2]:
                ts = f"{ref.start_time:.1f}s-{ref.end_time:.1f}s"
                lines.append(f"[SOURCE: {ref.source_video}  {ts}]")

            sections.append("\n".join(lines))

        text = "\n\n---\n\n".join(sections)

        if self.key_entities:
            ent_lines = ["KEY ENTITIES:"]
            for e in self.key_entities[:10]:
                desc = f" — {e.description[:80]}" if e.description else ""
                ent_lines.append(f"  {e.name} ({e.type}){desc}")
            text += "\n\n" + "\n".join(ent_lines)

        eg = self.entity_graph
        if eg is not None and not eg.is_empty():
            eg_lines = ["ENTITY GRAPH (query-derived):"]

            if eg.seed_entities:
                eg_lines.append("  Matched entities:")
                for e in eg.seed_entities:
                    desc = f" — {e.description[:80]}" if e.description else ""
                    eg_lines.append(f"    {e.name} ({e.type}){desc}")

            if eg.neighbor_entities:
                eg_lines.append("  Related entities (1-hop):")
                for e in eg.neighbor_entities[:8]:
                    desc = f" — {e.description[:80]}" if e.description else ""
                    eg_lines.append(f"    {e.name} ({e.type}){desc}")

            if eg.relationships:
                eg_lines.append("  Relationships:")
                for r in eg.relationships[:15]:
                    desc = f": {r.description[:100]}" if r.description else ""
                    eg_lines.append(f"    {r.source} → {r.target}{desc}")

            text += "\n\n" + "\n".join(eg_lines)

        return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 內部工具
# ══════════════════════════════════════════════════════════════════════════════

def _jaccard(a: str, b: str) -> float:
    """計算兩段文字的詞彙 Jaccard 相似度（0–1）。"""
    ws_a = set(a.lower().split())
    ws_b = set(b.lower().split())
    if not ws_a or not ws_b:
        return 0.0
    return len(ws_a & ws_b) / len(ws_a | ws_b)


def _dedup_chunks(
    chunks: list[OrganizedChunk],
    threshold: float,
) -> list[OrganizedChunk]:
    """
    Greedy 去重：chunks 必須已按 score 降序排列。
    若後續 chunk 與已保留的任一 chunk 的 Jaccard 相似度 >= threshold，則丟棄。
    """
    kept: list[OrganizedChunk] = []
    for chunk in chunks:
        for prev in kept:
            if _jaccard(chunk.content, prev.content) >= threshold:
                logger.debug(
                    "去重跳過 %s（與 %s Jaccard=%.2f）",
                    chunk.chunk_id, prev.chunk_id,
                    _jaccard(chunk.content, prev.content),
                )
                break
        else:
            kept.append(chunk)
    return kept


def _consolidate_entities(chunks: list[OrganizedChunk]) -> list:
    """
    跨所有 chunk 整合實體：
    - 以 name.lower() 去重
    - 有 description 的版本優先保留
    - 按出現頻率（跨 chunk）降序排列
    """
    freq: dict[str, int] = {}
    best: dict[str, object] = {}

    for chunk in chunks:
        for e in chunk.entities:
            key = e.name.lower()
            freq[key] = freq.get(key, 0) + 1
            # 新增 or 覆蓋為更豐富的版本（有描述 > 無描述）
            if key not in best or (not getattr(best[key], "description", "") and e.description):
                best[key] = e

    return [best[k] for k in sorted(best, key=lambda k: freq[k], reverse=True)]


def _consolidate_source_refs(chunks: list[OrganizedChunk]) -> list:
    """
    跨所有 chunk 去重來源參考：
    - 以 (source_video, start_time) 為 key
    - 按 (source_video, start_time) 升序排列
    """
    seen: dict[tuple, object] = {}
    for chunk in chunks:
        for ref in chunk.source_refs:
            key = (ref.source_video, ref.start_time)
            if key not in seen:
                seen[key] = ref
    return sorted(seen.values(), key=lambda r: (r.source_video, r.start_time))


def _make_source_ref(r):
    """
    將 TextUnitResult 轉為 SourceRef。
    延遲 import 避免循環引用。
    """
    from qa_Module.graphrag.searcher import SourceRef
    return SourceRef(
        text_unit_id = r.id,
        source_video = r.source_video,
        start_time   = r.start_time,
        end_time     = r.end_time,
        slide_image  = r.slide_image,
        text_snippet = r.text,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API
# ══════════════════════════════════════════════════════════════════════════════

def organize(
    retrieval_context,              # RetrievalContext（避免循環 import）
    query: str,
    score_cutoff: float  = -0.5,
    max_chunks:   int    = 5,
    dedup_threshold: float = 0.6,
) -> OrganizedContext:
    """
    整理 RetrievalContext，回傳去雜訊後的 OrganizedContext。

    Parameters
    ----------
    retrieval_context   retriever.retrieve() 的回傳值（RetrievalContext）
    query               原始查詢字串
    score_cutoff        分數門檻，低於此值的片段將被過濾（預設 -0.5）
    max_chunks          最多保留的片段數（預設 5）
    dedup_threshold     Jaccard 去重門檻，超過此值視為重複（預設 0.6）

    Returns
    -------
    OrganizedContext
    """
    query_type = retrieval_context.query_type
    raw_chunks: list[OrganizedChunk] = []

    # ── 格式轉換：統一為 OrganizedChunk ──────────────────────────────────────
    if query_type == "local":
        for r in retrieval_context.text_unit_results:
            title = (
                f"{r.source_video}  {r.start_time:.1f}s-{r.end_time:.1f}s"
                if r.source_video else r.id
            )
            raw_chunks.append(OrganizedChunk(
                chunk_id    = r.id,
                chunk_type  = "text_unit",
                title       = title,
                content     = r.text,
                score       = r.score,
                entities    = list(r.related_entities),
                source_refs = [_make_source_ref(r)],
            ))

    else:  # global
        for r in retrieval_context.community_results:
            raw_chunks.append(OrganizedChunk(
                chunk_id    = r.id,
                chunk_type  = "community",
                title       = r.title,
                content     = r.summary,
                score       = r.score,
                entities    = list(r.entities),
                source_refs = list(r.source_refs),
            ))

    total_raw = len(raw_chunks)

    # ── Step 1: 分數過濾 ──────────────────────────────────────────────────────
    raw_chunks = [c for c in raw_chunks if c.score >= score_cutoff]
    logger.info(
        "分數過濾（≥%.2f）：%d → %d 筆",
        score_cutoff, total_raw, len(raw_chunks),
    )

    # ── Step 2: 按分數降序排列（去重必須先排序） ──────────────────────────────
    raw_chunks.sort(key=lambda c: c.score, reverse=True)

    # ── Step 3: 內容去重 ──────────────────────────────────────────────────────
    before_dedup = len(raw_chunks)
    raw_chunks = _dedup_chunks(raw_chunks, threshold=dedup_threshold)
    if before_dedup != len(raw_chunks):
        logger.info("內容去重：%d → %d 筆", before_dedup, len(raw_chunks))

    # ── Step 4: 截斷至 max_chunks ─────────────────────────────────────────────
    raw_chunks = raw_chunks[:max_chunks]

    # ── Step 5: 整合實體與來源 ────────────────────────────────────────────────
    key_entities    = _consolidate_entities(raw_chunks)
    all_source_refs = _consolidate_source_refs(raw_chunks)

    logger.info(
        "organize 完成：%d chunks / %d 實體 / %d 來源",
        len(raw_chunks), len(key_entities), len(all_source_refs),
    )

    return OrganizedContext(
        query           = query,
        query_type      = query_type,
        chunks          = raw_chunks,
        key_entities    = key_entities,
        all_source_refs = all_source_refs,
        entity_graph    = getattr(retrieval_context, "entity_graph", None),
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
    from qa_Module.query_processor import process_query
    from qa_Module.retriever import retrieve
    from qa_Module.graphrag.vectorizer import _load_settings
    from qa_Module.llm import create_llm_from_config

    OUTPUT_DIR    = _ROOT / "qa_Module/graphrag/output"
    LANCEDB_PATH  = _ROOT / "qa_Module/graphrag/lancedb"
    SETTINGS_PATH = _ROOT / "qa_Module/graphrag/settings.yaml"

    settings = _load_settings(SETTINGS_PATH, SETTINGS_PATH.parent)
    llm      = create_llm_from_config(dict(settings.get("llm", {})))
    searcher = GraphRAGSearcher(OUTPUT_DIR, LANCEDB_PATH)

    TEST_QUERIES = [
        "Who chairs the IETF ADD working group?",
        "What are the main topics discussed across all sessions?",
    ]

    for q in TEST_QUERIES:
        print(f"\n{'='*60}")
        qr  = process_query(q, llm)
        ctx = retrieve(qr, searcher, top_k=5)
        org = organize(ctx, query=q)

        print(f"query      : {q}")
        print(f"query_type : {org.query_type}")
        print(f"chunks     : {len(org.chunks)}")
        print(f"entities   : {len(org.key_entities)}")
        print(f"sources    : {len(org.all_source_refs)}")

        print("\n--- OrganizedChunks ---")
        for i, chunk in enumerate(org.chunks, 1):
            print(f"  [{i}] ({chunk.chunk_type})  score={chunk.score:.4f}")
            print(f"       title  : {chunk.title}")
            print(f"       content: {chunk.content[:120]}...")
            if chunk.entities:
                print(f"       entities ({len(chunk.entities)}): " + ", ".join(
                    f"{e.name}({e.type})" for e in chunk.entities[:3]
                ))
            for ref in chunk.source_refs[:2]:
                print(f"       source : {ref.source_video}  "
                      f"{ref.start_time:.1f}s-{ref.end_time:.1f}s")

        print("\n--- as_prompt_text (preview) ---")
        print(org.as_prompt_text()[:])
