"""
retriever.py
============
根據 QueryResult.query_type 路由至對應的搜尋策略，
回傳統一格式的 RetrievalContext 供 organizer 使用。

路由規則
--------
  local  → GraphRAGSearcher.local_search()   (TextUnit 向量 + 實體展開)
  global → GraphRAGSearcher.global_search()  (Community Report 向量)

公開 API
--------
  retrieve(query_result, searcher, top_k) -> RetrievalContext
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


# ══════════════════════════════════════════════════════════════════════════════
# 資料類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalContext:
    """
    搜尋結果容器，供 organizer / generator 使用。

    Attributes
    ----------
    query_type        使用的搜尋策略（"local" 或 "global"）
    community_results Global Search 結果（query_type == "global" 時有值）
    text_unit_results Local Search 結果（query_type == "local" 時有值）
    entity_graph      以查詢實體為種子的圖上下文（有 entities 時才有值）
    """
    query_type: str
    community_results: list = field(default_factory=list)   # list[CommunityResult]
    text_unit_results: list = field(default_factory=list)   # list[TextUnitResult]
    entity_graph: object = None                             # EntityGraphContext | None

    def is_empty(self) -> bool:
        return not self.community_results and not self.text_unit_results

    def as_context_text(self) -> str:
        """
        將搜尋結果轉為純文字段落，供 LLM 作為 context 使用。
        每段附帶 SOURCE_ID 標注，方便 generator 追溯來源。
        """
        lines: list[str] = []

        for r in self.text_unit_results:
            lines.append(
                f"[SOURCE_ID: {r.id}  video={r.source_video}"
                f"  time={r.start_time:.1f}s-{r.end_time:.1f}s]"
            )
            lines.append(r.text)
            if r.related_entities:
                ent_str = ", ".join(
                    f"{e.name}({e.type})" for e in r.related_entities[:5]
                )
                lines.append(f"Related entities: {ent_str}")
            lines.append("")

        for r in self.community_results:
            lines.append(f"[COMMUNITY_ID: {r.id}  title={r.title}]")
            lines.append(r.summary)
            if r.entities:
                ent_str = ", ".join(
                    f"{e.name}({e.type})" for e in r.entities[:5]
                )
                lines.append(f"Key entities: {ent_str}")
            if r.source_refs:
                for ref in r.source_refs:
                    lines.append(
                        f"  [SOURCE_ID: {ref.text_unit_id}"
                        f"  video={ref.source_video}"
                        f"  time={ref.start_time:.1f}s-{ref.end_time:.1f}s]"
                    )
            lines.append("")

        return "\n".join(lines).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API
# ══════════════════════════════════════════════════════════════════════════════

def _filter_text_units(results: list, filter_stems: set[str]) -> list:
    """保留 source_video stem 在 filter_stems 中的 TextUnitResult。"""
    from pathlib import Path as _Path
    return [
        r for r in results
        if _Path(r.source_video).stem.lower() in filter_stems
    ]


def _filter_communities(results: list, filter_stems: set[str]) -> list:
    """
    保留社群結果中至少有一個 source_ref 符合目標會議的項目。
    若社群沒有任何 source_ref（純文字摘要），保留以免遺失跨會議洞察。
    """
    from pathlib import Path as _Path
    filtered = []
    for r in results:
        if not r.source_refs:
            filtered.append(r)   # 無來源可判斷，保留
        elif any(
            _Path(ref.source_video).stem.lower() in filter_stems
            for ref in r.source_refs if ref.source_video
        ):
            filtered.append(r)
    return filtered


def retrieve(
    query_result,                           # QueryResult（避免循環 import，不標注型別）
    searcher,                               # GraphRAGSearcher
    top_k: int = 5,
    meeting_filter: list[str] | None = None,  # None = 不限定會議，否則只保留指定會議
) -> RetrievalContext:
    """
    根據 query_result.query_type 路由搜尋。

    Parameters
    ----------
    query_result      process_query() 的回傳值（QueryResult）
    searcher          已初始化的 GraphRAGSearcher
    top_k             回傳結果筆數上限
    meeting_filter    會議名稱清單（不含副檔名），限制結果來源；None 表示全部會議

    Returns
    -------
    RetrievalContext
    """
    query_type     = query_result.query_type
    search_query   = query_result.expanded_query or query_result.query
    entities       = getattr(query_result, "entities", [])

    # 將 meeting_filter 轉為小寫 stem set，供後續比對
    filter_stems: set[str] | None = (
        {m.lower() for m in meeting_filter} if meeting_filter else None
    )

    # ── 向量搜尋（使用 expanded_query 優化檢索） ──────────────────────────────
    # 若有會議過濾，多取一倍結果再過濾，避免過濾後數量不足
    fetch_k = top_k * 2 if filter_stems else top_k

    if query_type == "global":
        logger.info("retrieve → global_search  query=%r  meeting_filter=%s",
                    search_query[:60], meeting_filter)
        results = searcher.global_search(search_query, top_k=fetch_k)
        if filter_stems:
            before = len(results)
            results = _filter_communities(results, filter_stems)
            logger.info("會議過濾（%s）：社群 %d → %d 筆", meeting_filter, before, len(results))
        ctx = RetrievalContext(
            query_type        = "global",
            community_results = results,
        )
    else:  # local（預設）
        logger.info("retrieve → local_search  query=%r  meeting_filter=%s",
                    search_query[:60], meeting_filter)
        results = searcher.local_search(search_query, top_k=fetch_k)
        if filter_stems:
            before = len(results)
            results = _filter_text_units(results, filter_stems)
            logger.info("會議過濾（%s）：文字塊 %d → %d 筆", meeting_filter, before, len(results))
        ctx = RetrievalContext(
            query_type        = "local",
            text_unit_results = results,
        )

    # ── 圖遍歷：以查詢實體為種子擴展相關節點與關係 ────────────────────────────
    if entities:
        logger.info("retrieve → entity_graph_search  entities=%s", entities)
        ctx.entity_graph = searcher.entity_graph_search(entities, hops=1, max_neighbors=10)

    return ctx


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
        ctx = retrieve(qr, searcher, top_k=3)

        print(f"query      : {q}")
        print(f"query_type : {ctx.query_type}")
        print(f"results    : {len(ctx.text_unit_results)} text_units / "
              f"{len(ctx.community_results)} communities")
        print("\n--- context_text ---")
        print(ctx.as_context_text()[:])
