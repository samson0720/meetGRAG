"""
run_query.py
============
meetGRAG 問答管線的互動式執行入口。

完整流程
--------
  query
    → process_query  （LLM 分類：local / global，抽取實體與擴展查詢）
    → retrieve       （向量搜尋 TextUnit 或 Community Report + 實體圖遍歷）
    → organize       （分數過濾、內容去重、整合實體、來源與圖上下文）
    → generate       （LLM 生成有來源標注的回覆）

執行方式（meetGRAG 根目錄）
---------------------------
  python -m qa_Module.run_query
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── 確保專案根目錄在 sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows 終端機 UTF-8 輸出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 載入 .env（若存在）
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 設定區（修改這裡）
# ══════════════════════════════════════════════════════════════════════════════

# GraphRAG 資料路徑
GRAPHRAG_OUTPUT_DIR = _ROOT / "qa_Module/graphrag/output"
LANCEDB_PATH        = _ROOT / "qa_Module/graphrag/lancedb"
SETTINGS_PATH       = _ROOT / "qa_Module/graphrag/settings.yaml"

# 檢索參數
TOP_K           = 5      # 最多取回幾筆搜尋結果
SCORE_CUTOFF    = -0.5   # 低於此相關性分數的結果直接丟棄
MAX_CHUNKS      = 5      # 最多傳入 LLM 的片段數
DEDUP_THRESHOLD = 0.6    # Jaccard 去重門檻（0~1，越小越嚴格）

# 生成參數
GEN_TEMPERATURE = 0.2    # 回覆生成溫度
GEN_MAX_TOKENS  = 30000   # 回覆最大 token 數

# 顯示選項
SHOW_CHUNKS    = True    # 顯示檢索到的 context 片段
SHOW_CITATIONS = True    # 顯示引用來源明細
VERBOSE        = False   # 是否開啟 DEBUG 日誌


# ══════════════════════════════════════════════════════════════════════════════
# 初始化
# ══════════════════════════════════════════════════════════════════════════════

def _init_components():
    """載入 settings、建立 LLM client 與 GraphRAGSearcher。"""
    from qa_Module.graphrag.vectorizer import _load_settings
    from qa_Module.graphrag.searcher import GraphRAGSearcher
    from qa_Module.llm import create_llm_from_config

    settings = _load_settings(SETTINGS_PATH, SETTINGS_PATH.parent)
    llm      = create_llm_from_config(dict(settings.get("llm", {})))
    searcher = GraphRAGSearcher(GRAPHRAG_OUTPUT_DIR, LANCEDB_PATH)

    return llm, searcher


# ══════════════════════════════════════════════════════════════════════════════
# 問答管線
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str, llm, searcher):
    """
    執行完整問答管線，回傳 (GeneratedAnswer, OrganizedContext)。

    Steps
    -----
    1. process_query  — LLM 分類查詢類型，抽取實體並產生 expanded_query
    2. retrieve       — 向量搜尋（local: TextUnit / global: Community）+ 實體圖遍歷
    3. organize       — 過濾、去重、整合實體、來源與圖上下文
    4. generate       — LLM 生成有來源標注的回覆
    """
    from qa_Module.query_processor import process_query
    from qa_Module.retriever import retrieve
    from qa_Module.organizer import organize
    from qa_Module.generator import generate

    # Step 1：查詢分類
    query_result = process_query(query, llm)
    logging.info(
        "[1] query_type=%s  entities=%s  expanded=%r  reason=%s",
        query_result.query_type,
        query_result.entities,
        query_result.expanded_query,
        query_result.reasoning,
    )

    # Step 2：檢索（向量搜尋 + 實體圖遍歷）
    retrieval_ctx = retrieve(query_result, searcher, top_k=TOP_K)
    result_count = (
        len(retrieval_ctx.text_unit_results)
        if retrieval_ctx.text_unit_results
        else len(retrieval_ctx.community_results)
    )
    eg = retrieval_ctx.entity_graph
    eg_info = (
        f"  entity_graph: {len(eg.seed_entities)} seeds / "
        f"{len(eg.neighbor_entities)} neighbors / "
        f"{len(eg.relationships)} rels"
        if eg and not eg.is_empty() else ""
    )
    logging.info("[2] 檢索結果：%d 筆%s", result_count, eg_info)

    # Step 3：整理
    organized_ctx = organize(
        retrieval_ctx,
        query           = query,
        score_cutoff    = SCORE_CUTOFF,
        max_chunks      = MAX_CHUNKS,
        dedup_threshold = DEDUP_THRESHOLD,
    )
    logging.info("[3] 整理後：%d 個 chunk", len(organized_ctx.chunks))

    # Step 4：生成
    answer = generate(
        organized_ctx,
        query       = query,
        llm         = llm,
        temperature = GEN_TEMPERATURE,
        max_tokens  = GEN_MAX_TOKENS,
    )
    logging.info("[4] 生成完成（fallback=%s）", answer.is_fallback)

    return answer, organized_ctx


# ══════════════════════════════════════════════════════════════════════════════
# 輸出格式化
# ══════════════════════════════════════════════════════════════════════════════

def _sep(char: str = "=", width: int = 70):
    print(char * width)


def _print_result(query: str, answer, organized_ctx) -> None:
    """格式化輸出問答結果。"""
    _sep()
    print(f"Query : {query}")
    print(f"Type  : {answer.query_type}  |  fallback={answer.is_fallback}")
    if answer.usage:
        print(
            f"Tokens: prompt={answer.usage.get('prompt', 0)}"
            f"  completion={answer.usage.get('completion', 0)}"
        )
    _sep("-")

    # 回覆本文
    print("\n[Answer]")
    print(answer.answer)

    # 引用來源
    if SHOW_CITATIONS and answer.citations:
        print(f"\n[Citations ({len(answer.citations)})]")
        for i, c in enumerate(answer.citations, 1):
            print(f"  [{i}] {c.chunk_type}  chunk_id={c.chunk_id}")
            print(f"       {c.title}")
            for ref in c.source_refs[:2]:
                ts    = f"{ref.start_time:.1f}s - {ref.end_time:.1f}s"
                slide = (
                    f"  slide={ref.slide_image}"
                    if getattr(ref, "slide_image", "") else ""
                )
                print(f"       > {ref.source_video}  {ts}{slide}")

    # 檢索片段摘要
    if SHOW_CHUNKS and organized_ctx.chunks:
        print(f"\n[Retrieved Chunks ({len(organized_ctx.chunks)})]")
        for i, chunk in enumerate(organized_ctx.chunks, 1):
            preview = chunk.content[:100].replace("\n", " ")
            print(f"  [{i}] ({chunk.chunk_type})  score={chunk.score:.4f}  {chunk.title}")
            print(f"       {preview}...")

    # 實體圖摘要
    eg = organized_ctx.entity_graph
    if SHOW_CHUNKS and eg and not eg.is_empty():
        print(f"\n[Entity Graph  seeds={len(eg.seed_entities)}  "
              f"neighbors={len(eg.neighbor_entities)}  "
              f"rels={len(eg.relationships)}]")
        if eg.seed_entities:
            print("  Seeds: " + ", ".join(
                f"{e.name}({e.type})" for e in eg.seed_entities
            ))
        if eg.neighbor_entities:
            print("  Neighbors: " + ", ".join(
                f"{e.name}({e.type})" for e in eg.neighbor_entities[:5]
            ))
        if eg.relationships:
            print("  Top relations:")
            for r in eg.relationships[:5]:
                print(f"    {r.source} → {r.target}"
                      + (f"  [{r.description[:60]}]" if r.description else ""))

    _sep()


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.DEBUG if VERBOSE else logging.WARNING,
        format  = "%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
    )

    print("=== meetGRAG 問答系統 ===")
    print(f"資料來源：{GRAPHRAG_OUTPUT_DIR}")
    print("輸入問題後按 Enter，輸入 'exit' 或 'quit' 離開\n")

    print("初始化中...", end="", flush=True)
    try:
        llm, searcher = _init_components()
    except Exception as exc:
        print(f"\n[ERROR] 初始化失敗：{exc}")
        print("請確認：")
        print("  1. settings.yaml 存在且設定正確")
        print("  2. GraphRAG index 已建立（先執行 run_index.py）")
        print("  3. LanceDB 向量索引已建立")
        sys.exit(1)
    print(" 完成\n")

    while True:
        try:
            query = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n離開。")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("離開。")
            break

        try:
            answer, organized_ctx = run_pipeline(query, llm, searcher)
            _print_result(query, answer, organized_ctx)
        except Exception as exc:
            logging.exception("問答管線執行失敗")
            print(f"\n[ERROR] {exc}\n")
