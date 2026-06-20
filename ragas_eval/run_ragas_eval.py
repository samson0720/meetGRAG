"""
run_ragas_eval.py
=================
使用 RAGAS 套件評估 meetGRAG GraphRAG pipeline 的準確度。

執行方式
--------
  ragas_eval\\.venv\\Scripts\\activate
  python ragas_eval\\run_ragas_eval.py

環境變數（放在 .env）
---------------------
  GROQ_API_KEY      : Groq API 金鑰
  OPENAI_API_KEY    : OpenAI API 金鑰（answer_relevancy / correctness 需要）
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── 確保 meetGRAG 根目錄在 sys.path ──────────────────────────────────────────
_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 載入 .env（若存在）
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    load_dotenv(_DIR / ".env")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 設定區（修改這裡來調整評估參數）
# ══════════════════════════════════════════════════════════════════════════════

# 測試集路徑（相對於專案根目錄）
TESTSET_PATH = "eval/testset/samples/sample_testset.json"

# Judge LLM：provider 選 "groq" 或 "openai"
LLM_PROVIDER = "groq"
LLM_MODEL    = "llama-3.3-70b-versatile"

# 要計算的 RAGAS 指標
# 可選：faithfulness / answer_relevancy / context_precision / context_recall / answer_correctness
# 注意：answer_relevancy / answer_correctness 需要 OPENAI_API_KEY（embeddings）
METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    # "answer_relevancy",   # 需要 OPENAI_API_KEY
    # "answer_correctness", # 需要 OPENAI_API_KEY
]

# meetGRAG Pipeline 參數
TOP_K        = 5
SCORE_CUTOFF = -0.5
MAX_CHUNKS   = 5

# 系統路徑（相對於專案根目錄）
GRAPHRAG_OUTPUT_DIR = "qa_Module/graphrag/output"
LANCEDB_PATH        = "qa_Module/graphrag/lancedb"
SETTINGS_PATH       = "qa_Module/graphrag/settings.yaml"

# 結果輸出目錄
OUTPUT_DIR = "ragas_eval/results"

# 篩選條件（留空表示全部執行）
FILTER_IDS        = []                    # 例如：["sample_001", "sample_002"]
FILTER_DIFFICULTY = []                    # 例如：["easy", "medium"]
FILTER_TAGS       = []                    # 例如：["global"]

# 日誌詳細程度
VERBOSE = False


# ══════════════════════════════════════════════════════════════════════════════
# LLM / Embeddings 建構
# ══════════════════════════════════════════════════════════════════════════════

def build_ragas_llm(provider: str, model: str):
    """建立 RAGAS 相容的 LLM wrapper。"""
    from ragas.llms import LangchainLLMWrapper

    if provider == "groq":
        from langchain_groq import ChatGroq
        lc_llm = ChatGroq(model=model, temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        lc_llm = ChatOpenAI(model=model, temperature=0)
    else:
        raise ValueError(f"不支援的 provider：{provider}")

    return LangchainLLMWrapper(lc_llm)


def build_ragas_embeddings(provider: str):
    """建立 RAGAS 相容的 Embeddings wrapper（answer_relevancy / correctness 需要）。"""
    import os
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 指標建構
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics(metric_names: list[str], ragas_llm, ragas_embeddings):
    """建立 RAGAS Metric 物件清單。"""
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )

    needs_embeddings = {"answer_relevancy", "answer_correctness"}

    metric_map = {
        "faithfulness":       lambda: Faithfulness(llm=ragas_llm),
        "answer_relevancy":   lambda: AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        "context_precision":  lambda: ContextPrecision(llm=ragas_llm),
        "context_recall":     lambda: ContextRecall(llm=ragas_llm),
        "answer_correctness": lambda: AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    }

    metrics = []
    for name in metric_names:
        if name in needs_embeddings and ragas_embeddings is None:
            logging.warning(
                "指標 %s 需要 embeddings，但無 OPENAI_API_KEY，略過", name
            )
            continue
        metrics.append(metric_map[name]())

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 結果儲存
# ══════════════════════════════════════════════════════════════════════════════

def save_results(scores_df, case_ids: list[str], output_dir: Path, run_id: str) -> Path:
    """將 RAGAS 評估結果儲存為 JSON + CSV。"""
    out = output_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    df = scores_df.copy()
    if len(case_ids) == len(df):
        df.insert(0, "case_id", case_ids)

    # CSV
    df.to_csv(out / "ragas_scores.csv", index=False, encoding="utf-8-sig")

    # JSON 摘要
    skip_cols = {"case_id", "user_input", "response", "retrieved_contexts", "reference"}
    metric_cols = [c for c in df.columns if c not in skip_cols]
    summary = {
        "run_id": run_id,
        "total_cases": len(df),
        "metric_averages": {
            col: round(float(df[col].mean()), 4)
            for col in metric_cols
            if df[col].dtype.kind in ("f", "i")
        },
        "per_case": (
            df[["case_id"] + metric_cols].to_dict(orient="records")
            if "case_id" in df.columns
            else df[metric_cols].to_dict(orient="records")
        ),
    }
    with open(out / "ragas_scores.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out


def print_summary(scores_df, metric_cols: list[str]) -> None:
    print("\n" + "=" * 60)
    print("RAGAS 評估結果摘要")
    print("=" * 60)
    for col in metric_cols:
        if col in scores_df.columns and scores_df[col].dtype.kind in ("f", "i"):
            print(f"  {col:<30} {scores_df[col].mean():.4f}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_id = f"ragas_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info("RAGAS 評估開始，run_id=%s", run_id)

    # Step 1：建構 RAGAS dataset
    from ragas_eval.adapter import build_ragas_dataset

    dataset, case_ids = build_ragas_dataset(
        testset_path=_ROOT / TESTSET_PATH,
        graphrag_output_dir=_ROOT / GRAPHRAG_OUTPUT_DIR,
        lancedb_path=_ROOT / LANCEDB_PATH,
        settings_path=_ROOT / SETTINGS_PATH,
        top_k=TOP_K,
        score_cutoff=SCORE_CUTOFF,
        max_chunks=MAX_CHUNKS,
        filter_ids=FILTER_IDS or None,
        filter_difficulty=FILTER_DIFFICULTY or None,
        filter_tags=FILTER_TAGS or None,
    )

    if len(dataset.samples) == 0:
        logging.error("沒有可評估的樣本，請確認測試集內容與篩選條件")
        sys.exit(1)

    # Step 2：建構 LLM / Embeddings
    logging.info("初始化 Judge LLM：%s / %s", LLM_PROVIDER, LLM_MODEL)
    ragas_llm = build_ragas_llm(LLM_PROVIDER, LLM_MODEL)
    ragas_embeddings = build_ragas_embeddings(LLM_PROVIDER)

    # Step 3：建構指標
    metrics = build_metrics(METRICS, ragas_llm, ragas_embeddings)
    if not metrics:
        logging.error("沒有有效的評估指標")
        sys.exit(1)
    logging.info("評估指標：%s", [m.name for m in metrics])

    # Step 4：執行 RAGAS evaluate
    from ragas import evaluate

    logging.info("開始 RAGAS 評估（%d 筆樣本）...", len(dataset.samples))
    result = evaluate(dataset=dataset, metrics=metrics)

    scores_df = result.to_pandas()
    metric_cols = [m.name for m in metrics]

    # Step 5：儲存結果
    out_path = save_results(scores_df, case_ids, _ROOT / OUTPUT_DIR, run_id)
    print_summary(scores_df, metric_cols)
    print(f"\n結果已儲存：{out_path}")
