"""
run_eval.py
===========
meetGRAG GraphRAG 評估框架入口點。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 確保 project root 在 sys.path
_EVAL_DIR = Path(__file__).resolve().parent
_ROOT = _EVAL_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# 設定區（修改這裡來調整評估參數）
# ══════════════════════════════════════════════════════════════════════════════

# 測試集路徑（相對於專案根目錄）
TESTSET_PATH = "eval/testset/samples/sample_testset.json"

# 要計算的指標
METRICS = [
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "citation_accuracy",
]

# 評估判斷模式："llm" | "rule" | "hybrid"
JUDGE_MODE     = "llm"
JUDGE_PROVIDER = "groq"
JUDGE_MODEL    = "llama-3.3-70b-versatile"

# Pipeline 參數
TOP_K        = 5
SCORE_CUTOFF = -0.5
MAX_CHUNKS   = 5

# 系統路徑（相對於專案根目錄）
GRAPHRAG_OUTPUT_DIR = "qa_Module/graphrag/output"
LANCEDB_PATH        = "qa_Module/graphrag/lancedb"
SETTINGS_PATH       = "qa_Module/graphrag/settings.yaml"

# 篩選條件（留空表示全部執行）
FILTER_IDS        = []          # 例如：["sample_001", "sample_002"]
FILTER_DIFFICULTY = []          # 例如：["easy", "medium"]
FILTER_TAGS       = []          # 例如：["global"]

# 輸出
OUTPUT_DIR      = "eval/results"
REPORT_FORMAT   = "both"        # "json" | "csv" | "both"
SAVE_INTERMEDIATE = True

# 日誌詳細程度
VERBOSE = False


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_validate(path: Path) -> bool:
    from eval.testset.validator import validate_and_report
    return validate_and_report(path)


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_logging(VERBOSE)

    from eval.config import EvalConfig
    from eval.runner import EvalRunner
    from eval.reporter import EvalReporter
    from eval.testset.validator import validate_testset

    testset_path = _ROOT / TESTSET_PATH

    # 驗證測試集格式
    errors = validate_testset(testset_path)
    if errors:
        print("測試集格式錯誤：", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    formats = ["json", "csv"] if REPORT_FORMAT == "both" else [REPORT_FORMAT]

    config = EvalConfig(
        testset_path=testset_path,
        output_dir=_ROOT / OUTPUT_DIR,
        top_k=TOP_K,
        score_cutoff=SCORE_CUTOFF,
        max_chunks=MAX_CHUNKS,
        metrics=METRICS,
        judge_mode=JUDGE_MODE,
        judge_llm_provider=JUDGE_PROVIDER,
        judge_llm_model=JUDGE_MODEL,
        save_intermediate=SAVE_INTERMEDIATE,
        report_formats=formats,
        filter_tags=FILTER_TAGS,
        filter_difficulty=FILTER_DIFFICULTY,
        filter_ids=FILTER_IDS,
        output_graphrag_dir=_ROOT / GRAPHRAG_OUTPUT_DIR,
        lancedb_path=_ROOT / LANCEDB_PATH,
        settings_path=_ROOT / SETTINGS_PATH,
    )

    runner = EvalRunner(config)
    report = runner.run()

    saved_paths = EvalReporter.save(report, config.output_dir / report.run_id, formats)
    EvalReporter.print_summary(report)

    for p in saved_paths:
        print(f"報告已儲存：{p}")
