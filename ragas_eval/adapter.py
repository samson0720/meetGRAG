"""
adapter.py
==========
將 meetGRAG 測試集（JSON）與 Pipeline 執行結果轉換為 RAGAS EvaluationDataset。

流程
----
1. 讀取 meetGRAG testset JSON（格式同 eval/testset/samples/sample_testset.json）
2. 對每道題呼叫 meetGRAG EvalPipeline.run()，取得 answer 與 retrieved contexts
3. 組裝成 ragas.dataset_schema.SingleTurnSample 清單
4. 回傳 ragas.EvaluationDataset
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── 確保 meetGRAG 根目錄在 sys.path ──────────────────────────────────────────
_RAGAS_EVAL_DIR = Path(__file__).resolve().parent
_ROOT = _RAGAS_EVAL_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_testset(testset_path: Path) -> tuple[dict, list[dict]]:
    """讀取 meetGRAG testset JSON，回傳 (metadata, test_cases)。"""
    with open(testset_path, encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    cases = data.get("test_cases", [])
    logger.info("載入測試集：%s（共 %d 題）", testset_path, len(cases))
    return metadata, cases


def filter_cases(
    cases: list[dict],
    filter_ids: Optional[list[str]] = None,
    filter_difficulty: Optional[list[str]] = None,
    filter_tags: Optional[list[str]] = None,
) -> list[dict]:
    """套用篩選條件，回傳過濾後的測試案例。"""
    result = cases
    if filter_ids:
        result = [c for c in result if c.get("id") in filter_ids]
    if filter_difficulty:
        result = [c for c in result if c.get("difficulty") in filter_difficulty]
    if filter_tags:
        tag_set = set(filter_tags)
        result = [c for c in result if tag_set & set(c.get("tags", []))]
    return result


def build_ragas_dataset(
    testset_path: Path,
    graphrag_output_dir: Path,
    lancedb_path: Path,
    settings_path: Path,
    top_k: int = 5,
    score_cutoff: float = -0.5,
    max_chunks: int = 5,
    filter_ids: Optional[list[str]] = None,
    filter_difficulty: Optional[list[str]] = None,
    filter_tags: Optional[list[str]] = None,
):
    """
    執行 meetGRAG pipeline，將結果包裝為 ragas EvaluationDataset。

    Parameters
    ----------
    testset_path        : meetGRAG testset JSON 路徑
    graphrag_output_dir : GraphRAG output 目錄（parquet 檔案）
    lancedb_path        : LanceDB 向量資料庫路徑
    settings_path       : GraphRAG settings.yaml 路徑
    top_k               : 每題檢索的向量鄰居數
    score_cutoff        : 向量相似度門檻
    max_chunks          : 最多保留的 chunk 數
    filter_*            : 與 run_eval.py 相同的篩選參數

    Returns
    -------
    dataset : ragas.EvaluationDataset
    case_ids : list[str]，對應每個 sample 的 case_id（方便結果對照）
    """
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    # 匯入 meetGRAG eval pipeline（重用現有邏輯）
    from eval.config import EvalConfig
    from eval.pipeline import EvalPipeline

    config = EvalConfig(
        testset_path=testset_path,
        output_graphrag_dir=graphrag_output_dir,
        lancedb_path=lancedb_path,
        settings_path=settings_path,
        top_k=top_k,
        score_cutoff=score_cutoff,
        max_chunks=max_chunks,
    )

    pipeline = EvalPipeline.from_config(config)
    logger.info("EvalPipeline 初始化完成")

    metadata, all_cases = load_testset(testset_path)
    cases = filter_cases(all_cases, filter_ids, filter_difficulty, filter_tags)
    logger.info("篩選後：%d 題", len(cases))

    samples: list[SingleTurnSample] = []
    case_ids: list[str] = []
    skipped = 0

    for i, tc in enumerate(cases, 1):
        case_id = tc.get("id", f"case_{i}")
        query = tc.get("query", "")
        reference = tc.get("expected_answer", "")

        logger.info("[%d/%d] 執行：%s", i, len(cases), case_id)
        try:
            pr = pipeline.run(query)
        except Exception as exc:
            logger.error("案例 %s pipeline 失敗：%s，跳過", case_id, exc)
            skipped += 1
            continue

        if pr.error:
            logger.warning("案例 %s 回傳 error：%s，跳過", case_id, pr.error)
            skipped += 1
            continue

        # retrieved_chunks_content 對應 RAGAS 的 retrieved_contexts
        retrieved_contexts = pr.retrieved_chunks_content or []
        if not retrieved_contexts:
            logger.warning("案例 %s 無 retrieved context，以空清單代替", case_id)

        sample = SingleTurnSample(
            user_input=query,
            response=pr.answer,
            retrieved_contexts=retrieved_contexts,
            reference=reference if reference else None,
        )
        samples.append(sample)
        case_ids.append(case_id)

    if skipped:
        logger.warning("共跳過 %d 題（pipeline 失敗或 error）", skipped)

    logger.info("建立 EvaluationDataset：%d 筆 sample", len(samples))
    return EvaluationDataset(samples=samples), case_ids
