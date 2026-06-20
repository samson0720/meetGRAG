"""
runner.py
=========
EvalRunner：批次執行評估測試集，產生 EvalReport。

使用方式
--------
  config = EvalConfig(testset_path=Path("eval/testset/samples/sample_testset.json"))
  runner = EvalRunner(config)
  report = runner.run()
"""
from __future__ import annotations

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from eval.config import EvalConfig
from eval.pipeline import EvalPipeline, PipelineResult
from eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 資料類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseResult:
    """單題評估結果。"""
    case_id: str
    query: str
    query_type_expected: str
    query_type_actual: str
    answer: str
    is_fallback: bool
    retrieved_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    metric_scores: dict[str, float] = field(default_factory=dict)
    metric_reasons: dict[str, str] = field(default_factory=dict)
    metric_details: dict[str, Optional[dict]] = field(default_factory=dict)
    passed_metrics: list[str] = field(default_factory=list)
    failed_metrics: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None

    def overall_score(self) -> float:
        if not self.metric_scores:
            return 0.0
        return sum(self.metric_scores.values()) / len(self.metric_scores)


@dataclass
class EvalReport:
    """完整評估報告。"""
    run_id: str
    started_at: str
    finished_at: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    case_results: list[CaseResult]
    metric_averages: dict[str, float] = field(default_factory=dict)
    metric_pass_rates: dict[str, float] = field(default_factory=dict)
    overall_average: float = 0.0
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_query_type: dict[str, dict] = field(default_factory=dict)
    by_tag: dict[str, dict] = field(default_factory=dict)
    # 保留測試集 metadata（方便報告中識別）
    testset_metadata: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# 主類別
# ══════════════════════════════════════════════════════════════════════════════

class EvalRunner:
    """批次評估執行器。"""

    def __init__(self, config: EvalConfig):
        self._config = config

    # ── 公開方法 ─────────────────────────────────────────────────────────────

    def run(self) -> EvalReport:
        """執行完整評估，回傳 EvalReport。"""
        config = self._config
        started_at = datetime.now(timezone.utc).isoformat()
        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info("開始評估，run_id=%s", run_id)

        # 載入測試集
        testset_data = self._load_testset()
        test_cases = testset_data.get("test_cases", [])
        testset_metadata = testset_data.get("metadata", {})

        # 套用過濾
        test_cases = self._filter_cases(test_cases)
        logger.info("評估 %d 個測試案例", len(test_cases))

        # 初始化 Pipeline
        pipeline = EvalPipeline.from_config(config)

        # 初始化指標
        metrics = self._load_metrics()

        # 初始化 judge LLM
        judge_llm = self._init_judge_llm() if config.judge_mode != "rule" else None

        # 執行
        case_results: list[CaseResult] = []
        if config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = [
                    executor.submit(
                        self._run_single_case, tc, pipeline, metrics, judge_llm
                    )
                    for tc in test_cases
                ]
                for f in futures:
                    case_results.append(f.result())
        else:
            for i, tc in enumerate(test_cases, 1):
                logger.info("[%d/%d] 執行：%s", i, len(test_cases), tc.get("id", "?"))
                cr = self._run_single_case(tc, pipeline, metrics, judge_llm)
                case_results.append(cr)
                if config.save_intermediate:
                    self._save_intermediate(cr, run_id)

        finished_at = datetime.now(timezone.utc).isoformat()

        # 聚合報告
        report = self._aggregate_report(
            case_results, run_id, started_at, finished_at, testset_metadata
        )
        logger.info(
            "評估完成：overall_avg=%.3f，%d/%d 案例成功",
            report.overall_average, report.successful_cases, report.total_cases,
        )
        return report

    # ── 內部方法 ─────────────────────────────────────────────────────────────

    def _load_testset(self) -> dict:
        path = self._config.testset_path
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _filter_cases(self, cases: list[dict]) -> list[dict]:
        config = self._config
        result = cases
        if config.filter_ids:
            result = [c for c in result if c.get("id") in config.filter_ids]
        if config.filter_difficulty:
            result = [c for c in result if c.get("difficulty") in config.filter_difficulty]
        if config.filter_tags:
            tag_set = set(config.filter_tags)
            result = [c for c in result if tag_set & set(c.get("tags", []))]
        return result

    def _load_metrics(self) -> list[BaseMetric]:
        from eval.metrics import METRIC_REGISTRY
        metrics = []
        for name in self._config.metrics:
            cls = METRIC_REGISTRY.get(name)
            if cls is None:
                logger.warning("未知指標：%s，跳過", name)
                continue
            metrics.append(cls())
        return metrics

    def _init_judge_llm(self):
        config = self._config
        try:
            from qa_Module.llm.factory import create_llm
            judge_llm = create_llm(
                provider=config.judge_llm_provider,
                model=config.judge_llm_model,
            )
            logger.info(
                "Judge LLM 初始化：%s/%s",
                config.judge_llm_provider, config.judge_llm_model,
            )
            return judge_llm
        except Exception as exc:
            logger.error("Judge LLM 初始化失敗：%s，改用規則模式", exc)
            return None

    def _run_single_case(
        self,
        test_case: dict,
        pipeline: EvalPipeline,
        metrics: list[BaseMetric],
        judge_llm,
    ) -> CaseResult:
        case_id = test_case.get("id", str(uuid4())[:8])
        query = test_case.get("query", "")
        query_type_expected = test_case.get("query_type", "auto")

        try:
            pr = pipeline.run(query)
        except Exception as exc:
            logger.error("案例 %s pipeline 執行失敗：%s", case_id, exc)
            return CaseResult(
                case_id=case_id,
                query=query,
                query_type_expected=query_type_expected,
                query_type_actual="unknown",
                answer="",
                is_fallback=False,
                retrieved_chunk_ids=[],
                cited_chunk_ids=[],
                error=str(exc),
            )

        cr = CaseResult(
            case_id=case_id,
            query=query,
            query_type_expected=query_type_expected,
            query_type_actual=pr.query_type,
            answer=pr.answer,
            is_fallback=pr.is_fallback,
            retrieved_chunk_ids=pr.retrieved_chunk_ids,
            cited_chunk_ids=pr.cited_chunk_ids,
            latency_ms=pr.latency_ms,
            error=pr.error,
        )

        if pr.error:
            return cr

        for metric in metrics:
            try:
                mr: MetricResult = metric.compute(pr, test_case, judge_llm)
                cr.metric_scores[mr.metric_name] = mr.score
                cr.metric_reasons[mr.metric_name] = mr.reason
                cr.metric_details[mr.metric_name] = mr.details
                if mr.passed:
                    cr.passed_metrics.append(mr.metric_name)
                else:
                    cr.failed_metrics.append(mr.metric_name)
            except Exception as exc:
                logger.error("指標 %s 計算失敗（案例 %s）：%s", metric.name, case_id, exc)
                cr.metric_scores[metric.name] = 0.0
                cr.metric_reasons[metric.name] = f"[ERROR] {exc}"
                cr.failed_metrics.append(metric.name)

        return cr

    def _save_intermediate(self, cr: CaseResult, run_id: str) -> None:
        out_dir = self._config.output_dir / run_id / "intermediate"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{cr.case_id}.json"
        data = {
            "case_id": cr.case_id,
            "query": cr.query,
            "answer": cr.answer,
            "is_fallback": cr.is_fallback,
            "query_type_actual": cr.query_type_actual,
            "retrieved_chunk_ids": cr.retrieved_chunk_ids,
            "cited_chunk_ids": cr.cited_chunk_ids,
            "metric_scores": cr.metric_scores,
            "metric_reasons": cr.metric_reasons,
            "latency_ms": cr.latency_ms,
            "error": cr.error,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _aggregate_report(
        self,
        case_results: list[CaseResult],
        run_id: str,
        started_at: str,
        finished_at: str,
        testset_metadata: dict,
    ) -> EvalReport:
        successful = [cr for cr in case_results if cr.error is None]
        failed = [cr for cr in case_results if cr.error is not None]

        # 各指標平均 / pass rate
        all_metric_names = set()
        for cr in successful:
            all_metric_names.update(cr.metric_scores.keys())

        metric_averages = {}
        metric_pass_rates = {}
        for name in all_metric_names:
            scores = [cr.metric_scores[name] for cr in successful if name in cr.metric_scores]
            passes = [cr for cr in successful
                      if name in cr.passed_metrics]
            metric_averages[name] = sum(scores) / len(scores) if scores else 0.0
            metric_pass_rates[name] = len(passes) / len(successful) if successful else 0.0

        overall_avg = (
            sum(cr.overall_score() for cr in successful) / len(successful)
            if successful else 0.0
        )

        # 分層統計
        by_difficulty = self._group_stats(successful, lambda cr, cases: cases[cr.case_id].get("difficulty"))
        by_query_type = self._group_by(successful, lambda cr: cr.query_type_actual)
        by_tag = self._group_by_tags(successful)

        return EvalReport(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            total_cases=len(case_results),
            successful_cases=len(successful),
            failed_cases=len(failed),
            case_results=case_results,
            metric_averages=metric_averages,
            metric_pass_rates=metric_pass_rates,
            overall_average=overall_avg,
            by_difficulty=by_difficulty,
            by_query_type=by_query_type,
            by_tag=by_tag,
            testset_metadata=testset_metadata,
        )

    def _group_by(
        self, case_results: list[CaseResult], key_fn
    ) -> dict[str, dict]:
        groups: dict[str, list[CaseResult]] = {}
        for cr in case_results:
            k = key_fn(cr) or "unknown"
            groups.setdefault(k, []).append(cr)
        return {
            k: {
                "count": len(crs),
                "avg_score": sum(cr.overall_score() for cr in crs) / len(crs),
            }
            for k, crs in groups.items()
        }

    def _group_by_tags(self, case_results: list[CaseResult]) -> dict[str, dict]:
        # 需要從測試集重新讀取 tags（CaseResult 未儲存 tags）
        # 簡化：回傳空 dict，tag 統計由 reporter 補充
        return {}

    def _group_stats(self, case_results, _) -> dict[str, dict]:
        # 簡化：不依賴測試集重查，直接回傳 by_query_type
        return {}
