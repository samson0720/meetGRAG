"""
reporter.py
===========
EvalReporter：將 EvalReport 序列化為 JSON / CSV，並在 terminal 印出摘要。

輸出檔案命名：eval/results/{run_id}_report.json、{run_id}_summary.csv
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.runner import EvalReport, CaseResult

logger = logging.getLogger(__name__)


class EvalReporter:

    @staticmethod
    def save(report: "EvalReport", output_dir: Path, formats: list[str]) -> list[Path]:
        """儲存報告，回傳所有已儲存的路徑列表。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        if "json" in formats:
            path = EvalReporter.to_json(report, output_dir)
            saved.append(path)
        if "csv" in formats:
            path = EvalReporter.to_csv(report, output_dir)
            saved.append(path)
        return saved

    @staticmethod
    def to_json(report: "EvalReport", output_dir: Path) -> Path:
        """輸出完整 JSON 報告。"""
        path = output_dir / f"{report.run_id}_report.json"
        data = {
            "run_id": report.run_id,
            "started_at": report.started_at,
            "finished_at": report.finished_at,
            "testset_metadata": report.testset_metadata,
            "summary": {
                "total_cases": report.total_cases,
                "successful_cases": report.successful_cases,
                "failed_cases": report.failed_cases,
                "overall_average": round(report.overall_average, 4),
                "metric_averages": {
                    k: round(v, 4) for k, v in report.metric_averages.items()
                },
                "metric_pass_rates": {
                    k: round(v, 4) for k, v in report.metric_pass_rates.items()
                },
                "by_query_type": report.by_query_type,
            },
            "cases": [EvalReporter._case_to_dict(cr) for cr in report.case_results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("JSON 報告已儲存：%s", path)
        return path

    @staticmethod
    def to_csv(report: "EvalReport", output_dir: Path) -> Path:
        """輸出摘要 CSV（每行一題）。"""
        path = output_dir / f"{report.run_id}_summary.csv"
        # 動態取得所有指標名稱
        all_metrics = sorted(report.metric_averages.keys())
        fieldnames = (
            ["case_id", "query", "query_type_expected", "query_type_actual",
             "overall_score", "is_fallback", "latency_ms", "error"]
            + all_metrics
            + ["retrieved_chunks_count", "cited_chunks_count"]
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for cr in report.case_results:
                row: dict = {
                    "case_id": cr.case_id,
                    "query": cr.query[:100],
                    "query_type_expected": cr.query_type_expected,
                    "query_type_actual": cr.query_type_actual,
                    "overall_score": round(cr.overall_score(), 4),
                    "is_fallback": cr.is_fallback,
                    "latency_ms": round(cr.latency_ms, 1),
                    "error": cr.error or "",
                    "retrieved_chunks_count": len(cr.retrieved_chunk_ids),
                    "cited_chunks_count": len(cr.cited_chunk_ids),
                }
                for m in all_metrics:
                    row[m] = round(cr.metric_scores.get(m, float("nan")), 4)
                writer.writerow(row)
        logger.info("CSV 摘要已儲存：%s", path)
        return path

    @staticmethod
    def print_summary(report: "EvalReport") -> None:
        """在 terminal 印出人類可讀的評估摘要。"""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  meetGRAG 評估報告  |  run_id: {report.run_id}")
        print(sep)
        print(f"  總案例數:  {report.total_cases}")
        print(f"  成功執行:  {report.successful_cases}")
        print(f"  執行失敗:  {report.failed_cases}")
        print(f"  整體平均:  {report.overall_average:.4f}")
        print()
        print("  各指標分數：")
        for name, avg in sorted(report.metric_averages.items()):
            pass_rate = report.metric_pass_rates.get(name, 0.0)
            print(f"    {name:<25} avg={avg:.4f}  pass_rate={pass_rate:.1%}")
        print()
        if report.by_query_type:
            print("  依 query_type 分層：")
            for qt, stats in report.by_query_type.items():
                print(f"    {qt:<10} count={stats['count']}  avg={stats['avg_score']:.4f}")
        print(f"{sep}\n")

    @staticmethod
    def _case_to_dict(cr: "CaseResult") -> dict:
        return {
            "case_id": cr.case_id,
            "query": cr.query,
            "query_type_expected": cr.query_type_expected,
            "query_type_actual": cr.query_type_actual,
            "answer": cr.answer[:500] if cr.answer else "",
            "is_fallback": cr.is_fallback,
            "overall_score": round(cr.overall_score(), 4),
            "latency_ms": round(cr.latency_ms, 1),
            "error": cr.error,
            "retrieved_chunk_ids": cr.retrieved_chunk_ids,
            "cited_chunk_ids": cr.cited_chunk_ids,
            "metric_scores": {k: round(v, 4) for k, v in cr.metric_scores.items()},
            "metric_reasons": cr.metric_reasons,
            "passed_metrics": cr.passed_metrics,
            "failed_metrics": cr.failed_metrics,
        }
