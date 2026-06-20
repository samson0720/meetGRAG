"""
metrics/base.py
===============
BaseMetric 抽象基類 + MetricResult 資料結構。

所有評估指標都繼承 BaseMetric，實作 compute() 方法，
回傳 MetricResult（score 0.0~1.0）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from eval.pipeline import PipelineResult


@dataclass
class MetricResult:
    """
    單個指標對單個測試案例的評估結果。

    Attributes
    ----------
    metric_name  指標名稱
    score        0.0~1.0 的評分
    passed       score >= threshold
    reason       LLM judge 的解釋或規則說明（人類可讀）
    details      細節資料，例如每個 claim 或 chunk 的子分數
    """
    metric_name: str
    score: float
    passed: bool
    reason: str = ""
    details: Optional[dict] = None

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))


class BaseMetric(ABC):
    """所有評估指標的抽象基類。"""

    name: str = ""
    threshold: float = 0.5

    @abstractmethod
    def compute(
        self,
        pipeline_result: "PipelineResult",
        test_case: dict,
        judge_llm=None,
    ) -> MetricResult:
        """
        計算單筆測試案例的指標分數。

        Parameters
        ----------
        pipeline_result
            EvalPipeline.run() 的回傳值，含各階段中間物件。
        test_case
            測試集中的單筆 dict，含 expected_answer 等 ground truth。
        judge_llm
            LLM-as-judge 客戶端（BaseLLMClient）。
            若 judge_mode="rule" 可傳 None。
        """

    def _make_result(
        self,
        score: float,
        reason: str = "",
        details: Optional[dict] = None,
    ) -> MetricResult:
        """便利方法：依 threshold 自動判斷 passed。"""
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            reason=reason,
            details=details,
        )

    def _error_result(self, error_msg: str) -> MetricResult:
        """發生錯誤時回傳 score=0.0 的結果。"""
        return MetricResult(
            metric_name=self.name,
            score=0.0,
            passed=False,
            reason=f"[ERROR] {error_msg}",
        )

    def _judge_yes_no(
        self,
        judge_llm,
        system_prompt: str,
        user_prompt: str,
        partial_value: float = 0.5,
    ) -> tuple[float, str]:
        """
        用 judge LLM 回答 YES / PARTIAL / NO 問題，回傳 (score, reason)。

        Returns
        -------
        (score, reason)
            score: YES=1.0, PARTIAL=partial_value, NO=0.0
        """
        try:
            resp = judge_llm.complete(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.0,
                max_tokens=256,
            )
            content = resp.content.strip()
            upper = content.upper()
            if upper.startswith("YES"):
                return 1.0, content
            elif upper.startswith("PARTIAL"):
                return partial_value, content
            elif upper.startswith("NO"):
                return 0.0, content
            # 若回應不符合格式，嘗試從內容判斷
            if "YES" in upper:
                return 1.0, content
            if "NO" in upper:
                return 0.0, content
            return partial_value, content
        except Exception as exc:
            return partial_value, f"[judge error] {exc}"

    def _judge_score(
        self,
        judge_llm,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[float, str]:
        """
        用 judge LLM 回答 0-10 分問題，回傳 (score 0.0~1.0, reason)。
        """
        try:
            resp = judge_llm.complete(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.0,
                max_tokens=256,
            )
            content = resp.content.strip()
            import re
            m = re.search(r'\b(\d+(?:\.\d+)?)\b', content)
            if m:
                raw = float(m.group(1))
                score = min(raw / 10.0, 1.0) if raw > 1.0 else raw
                return score, content
            return 0.5, content
        except Exception as exc:
            return 0.5, f"[judge error] {exc}"
