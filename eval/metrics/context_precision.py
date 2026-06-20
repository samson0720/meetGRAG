"""
metrics/context_precision.py
============================
Context Precision：被檢索到的 chunks 中，有多少比例是真正相關的。

LLM judge 模式
--------------
  對每個 retrieved chunk，詢問 judge LLM 是否與 query 相關。
  context_precision = (相關 chunk 數) / len(retrieved_chunks)

規則模式
--------
  若 test_case["expected_chunks"] 有值：
    precision = |retrieved ∩ expected| / |retrieved|
  否則：
    precision = 有引用的 chunks / 全部 retrieved chunks
    （LLM 至少引用過的 chunk 視為相關）
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from eval.metrics.base import BaseMetric, MetricResult
from prompts.context_precision import RELEVANCE_SYSTEM, RELEVANCE_USER

if TYPE_CHECKING:
    from eval.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class ContextPrecisionMetric(BaseMetric):
    name = "context_precision"
    threshold = 0.5

    def compute(
        self,
        pipeline_result: "PipelineResult",
        test_case: dict,
        judge_llm=None,
    ) -> MetricResult:
        if not pipeline_result.success:
            return self._error_result(pipeline_result.error)

        org = pipeline_result.organized_context
        if org is None or not org.chunks:
            return self._make_result(0.0, "no retrieved chunks")

        query = pipeline_result.query
        chunks = org.chunks

        if judge_llm is not None:
            return self._llm_precision(query, chunks, judge_llm)
        else:
            return self._rule_precision(pipeline_result, test_case, chunks)

    def _llm_precision(self, query: str, chunks, judge_llm) -> MetricResult:
        scores = []
        details_list = []
        for chunk in chunks:
            user_prompt = RELEVANCE_USER.format(
                query=query,
                passage=chunk.content[:600],
            )
            score, reason = self._judge_yes_no(judge_llm, RELEVANCE_SYSTEM, user_prompt)
            scores.append(score)
            details_list.append({
                "chunk_id": chunk.chunk_id,
                "score": score,
                "reason": reason[:200],
            })

        final = sum(scores) / len(scores) if scores else 0.0
        reason = f"{len(chunks)} chunks evaluated, precision={final:.2f}"
        return self._make_result(final, reason, {"per_chunk": details_list})

    def _rule_precision(
        self, pipeline_result: "PipelineResult", test_case: dict, chunks
    ) -> MetricResult:
        expected_chunks = test_case.get("expected_chunks", [])
        retrieved_ids = {c.chunk_id for c in chunks}

        if expected_chunks:
            expected_set = set(expected_chunks)
            true_pos = retrieved_ids & expected_set
            score = len(true_pos) / len(retrieved_ids) if retrieved_ids else 0.0
            reason = (
                f"rule: |retrieved ∩ expected| / |retrieved| = "
                f"{len(true_pos)}/{len(retrieved_ids)} = {score:.2f}"
            )
            return self._make_result(score, reason, {
                "retrieved": list(retrieved_ids),
                "expected": list(expected_set),
                "true_positives": list(true_pos),
            })
        else:
            # 以 LLM 實際引用的 chunks 作為相關代理
            cited_ids = set(pipeline_result.cited_chunk_ids)
            if not cited_ids and pipeline_result.is_fallback:
                return self._make_result(1.0, "is_fallback, no context needed")
            relevant_count = len(retrieved_ids & cited_ids)
            score = relevant_count / len(retrieved_ids) if retrieved_ids else 0.0
            reason = (
                f"rule (proxy): cited chunks / retrieved chunks = "
                f"{relevant_count}/{len(retrieved_ids)} = {score:.2f} "
                f"(no expected_chunks provided)"
            )
            return self._make_result(score, reason, {
                "retrieved": list(retrieved_ids),
                "cited": list(cited_ids),
            })
