"""
metrics/context_recall.py
=========================
Context Recall：相關資訊是否都被成功檢索到。

LLM judge 模式（預設，方式 2）
--------------------------------
  1. 將 expected_answer 分解為 M 個原子主張
  2. 對每個主張，判斷是否有任何 retrieved chunk 能支撐
  context_recall = (有 chunk 支撐的主張數) / M

規則模式
--------
  若 test_case["expected_chunks"] 存在：
    recall = |retrieved ∩ expected| / |expected|
  否則（使用 ground_truth_entities）：
    recall = |retrieved_entities ∩ ground_truth_entities| / |ground_truth_entities|
"""
from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from eval.metrics.base import BaseMetric, MetricResult
from prompts.context_recall import (
    VERIFY_CLAIM_SYSTEM,
    VERIFY_CLAIM_USER,
    DECOMPOSE_CLAIMS_SYSTEM,
)

if TYPE_CHECKING:
    from eval.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class ContextRecallMetric(BaseMetric):
    name = "context_recall"
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
        context_text = "\n\n".join(pipeline_result.retrieved_chunks_content)
        expected_answer = test_case.get("expected_answer", "")
        expected_chunks = test_case.get("expected_chunks", [])
        ground_truth_entities = test_case.get("ground_truth_entities", [])

        if judge_llm is not None and expected_answer:
            return self._llm_recall(expected_answer, context_text, judge_llm)
        else:
            return self._rule_recall(pipeline_result, expected_chunks, ground_truth_entities, org)

    def _llm_recall(
        self, expected_answer: str, context_text: str, judge_llm
    ) -> MetricResult:
        """分解 expected_answer 為原子主張，逐一驗證 context 是否涵蓋。"""
        # Step 1: 分解主張
        claims = self._decompose_claims(expected_answer, judge_llm)
        if not claims:
            return self._make_result(0.5, "could not decompose expected_answer into claims")

        if not context_text.strip():
            return self._make_result(0.0, "no retrieved context", {"claims_count": len(claims)})

        # Step 2: 逐一驗證
        scores = []
        details_list = []
        for claim in claims:
            user_prompt = VERIFY_CLAIM_USER.format(
                context=context_text[:2000],
                claim=claim,
            )
            score, reason = self._judge_yes_no(judge_llm, VERIFY_CLAIM_SYSTEM, user_prompt)
            scores.append(score)
            details_list.append({"claim": claim[:150], "score": score, "reason": reason[:200]})

        final = sum(scores) / len(scores)
        reason = f"{len(claims)} claims checked, recall={final:.2f}"
        return self._make_result(final, reason, {"claims": details_list})

    def _decompose_claims(self, text: str, judge_llm) -> list[str]:
        """將文字分解為原子主張。"""
        try:
            resp = judge_llm.complete(
                prompt=text,
                system=DECOMPOSE_CLAIMS_SYSTEM,
                temperature=0.0,
                max_tokens=512,
            )
            lines = resp.content.strip().split("\n")
            claims = []
            for line in lines:
                line = line.strip()
                if line.startswith("-"):
                    claim = line.lstrip("-").strip()
                    if claim:
                        claims.append(claim)
            return claims if claims else [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        except Exception as exc:
            logger.warning("主張分解失敗：%s", exc)
            return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

    def _rule_recall(
        self,
        pipeline_result: "PipelineResult",
        expected_chunks: list[str],
        ground_truth_entities: list[str],
        org,
    ) -> MetricResult:
        retrieved_ids = {c.chunk_id for c in org.chunks} if org else set()

        # 優先用 expected_chunks
        if expected_chunks:
            expected_set = set(expected_chunks)
            true_pos = retrieved_ids & expected_set
            score = len(true_pos) / len(expected_set) if expected_set else 0.0
            reason = (
                f"rule: |retrieved ∩ expected| / |expected| = "
                f"{len(true_pos)}/{len(expected_set)} = {score:.2f}"
            )
            return self._make_result(score, reason, {
                "retrieved": list(retrieved_ids),
                "expected": list(expected_set),
                "true_positives": list(true_pos),
            })

        # 其次用 ground_truth_entities
        if ground_truth_entities and org:
            retrieved_entities: set[str] = set()
            for chunk in org.chunks:
                for e in chunk.entities:
                    retrieved_entities.add(e.name.lower())

            gt_set = {e.lower() for e in ground_truth_entities}
            hits = retrieved_entities & gt_set
            score = len(hits) / len(gt_set) if gt_set else 0.0
            reason = (
                f"rule (entities): retrieved_entities ∩ ground_truth = "
                f"{len(hits)}/{len(gt_set)} = {score:.2f}"
            )
            return self._make_result(score, reason, {
                "retrieved_entities": list(retrieved_entities),
                "ground_truth": list(gt_set),
                "hits": list(hits),
            })

        return self._make_result(
            0.5,
            "rule: no expected_chunks or ground_truth_entities provided, score=0.5 (unknown)",
        )
