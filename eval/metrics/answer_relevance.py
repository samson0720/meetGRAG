"""
metrics/answer_relevance.py
===========================
Answer Relevance：答案是否切題地回答了問題。

LLM judge 模式
--------------
  直接用 judge LLM 評分（0-10），判斷答案是否完整且直接地回應問題。

規則模式
--------
  統計 QueryResult.core_concepts 中有多少在 answer 裡出現，
  再加上基本的 query token 重疊率作為混合分數。
"""
from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from eval.metrics.base import BaseMetric, MetricResult
from prompts.answer_relevance import RELEVANCE_SYSTEM, RELEVANCE_USER

if TYPE_CHECKING:
    from eval.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class AnswerRelevanceMetric(BaseMetric):
    name = "answer_relevance"
    threshold = 0.6

    def compute(
        self,
        pipeline_result: "PipelineResult",
        test_case: dict,
        judge_llm=None,
    ) -> MetricResult:
        if not pipeline_result.success:
            return self._error_result(pipeline_result.error)

        if pipeline_result.is_fallback:
            # Fallback 回應難以評估相關性，給予中性分數
            return self._make_result(0.5, "is_fallback=True, relevance undefined")

        answer_plain = re.sub(r'\[REF:[^\]]+\]', '', pipeline_result.answer).strip()
        query = pipeline_result.query

        if not answer_plain:
            return self._make_result(0.0, "empty answer")

        if judge_llm is not None:
            return self._llm_relevance(query, answer_plain, judge_llm)
        else:
            return self._rule_relevance(query, answer_plain, pipeline_result)

    def _llm_relevance(
        self, query: str, answer: str, judge_llm
    ) -> MetricResult:
        user_prompt = RELEVANCE_USER.format(query=query, answer=answer)
        score, reason = self._judge_score(judge_llm, RELEVANCE_SYSTEM, user_prompt)
        return self._make_result(score, reason)

    def _rule_relevance(
        self, query: str, answer: str, pipeline_result: "PipelineResult"
    ) -> MetricResult:
        scores = []
        details = {}

        # 1. query token 覆蓋率
        q_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                     "how", "when", "where", "which", "that", "this", "of", "in",
                     "to", "for", "with", "on", "at", "by", "from", "and", "or"}
        q_content = q_tokens - stopwords
        if q_content:
            coverage = len(q_content & a_tokens) / len(q_content)
            scores.append(coverage)
            details["query_token_coverage"] = coverage

        # 2. entities 命中率
        qr = pipeline_result.query_result
        if qr is not None and hasattr(qr, "entities") and qr.entities:
            concepts = [c.lower() for c in qr.entities]
            answer_lower = answer.lower()
            hits = sum(1 for c in concepts if c in answer_lower)
            concept_score = hits / len(concepts)
            scores.append(concept_score)
            details["concept_coverage"] = concept_score
            details["concepts"] = qr.entities

        final = sum(scores) / len(scores) if scores else 0.5
        reason = f"rule-based: avg({', '.join(f'{v:.2f}' for v in scores)}) = {final:.2f}"
        return self._make_result(final, reason, details)
