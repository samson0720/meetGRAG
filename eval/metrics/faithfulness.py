"""
metrics/faithfulness.py
=======================
Faithfulness：答案是否忠實於已檢索到的 context，不捏造資訊。

LLM judge 模式
--------------
  1. 將 answer（去除 [REF:xxx]）分解為 N 個原子主張（atomic claims）
  2. 對每個主張，詢問 judge LLM 是否有 retrieved context 支撐
  3. faithfulness = (YES 數 + 0.5 × PARTIAL 數) / N
  特例：is_fallback=True → score=1.0（系統誠實說不知道）

規則模式（fallback）
--------------------
  計算 answer token 與所有 retrieved chunks content 的 token 重疊率（Jaccard）。
"""
from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from eval.metrics.base import BaseMetric, MetricResult
from prompts.faithfulness import (
    VERIFY_CLAIM_SYSTEM,
    VERIFY_CLAIM_USER,
    DECOMPOSE_CLAIMS_SYSTEM,
    DECOMPOSE_CLAIMS_USER,
)

if TYPE_CHECKING:
    from eval.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class FaithfulnessMetric(BaseMetric):
    name = "faithfulness"
    threshold = 0.7

    def compute(
        self,
        pipeline_result: "PipelineResult",
        test_case: dict,
        judge_llm=None,
    ) -> MetricResult:
        if not pipeline_result.success:
            return self._error_result(pipeline_result.error)

        # Fallback 回應（系統說不知道）視為忠實
        if pipeline_result.is_fallback:
            return self._make_result(1.0, "is_fallback=True, system correctly abstained")

        answer_plain = self._strip_refs(pipeline_result.answer)
        context_text = "\n\n".join(pipeline_result.retrieved_chunks_content)

        if not answer_plain.strip():
            return self._make_result(1.0, "empty answer")

        if not context_text.strip():
            return self._make_result(0.0, "no retrieved context to verify against")

        if judge_llm is not None:
            return self._llm_faithfulness(answer_plain, context_text, judge_llm)
        else:
            return self._rule_faithfulness(answer_plain, context_text)

    def _llm_faithfulness(
        self, answer_plain: str, context_text: str, judge_llm
    ) -> MetricResult:
        # Step 1: 分解為原子主張
        claims = self._decompose_claims(answer_plain, judge_llm)
        if not claims:
            return self._make_result(0.5, "could not decompose answer into claims")

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
        reason = f"{len(claims)} claims verified, avg={final:.2f}"
        return self._make_result(final, reason, {"claims": details_list})

    def _decompose_claims(self, answer: str, judge_llm) -> list[str]:
        """用 judge LLM 將 answer 分解為原子主張列表。"""
        try:
            resp = judge_llm.complete(
                prompt=DECOMPOSE_CLAIMS_USER.format(answer=answer),
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
            return claims if claims else [answer[:300]]
        except Exception as exc:
            logger.warning("主張分解失敗：%s", exc)
            # fallback：按句子切分
            return [s.strip() for s in re.split(r'[.!?]', answer) if s.strip()]

    def _rule_faithfulness(self, answer: str, context_text: str) -> MetricResult:
        """規則型：計算 answer 與 context 的 token 重疊率（Jaccard）。"""
        a_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
        c_tokens = set(re.findall(r'\b\w+\b', context_text.lower()))
        if not a_tokens:
            return self._make_result(1.0, "empty answer tokens")
        intersection = a_tokens & c_tokens
        union = a_tokens | c_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        # Jaccard 偏低但不代表不忠實（答案措辭可能不同），映射到較寬鬆的分數
        score = min(1.0, jaccard * 3.0)
        reason = (
            f"rule-based Jaccard overlap: {jaccard:.3f} → score={score:.2f} "
            f"(answer tokens={len(a_tokens)}, context tokens={len(c_tokens)}, "
            f"overlap={len(intersection)})"
        )
        return self._make_result(score, reason, {
            "jaccard": jaccard,
            "answer_token_count": len(a_tokens),
            "context_token_count": len(c_tokens),
            "overlap_count": len(intersection),
        })

    @staticmethod
    def _strip_refs(text: str) -> str:
        return re.sub(r'\[REF:[^\]]+\]', '', text).strip()
