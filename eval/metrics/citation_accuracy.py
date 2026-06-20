"""
metrics/citation_accuracy.py
============================
Citation Accuracy：檢驗 [REF:xxx] 引用的三個層面。

  (a) 語法準確率  — [REF:xxx] 能否被成功解析為 Citation 物件（規則型）
  (b) 語意準確率  — 引用的 chunk 內容是否真的支撐對應的句子（LLM judge）
  (c) 來源匹配率  — 引用的 source_video 是否與 expected 相符（規則型）

最終分數 = 0.4 × syntax + 0.4 × semantic + 0.2 × source_match
規則模式（無 LLM）：= 0.6 × syntax + 0.4 × source_match
"""
from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from eval.metrics.base import BaseMetric, MetricResult
from prompts.citation_accuracy import VERIFY_CITATION_SYSTEM, VERIFY_CITATION_USER

if TYPE_CHECKING:
    from eval.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class CitationAccuracyMetric(BaseMetric):
    name = "citation_accuracy"
    threshold = 0.6

    def compute(
        self,
        pipeline_result: "PipelineResult",
        test_case: dict,
        judge_llm=None,
    ) -> MetricResult:
        if not pipeline_result.success:
            return self._error_result(pipeline_result.error)

        answer = pipeline_result.answer
        citations = pipeline_result.citations
        org = pipeline_result.organized_context

        # ── (a) 語法準確率 ────────────────────────────────────────────────
        syntax_score, syntax_detail = self._syntax_accuracy(answer, citations)

        # ── (b) 語意準確率 ────────────────────────────────────────────────
        if judge_llm is not None and citations and org is not None:
            semantic_score, semantic_detail = self._semantic_accuracy(
                answer, citations, org, judge_llm
            )
        else:
            semantic_score = syntax_score   # 無 LLM 時以語法分數近似
            semantic_detail = {"note": "skipped (no judge_llm or no citations)"}

        # ── (c) 來源匹配率 ────────────────────────────────────────────────
        expected_videos = test_case.get("relevant_source_videos", [])
        source_score, source_detail = self._source_match(citations, expected_videos)

        # ── 加權合計 ─────────────────────────────────────────────────────
        if judge_llm is not None:
            final = 0.4 * syntax_score + 0.4 * semantic_score + 0.2 * source_score
        else:
            final = 0.6 * syntax_score + 0.4 * source_score

        reason = (
            f"syntax={syntax_score:.2f} semantic={semantic_score:.2f} "
            f"source={source_score:.2f} → final={final:.2f}"
        )
        details = {
            "syntax_score": syntax_score,
            "semantic_score": semantic_score,
            "source_score": source_score,
            "syntax_detail": syntax_detail,
            "semantic_detail": semantic_detail,
            "source_detail": source_detail,
        }
        return self._make_result(final, reason, details)

    # ── 語法準確率 ──────────────────────────────────────────────────────────

    def _syntax_accuracy(self, answer: str, citations: list) -> tuple[float, dict]:
        """計算 [REF:xxx] 的語法解析成功率。"""
        # 從 answer 中提取所有 [REF:xxx] 標記
        all_refs = re.findall(r'\[REF:([^\]]+)\]', answer)
        total = len(all_refs)
        if total == 0:
            # 沒有引用標記——若有 citations 物件則視為部分正確（generator 可能已清理）
            score = 0.5 if citations else 0.0
            return score, {"total_refs": 0, "valid_refs": len(citations), "note": "no REF tags found"}

        valid_ids = {c.chunk_id for c in citations}
        valid_count = sum(1 for r in all_refs if r.strip() in valid_ids)
        score = valid_count / total
        return score, {
            "total_refs": total,
            "valid_refs": valid_count,
            "all_refs": all_refs,
        }

    # ── 語意準確率 ──────────────────────────────────────────────────────────

    def _semantic_accuracy(
        self, answer: str, citations: list, org, judge_llm
    ) -> tuple[float, dict]:
        """用 judge LLM 判斷每筆引用是否語意支撐對應句子。"""
        # 建立 chunk_id → content 的快速查找
        chunk_map = {c.chunk_id: c.content for c in org.chunks}

        scores = []
        details_list = []

        # 將 answer 按句子切分，找出各句引用的 chunk
        sentences = self._split_by_citation(answer)

        for sent_text, ref_ids in sentences:
            if not ref_ids:
                continue
            for ref_id in ref_ids:
                chunk_content = chunk_map.get(ref_id, "")
                if not chunk_content:
                    scores.append(0.0)
                    details_list.append({"ref_id": ref_id, "score": 0.0, "reason": "chunk not found"})
                    continue

                user_prompt = VERIFY_CITATION_USER.format(
                    statement=sent_text.strip(),
                    passage=chunk_content[:600],
                )
                score, reason = self._judge_yes_no(judge_llm, VERIFY_CITATION_SYSTEM, user_prompt)
                scores.append(score)
                details_list.append({"ref_id": ref_id, "score": score, "reason": reason[:200]})

        if not scores:
            return 1.0, {"note": "no citations to verify"}

        avg = sum(scores) / len(scores)
        return avg, {"per_citation": details_list, "average": avg}

    def _split_by_citation(self, answer: str) -> list[tuple[str, list[str]]]:
        """
        將 answer 依句子切分，回傳 [(sentence_text, [ref_ids])] 列表。
        每個句子含其後緊接的 [REF:xxx] 標記。
        """
        pattern = re.compile(r'([^.!?]*[.!?]?)\s*((?:\[REF:[^\]]+\]\s*)*)')
        results = []
        for m in pattern.finditer(answer):
            sent = m.group(1).strip()
            refs_str = m.group(2)
            if not sent and not refs_str:
                continue
            ref_ids = re.findall(r'\[REF:([^\]]+)\]', refs_str)
            if sent or ref_ids:
                results.append((sent, ref_ids))
        return results

    # ── 來源匹配率 ──────────────────────────────────────────────────────────

    def _source_match(
        self, citations: list, expected_videos: list[str]
    ) -> tuple[float, dict]:
        """計算引用的影片來源與期望來源的重疊率。"""
        if not expected_videos:
            return 1.0, {"note": "no expected_videos provided, skipped"}

        cited_videos: set[str] = set()
        for citation in citations:
            for ref in citation.source_refs:
                cited_videos.add(ref.source_video)

        expected_set = set(expected_videos)
        hits = cited_videos & expected_set
        score = len(hits) / len(expected_set) if expected_set else 1.0
        return score, {
            "expected": list(expected_set),
            "cited": list(cited_videos),
            "hits": list(hits),
        }
