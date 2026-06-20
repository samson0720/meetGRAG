"""
testset/builder.py
==================
TestSetBuilder：測試集建立工具，支援手動輸入與半自動 LLM 生成。

使用方式（半自動）
------------------
  builder = TestSetBuilder(output_graphrag_dir=Path("qa_Module/graphrag/output"))
  builder.generate_from_graph(n_samples=10, llm=llm)
  builder.save(Path("eval/testset/my_testset.json"))

使用方式（手動）
----------------
  builder = TestSetBuilder()
  builder.add_case(
      query="Who chairs the IETF ADD working group?",
      expected_answer="David Lawrence and Glenn Deen.",
      query_type="local",
      ground_truth_entities=["David Lawrence", "Glenn Deen"],
      tags=["person", "easy"],
      difficulty="easy",
  )
  builder.save(Path("eval/testset/my_testset.json"))
"""
from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prompts.testset_builder import GENERATE_QUESTIONS_SYSTEM, GENERATE_QUESTIONS_USER

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestSetBuilder:
    """
    測試集建立器。

    Parameters
    ----------
    output_graphrag_dir
        GraphRAG output 目錄（含 text_units.parquet 等）。
    description
        測試集描述，存入 metadata。
    """

    def __init__(
        self,
        output_graphrag_dir: Path = Path("qa_Module/graphrag/output"),
        description: str = "meetGRAG evaluation testset",
    ):
        self._output_dir = Path(output_graphrag_dir)
        self._description = description
        self._cases: list[dict] = []
        self._source_videos: set[str] = set()

    # ── 公開方法 ─────────────────────────────────────────────────────────────

    def add_case(
        self,
        query: str,
        expected_answer: str,
        query_type: str = "auto",
        expected_chunks: Optional[list[str]] = None,
        relevant_source_videos: Optional[list[str]] = None,
        ground_truth_entities: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        difficulty: str = "medium",
        notes: str = "",
    ) -> str:
        """手動新增一個測試案例，回傳自動生成的 case_id。"""
        case_id = f"case_{len(self._cases) + 1:04d}"
        case = {
            "id": case_id,
            "query": query,
            "query_type": query_type,
            "expected_answer": expected_answer,
            "expected_chunks": expected_chunks or [],
            "relevant_source_videos": relevant_source_videos or [],
            "ground_truth_entities": ground_truth_entities or [],
            "tags": tags or [],
            "difficulty": difficulty,
            "notes": notes,
        }
        self._cases.append(case)
        if relevant_source_videos:
            self._source_videos.update(relevant_source_videos)
        return case_id

    def generate_from_graph(
        self,
        n_samples: int,
        llm,
        source_video_filter: Optional[str] = None,
        questions_per_unit: int = 2,
        interactive_review: bool = True,
    ) -> list[dict]:
        """
        半自動生成測試案例。

        從 text_units.parquet 隨機抽取樣本，用 LLM 生成問題+預期答案，
        可選擇性地進行人工審核。

        Parameters
        ----------
        n_samples           最多從 text_units 中抽取幾個樣本
        llm                 LLM 客戶端（用於生成問題）
        source_video_filter 若指定，只從此影片的 text_units 中抽取
        questions_per_unit  每個 text_unit 生成幾個問題
        interactive_review  是否啟用互動式審核（逐題確認）

        Returns
        -------
        新增的測試案例列表
        """
        text_units = self._load_text_units(source_video_filter)
        if not text_units:
            logger.warning("沒有找到 text_units，請先建立索引")
            return []

        # 隨機抽樣
        sample_size = min(n_samples, len(text_units))
        samples = random.sample(text_units, sample_size)
        logger.info("從 %d 個 text_units 中抽取 %d 個樣本", len(text_units), sample_size)

        new_cases = []
        for i, tu in enumerate(samples, 1):
            logger.info("[%d/%d] 生成問題：%s", i, sample_size, tu.get("id"))
            generated = self._generate_questions(tu, llm, questions_per_unit)
            if not generated:
                continue

            for q_data in generated:
                if interactive_review:
                    approved = self._interactive_review(tu, q_data)
                    if not approved:
                        continue

                case_id = f"auto_{tu.get('id', 'unk')[:8]}_{len(self._cases)+1:03d}"
                case = {
                    "id": case_id,
                    "query": q_data.get("question", ""),
                    "query_type": q_data.get("query_type", "local"),
                    "expected_answer": q_data.get("expected_answer", ""),
                    "expected_chunks": [],
                    "relevant_source_videos": [tu.get("source_video", "")] if tu.get("source_video") else [],
                    "ground_truth_entities": q_data.get("key_entities", []),
                    "tags": ["auto-generated"],
                    "difficulty": q_data.get("difficulty", "medium"),
                    "notes": f"generated from text_unit {tu.get('id')}",
                }
                self._cases.append(case)
                new_cases.append(case)
                if tu.get("source_video"):
                    self._source_videos.add(tu["source_video"])

        logger.info("半自動生成完成，新增 %d 個測試案例", len(new_cases))
        return new_cases

    def save(self, path: Path) -> None:
        """將測試集儲存為 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_videos": sorted(self._source_videos),
                "description": self._description,
                "total_cases": len(self._cases),
            },
            "test_cases": self._cases,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("測試集已儲存：%s（%d 個案例）", path, len(self._cases))

    # ── 內部方法 ─────────────────────────────────────────────────────────────

    def _load_text_units(self, source_video_filter: Optional[str]) -> list[dict]:
        try:
            from qa_Module.graphrag.storage import load_table
            units = load_table("text_units", self._output_dir)
            if source_video_filter:
                units = [u for u in units if source_video_filter in u.get("source_video", "")]
            # 過濾掉文字太短的 unit
            units = [u for u in units if len(u.get("text", "")) > 100]
            return units
        except Exception as exc:
            logger.error("載入 text_units 失敗：%s", exc)
            return []

    def _generate_questions(self, text_unit: dict, llm, count: int) -> list[dict]:
        """用 LLM 從 text_unit 生成問題。"""
        text = text_unit.get("text", "")
        source_video = text_unit.get("source_video", "unknown")
        start_time = text_unit.get("start_time", 0)
        end_time = text_unit.get("end_time", 0)

        try:
            resp = llm.complete(
                prompt=GENERATE_QUESTIONS_USER.format(
                    text=text[:800],
                    source_video=source_video,
                    start_time=f"{start_time:.0f}",
                    end_time=f"{end_time:.0f}",
                    count=count,
                ),
                system=GENERATE_QUESTIONS_SYSTEM,
                temperature=0.3,
                max_tokens=1024,
            )
            import re
            # 提取 JSON 陣列
            content = resp.content.strip()
            m = re.search(r'\[.*\]', content, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return json.loads(content)
        except Exception as exc:
            logger.warning("問題生成失敗（unit %s）：%s", text_unit.get("id"), exc)
            return []

    def _interactive_review(self, text_unit: dict, q_data: dict) -> bool:
        """互動式審核：印出問題讓使用者確認。"""
        print("\n" + "─" * 50)
        print(f"Source: {text_unit.get('source_video')} "
              f"({text_unit.get('start_time', 0):.0f}s - {text_unit.get('end_time', 0):.0f}s)")
        print(f"Text: {text_unit.get('text', '')[:200]}...")
        print()
        print(f"Q: {q_data.get('question')}")
        print(f"A: {q_data.get('expected_answer')}")
        print(f"Entities: {q_data.get('key_entities')}")
        print(f"Type: {q_data.get('query_type')}  Difficulty: {q_data.get('difficulty')}")
        print()
        choice = input("加入測試集？[Y/n/e(dit)] ").strip().lower()
        if choice in ("", "y"):
            return True
        if choice == "n":
            return False
        if choice == "e":
            q_data["question"] = input(f"修改 question [{q_data['question']}]: ").strip() or q_data["question"]
            q_data["expected_answer"] = input(f"修改 expected_answer [{q_data['expected_answer']}]: ").strip() or q_data["expected_answer"]
            return True
        return False
