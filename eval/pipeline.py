"""
pipeline.py
===========
EvalPipeline：不依賴 FastAPI server，直接呼叫 qa_Module 各階段函數的評估用包裝。

使用方式
--------
  pipeline = EvalPipeline.from_config(config)
  result   = pipeline.run("Who chairs the ADD working group?")
  print(result.answer)
  print(result.retrieved_chunk_ids)
"""
from __future__ import annotations

import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

# 確保專案根目錄在 sys.path 中（讓 eval/ 可以獨立執行）
_EVAL_DIR = Path(__file__).resolve().parent
_ROOT = _EVAL_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if TYPE_CHECKING:
    from eval.config import EvalConfig


# ══════════════════════════════════════════════════════════════════════════════
# 資料類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """
    單次 RAG Pipeline 執行的完整快照。

    保留各階段的原始物件，供各評估指標直接存取細部資料，
    也提供常用屬性的便捷存取。
    """
    query: str
    query_type: str = "unknown"

    # 各階段原始物件（供指標深度分析）
    query_result: object = None        # QueryResult
    retrieval_context: object = None   # RetrievalContext
    organized_context: object = None   # OrganizedContext
    generated_answer: object = None    # GeneratedAnswer

    # 常用屬性（快速存取）
    answer: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_chunks_content: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    citations: list = field(default_factory=list)   # list[Citation]
    is_fallback: bool = False
    usage: dict = field(default_factory=dict)

    # 執行資訊
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ══════════════════════════════════════════════════════════════════════════════
# 主類別
# ══════════════════════════════════════════════════════════════════════════════

class EvalPipeline:
    """
    評估用 RAG Pipeline，直接呼叫 qa_Module 各階段函數。

    與 app.py 中的 _run_pipeline 邏輯相同，但不依賴 FastAPI 的 app.state，
    可獨立在命令列或評估腳本中執行。
    """

    def __init__(self, llm, searcher, config: "EvalConfig"):
        self._llm = llm
        self._searcher = searcher
        self._config = config

    @classmethod
    def from_config(cls, config: "EvalConfig") -> "EvalPipeline":
        """
        從 EvalConfig 初始化 LLM + GraphRAGSearcher。

        讀取 settings.yaml → 建立 LLM 客戶端 → 建立搜尋器。
        """
        from qa_Module.graphrag.vectorizer import _load_settings
        from qa_Module.llm.factory import create_llm_from_config
        from qa_Module.graphrag.searcher import GraphRAGSearcher

        settings = _load_settings(config.settings_path, config.settings_path.parent)
        llm_conf = settings.get("llm", {})
        llm = create_llm_from_config(dict(llm_conf))

        searcher = GraphRAGSearcher(
            output_dir=config.output_graphrag_dir,
            lancedb_path=config.lancedb_path,
            settings_path=config.settings_path,
        )
        logger.info("EvalPipeline 初始化完成，LLM: %s/%s",
                    llm_conf.get("provider"), llm_conf.get("model"))
        return cls(llm, searcher, config)

    def run(self, query: str) -> PipelineResult:
        """
        執行完整 RAG Pipeline，回傳各階段快照。

        發生錯誤時不拋出例外，而是在 PipelineResult.error 記錄錯誤訊息。
        """
        from qa_Module.query_processor import process_query
        from qa_Module.retriever import retrieve
        from qa_Module.organizer import organize
        from qa_Module.generator import generate

        t0 = time.monotonic()
        try:
            qr  = process_query(query, self._llm)
            ctx = retrieve(qr, self._searcher, top_k=self._config.top_k)
            org = organize(
                ctx,
                query=query,
                score_cutoff=self._config.score_cutoff,
                max_chunks=self._config.max_chunks,
            )
            ans = generate(org, query=query, llm=self._llm)

            return PipelineResult(
                query=query,
                query_type=qr.query_type,
                query_result=qr,
                retrieval_context=ctx,
                organized_context=org,
                generated_answer=ans,
                answer=ans.answer,
                retrieved_chunk_ids=[c.chunk_id for c in org.chunks],
                retrieved_chunks_content=[c.content for c in org.chunks],
                cited_chunk_ids=ans.cited_chunk_ids(),
                citations=ans.citations,
                is_fallback=ans.is_fallback,
                usage=ans.usage,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            logger.error("Pipeline 執行失敗：%s", exc, exc_info=True)
            return PipelineResult(
                query=query,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )
