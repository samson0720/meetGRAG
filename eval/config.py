"""
config.py
=========
EvalConfig：評估框架的所有執行參數集中在此 dataclass。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalConfig:
    # ── 必填 ────────────────────────────────────────────────────────────────
    testset_path: Path

    # ── 輸出 ────────────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("eval/results"))

    # ── RAG Pipeline 參數 ────────────────────────────────────────────────────
    top_k: int = 5
    score_cutoff: float = -0.5
    max_chunks: int = 5

    # ── 指標選擇 ─────────────────────────────────────────────────────────────
    metrics: list[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
        "citation_accuracy",
    ])

    # ── Judge LLM ────────────────────────────────────────────────────────────
    judge_mode: str = "llm"                       # "llm" | "rule" | "hybrid"
    judge_llm_provider: str = "groq"
    judge_llm_model: str = "llama-3.3-70b-versatile"

    # ── 執行控制 ─────────────────────────────────────────────────────────────
    max_workers: int = 1
    save_intermediate: bool = True
    report_formats: list[str] = field(default_factory=lambda: ["json", "csv"])

    # ── 過濾 ─────────────────────────────────────────────────────────────────
    filter_tags: list[str] = field(default_factory=list)
    filter_difficulty: list[str] = field(default_factory=list)
    filter_ids: list[str] = field(default_factory=list)

    # ── 系統路徑（繼承自主系統） ─────────────────────────────────────────────
    output_graphrag_dir: Path = field(
        default_factory=lambda: Path("qa_Module/graphrag/output")
    )
    lancedb_path: Path = field(
        default_factory=lambda: Path("qa_Module/graphrag/lancedb")
    )
    settings_path: Path = field(
        default_factory=lambda: Path("qa_Module/graphrag/settings.yaml")
    )

    def __post_init__(self):
        self.testset_path = Path(self.testset_path)
        self.output_dir = Path(self.output_dir)
        self.output_graphrag_dir = Path(self.output_graphrag_dir)
        self.lancedb_path = Path(self.lancedb_path)
        self.settings_path = Path(self.settings_path)
