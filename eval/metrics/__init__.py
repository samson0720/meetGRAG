"""
metrics — 評估指標模組

METRIC_REGISTRY 對應名稱 → 指標類別，供 EvalRunner 動態載入。
"""
from eval.metrics.faithfulness import FaithfulnessMetric
from eval.metrics.answer_relevance import AnswerRelevanceMetric
from eval.metrics.context_precision import ContextPrecisionMetric
from eval.metrics.context_recall import ContextRecallMetric
from eval.metrics.citation_accuracy import CitationAccuracyMetric

METRIC_REGISTRY: dict[str, type] = {
    "faithfulness":       FaithfulnessMetric,
    "answer_relevance":   AnswerRelevanceMetric,
    "context_precision":  ContextPrecisionMetric,
    "context_recall":     ContextRecallMetric,
    "citation_accuracy":  CitationAccuracyMetric,
}

__all__ = [
    "METRIC_REGISTRY",
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "CitationAccuracyMetric",
]
