"""
提示詞：上下文精確度評估（eval/metrics/context_precision.py）
"""

RELEVANCE_SYSTEM = (
    "You are a relevance judge. "
    "Determine if a text passage is relevant to answering the given question."
)

# 變數：{query}、{passage}
RELEVANCE_USER = (
    "Question: {query}\n\n"
    "Passage:\n{passage}\n\n"
    "Is this passage relevant to answering the question? "
    "Answer YES, PARTIAL, or NO. Give a one-sentence reason."
)
