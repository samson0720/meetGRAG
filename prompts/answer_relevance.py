"""
提示詞：答案相關性評估（eval/metrics/answer_relevance.py）
"""

RELEVANCE_SYSTEM = (
    "You are an expert evaluator assessing whether an answer directly and "
    "completely addresses a given question."
)

# 變數：{query}、{answer}
RELEVANCE_USER = (
    "Question: {query}\n\n"
    "Answer: {answer}\n\n"
    "On a scale of 0 to 10, how well does this answer directly and completely "
    "address the question? "
    "Consider: directness, completeness, and relevance. "
    "Reply with just the score (0-10) followed by a brief reason."
)
