"""
提示詞：引用準確度評估（eval/metrics/citation_accuracy.py）
"""

VERIFY_CITATION_SYSTEM = (
    "You are a factual verification expert. "
    "Determine if a cited passage supports a given statement."
)

# 變數：{statement}、{passage}
VERIFY_CITATION_USER = (
    "Statement: {statement}\n\n"
    "Cited passage:\n{passage}\n\n"
    "Does the cited passage directly support or substantiate the statement? "
    "Answer with YES, PARTIAL, or NO, then give a one-sentence reason."
)
