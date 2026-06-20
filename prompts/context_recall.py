"""
提示詞：上下文回溯率評估（eval/metrics/context_recall.py）
"""

# ── 主張驗證 ────────────────────────────────────────────────────────────────

VERIFY_CLAIM_SYSTEM = (
    "You are a factual verification expert. "
    "Determine if a claim can be found or inferred from the given context."
)

# 變數：{context}、{claim}
VERIFY_CLAIM_USER = (
    "Context:\n{context}\n\n"
    "Claim: {claim}\n\n"
    "Can this claim be found or directly inferred from the context? "
    "Answer YES, PARTIAL, or NO. Give a one-sentence reason."
)

# ── 主張分解 ────────────────────────────────────────────────────────────────

DECOMPOSE_CLAIMS_SYSTEM = (
    "Break down the following text into atomic factual claims, "
    "one per line, prefixed with '- '."
)
