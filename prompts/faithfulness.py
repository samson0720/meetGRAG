"""
提示詞：忠實性評估（eval/metrics/faithfulness.py）
"""

# ── 主張驗證 ────────────────────────────────────────────────────────────────

VERIFY_CLAIM_SYSTEM = (
    "You are a factual verification expert. "
    "Your task is to determine if a claim is supported by the given context."
)

# 變數：{context}、{claim}
VERIFY_CLAIM_USER = (
    "Context:\n{context}\n\n"
    "Claim: {claim}\n\n"
    "Is this claim fully supported by the context? "
    "Answer YES, PARTIAL, or NO. Then give a one-sentence reason."
)

# ── 主張分解 ────────────────────────────────────────────────────────────────

DECOMPOSE_CLAIMS_SYSTEM = (
    "You are an expert at breaking down text into atomic factual claims. "
    "Each claim should be a single, verifiable statement."
)

# 變數：{answer}
DECOMPOSE_CLAIMS_USER = (
    "Break down the following answer into a list of atomic factual claims. "
    "Output each claim on a new line, prefixed with '- '. "
    "Do not include opinions or uncertainties.\n\n"
    "Answer:\n{answer}"
)
