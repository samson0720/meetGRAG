"""
提示詞：測試集問題生成（eval/testset/builder.py）
"""

GENERATE_QUESTIONS_SYSTEM = (
    "You are creating test questions for a technical Q&A system about IETF meetings. "
    "Generate factual questions that can be answered from the provided transcript segment."
)

# 變數：{text}、{source_video}、{start_time}、{end_time}、{count}
GENERATE_QUESTIONS_USER = (
    "Transcript segment:\n{text}\n\n"
    "Source: {source_video}, {start_time}s - {end_time}s\n\n"
    "Generate {count} questions that:\n"
    "1. Can be DIRECTLY answered using only this segment\n"
    "2. Are factual (not opinion-based)\n"
    "3. Test different aspects (WHO, WHAT, HOW, WHY)\n\n"
    "For each question, provide a JSON object with:\n"
    '  "question": str\n'
    '  "expected_answer": str\n'
    '  "key_entities": list[str]\n'
    '  "query_type": "local" or "global"\n'
    '  "difficulty": "easy", "medium", or "hard"\n\n'
    "Output a JSON array of these objects. No extra text."
)
