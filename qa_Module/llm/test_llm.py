"""
LLM client smoke test
=====================
執行方式：
    python -m qa_Module.llm.test_llm                  # 測試所有已設定的 provider
    python -m qa_Module.llm.test_llm --provider ollama
    python -m qa_Module.llm.test_llm --provider groq
    python -m qa_Module.llm.test_llm --provider openai

或直接執行（在 meetGRAG 根目錄）：
    python qa_Module/llm/test_llm.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import os
from dotenv import load_dotenv

# ── 根目錄加入 sys.path，讓直接執行時也能 import ─────────────────────────────
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qa_Module.llm import LLMResponse, Message, create_llm
from prompts.indexer import EXTRACT_SYSTEM as _EXTRACT_SYSTEM, EXTRACT_USER as _EXTRACT_USER

# 取得API KEY
load_dotenv()
OPENAPI_API_KEY = os.getenv("OPENAPI_API_KEY", "default_key_if_not_found")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "default_key_if_not_found")

# ══════════════════════════════════════════════════════════════════════════════
# Provider 設定（按需填入 API key；不想測的項目設 enabled=False）
# ══════════════════════════════════════════════════════════════════════════════

PROVIDERS: list[dict] = [
    {
        "name": "ollama",
        "enabled": False,
        "kwargs": {
            "model": "mistral",
            "base_url": "http://localhost:11434/v1",
        },
    },
    {
        "name": "groq",
        "enabled": True,          # 填入 api_key 後改為 True
        "kwargs": {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "api_key": GROQ_API_KEY,
        },
    },
    {
        "name": "openai",
        "enabled": False,          # 填入 api_key 後改為 True
        "kwargs": {
            "model": "gpt-4o-mini",
            "api_key": OPENAPI_API_KEY,
        },
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# 測試案例
# ══════════════════════════════════════════════════════════════════════════════

TEST_PROMPT = "用中文說明:Reply with exactly one sentence: What is QUIC?"
SYSTEM_PROMPT = "You are a networking expert. Be concise."

CHAT_MESSAGES = [
    Message("system", "You are a networking expert. Be concise."),
    Message("user", "What problem does HTTP/3 solve compared to HTTP/2?"),
]


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def _header(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def run_test(provider_name: str, kwargs: dict) -> bool:
    """Run all sub-tests for one provider. Returns True if all pass."""
    try:
        llm = create_llm(provider_name, **kwargs)
    except Exception as exc:
        print(_red(f"  [FAIL] Could not create client: {exc}"))
        return False

    results: list[bool] = []

    # # ── Test 1: complete() ───────────────────────────────────────────────────
    # print(f"\n  Test 1 — complete()")
    # try:
    #     t0 = time.perf_counter()
    #     resp = llm.complete(TEST_PROMPT, system=SYSTEM_PROMPT, max_tokens=128)
    #     elapsed = time.perf_counter() - t0

    #     print(f"         model    : {resp.model}")
    #     print(f"         tokens   : prompt={resp.prompt_tokens}, completion={resp.completion_tokens}")

    #     if resp.content.strip():
    #         print(_green(f"  [PASS] {elapsed:.2f}s"))
    #         print(f"         response : {resp.content.strip()[:120]}")
    #         results.append(True)
    #     else:
    #         # Show raw API response for diagnosis (e.g. reasoning model fields)
    #         raw_choice = (resp.raw or {}).get("choices", [{}])[0]
    #         print(_yellow(f"  [WARN] content is empty — raw choice keys: {list(raw_choice.keys())}"))
    #         print(_yellow(f"         raw message: {raw_choice.get('message')}"))
    #         # Still count as pass if API call succeeded without exception
    #         results.append(True)

    # except Exception as exc:
    #     print(_red(f"  [FAIL] {exc}"))
    #     traceback.print_exc()
    #     results.append(False)

    # # ── Test 2: chat() ───────────────────────────────────────────────────────
    # print(f"\n  Test 2 — chat() with multi-turn messages")
    # try:
    #     t0 = time.perf_counter()
    #     resp = llm.chat(CHAT_MESSAGES, temperature=0.1, max_tokens=256)
    #     elapsed = time.perf_counter() - t0

    #     assert isinstance(resp.content, str) and resp.content.strip(), \
    #         "Response content is empty"

    #     print(_green(f"  [PASS] {elapsed:.2f}s"))
    #     print(f"         response : {resp.content.strip()[:200]}")
    #     results.append(True)

    # except Exception as exc:
    #     print(_red(f"  [FAIL] {exc}"))
    #     traceback.print_exc()
    #     results.append(False)

    # ── Test 3: 實體抽取（使用 indexer prompt） ──────────────────────────────
    print(f"\n  Test 3 — entity extraction (JSON output)")
    _SAMPLE_TEXT = """\
In the IETF 125 ADD working group session, chairs David Lawrence and Glenn Deen opened the meeting.
Sara Dickinson presented an update on Oblivious DNS over HTTPS (ODoH), which builds on DNS over HTTPS (DoH)
defined in RFC 8484. The IETF ADD working group, chartered in 2020, focuses on adaptive DNS discovery.
Mozilla and Apple have both implemented DoH in their browsers. The session also discussed DNS Privacy,
and its relationship to RFC 7858 (DNS over TLS).
"""
    _EXTRACT_MESSAGES = [
        Message("system", _EXTRACT_SYSTEM),
        Message("user", _EXTRACT_USER.format(text=_SAMPLE_TEXT)),
    ]
    try:
        t0 = time.perf_counter()
        resp = llm.chat(_EXTRACT_MESSAGES, temperature=0.2, max_tokens=1024)
        elapsed = time.perf_counter() - t0

        assert isinstance(resp.content, str) and resp.content.strip(), \
            "Response content is empty"

        raw = resp.content.strip()
        print('--------------------')
        print(raw)
        print('--------------------')
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        print('--------------------')
        print(raw)
        print('--------------------')
        data = json.loads(raw)
        print('--------------------')
        print(data)
        print('--------------------')

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        assert isinstance(entities, list), "entities 不是 list"
        assert isinstance(relationships, list), "relationships 不是 list"
        assert len(entities) > 0, "未擷取到任何實體"

        print(_green(f"  [PASS] {elapsed:.2f}s — {len(entities)} 實體 / {len(relationships)} 關係"))
        print(f"         tokens   : prompt={resp.prompt_tokens}, completion={resp.completion_tokens}")
        print(f"         entities :")
        for e in entities[:8]:
            print(f"           [{e.get('type','?')}] {e.get('name','?')} — {e.get('description','')[:60]}")
        if relationships:
            print(f"         relationships :")
            for r in relationships[:5]:
                print(f"           {r.get('source','?')} →[{r.get('rel_type','?')}]→ {r.get('target','?')}")
        results.append(True)

    except json.JSONDecodeError as exc:
        print(_red(f"  [FAIL] JSON 解析失敗：{exc}"))
        print(f"         raw response : {resp.content.strip()[:300]}")
        results.append(False)
    except AssertionError as exc:
        print(_red(f"  [FAIL] 驗證失敗：{exc}"))
        results.append(False)
    except Exception as exc:
        print(_red(f"  [FAIL] {exc}"))
        traceback.print_exc()
        results.append(False)

    # # ── Test 4: temperature & max_tokens params ──────────────────────────────
    # print(f"\n  Test 3 — parameter: temperature=0, max_tokens=32")
    # try:
    #     resp = llm.complete("Say the word 'hello'.", max_tokens=32, temperature=0)
    #     # Some reasoning models (e.g. Groq gpt-oss-120b) return empty `content`
    #     # and put the answer in `reasoning_content`; we just verify the call
    #     # succeeded and returned a valid LLMResponse.
    #     assert isinstance(resp, LLMResponse), "Did not return LLMResponse"
    #     if resp.content.strip():
    #         print(_green(f"  [PASS] response: {resp.content.strip()[:80]}"))
    #     else:
    #         print(_yellow(f"  [PASS] API call succeeded but content is empty "
    #                       f"(reasoning model?). raw finish_reason: "
    #                       f"{resp.raw.get('choices', [{}])[0].get('finish_reason') if resp.raw else 'N/A'}"))
    #     results.append(True)

    # except Exception as exc:
    #     print(_red(f"  [FAIL] {exc}"))
    #     traceback.print_exc()
    #     results.append(False)

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM client smoke test")
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq", "openai"],
        default=None,
        help="Only test this provider (default: all enabled providers)",
    )
    args = parser.parse_args()

    # Filter providers
    targets = [
        p for p in PROVIDERS
        if p["enabled"] and (args.provider is None or p["name"] == args.provider)
    ]

    if not targets:
        if args.provider:
            print(_yellow(f"Provider '{args.provider}' is disabled. Set enabled=True in PROVIDERS."))
        else:
            print(_yellow("No providers enabled. Edit PROVIDERS list in this file."))
        sys.exit(0)

    summary: dict[str, bool] = {}

    for provider in targets:
        name = provider["name"]
        kwargs = provider["kwargs"]
        _header(f"Provider: {name.upper()}  (model: {kwargs.get('model', '?')})")
        passed = run_test(name, kwargs)
        summary[name] = passed

    # ── Summary ──────────────────────────────────────────────────────────────
    _header("Summary")
    all_passed = True
    for name, passed in summary.items():
        status = _green("PASS") if passed else _red("FAIL")
        print(f"  {name:<12} {status}")
        if not passed:
            all_passed = False

    print()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
