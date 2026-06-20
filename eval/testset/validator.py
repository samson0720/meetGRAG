"""
testset/validator.py
====================
測試集 JSON 格式驗證。

用法
----
  from eval.testset.validator import validate_testset
  errors = validate_testset(Path("eval/testset/samples/sample_testset.json"))
  if errors:
      for e in errors:
          print(e)
"""
from __future__ import annotations

import json
from pathlib import Path


REQUIRED_METADATA = {"version", "created_at", "source_videos", "description"}
REQUIRED_CASE = {"id", "query", "expected_answer", "query_type"}
VALID_QUERY_TYPES = {"local", "global", "auto"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def validate_testset(path: Path) -> list[str]:
    """
    驗證測試集 JSON 格式，回傳所有錯誤訊息列表。
    列表為空代表格式正確。
    """
    errors: list[str] = []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"JSON 解析失敗：{e}"]
    except FileNotFoundError:
        return [f"檔案不存在：{path}"]

    if not isinstance(data, dict):
        return ["根層級應為 dict（物件）"]

    # 驗證 metadata
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("缺少 'metadata' 欄位或格式錯誤")
    else:
        missing_meta = REQUIRED_METADATA - set(metadata.keys())
        if missing_meta:
            errors.append(f"metadata 缺少必填欄位：{missing_meta}")

    # 驗證 test_cases
    test_cases = data.get("test_cases")
    if not isinstance(test_cases, list):
        errors.append("缺少 'test_cases' 欄位或應為 list")
        return errors

    if len(test_cases) == 0:
        errors.append("test_cases 不能為空陣列")

    seen_ids: set[str] = set()
    for i, case in enumerate(test_cases):
        prefix = f"test_cases[{i}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix}：應為 dict")
            continue

        # 必填欄位
        missing = REQUIRED_CASE - set(case.keys())
        if missing:
            errors.append(f"{prefix}：缺少必填欄位 {missing}")

        # id 不重複
        case_id = case.get("id")
        if case_id:
            if case_id in seen_ids:
                errors.append(f"{prefix}：重複的 id='{case_id}'")
            seen_ids.add(case_id)

        # query_type 值域
        qt = case.get("query_type")
        if qt and qt not in VALID_QUERY_TYPES:
            errors.append(f"{prefix} (id={case_id})：query_type='{qt}' 不合法，應為 {VALID_QUERY_TYPES}")

        # difficulty 值域（選填）
        diff = case.get("difficulty")
        if diff and diff not in VALID_DIFFICULTIES:
            errors.append(f"{prefix} (id={case_id})：difficulty='{diff}' 不合法，應為 {VALID_DIFFICULTIES}")

        # 欄位類型檢查（選填欄位）
        for list_field in ("expected_chunks", "relevant_source_videos", "ground_truth_entities", "tags"):
            val = case.get(list_field)
            if val is not None and not isinstance(val, list):
                errors.append(f"{prefix} (id={case_id})：'{list_field}' 應為 list")

    return errors


def validate_and_report(path: Path) -> bool:
    """驗證並印出結果，回傳 True 代表通過。"""
    errors = validate_testset(path)
    if errors:
        print(f"[FAIL] 測試集 {path} 格式錯誤：")
        for e in errors:
            print(f"  - {e}")
        return False
    print(f"[OK] 測試集 {path} 格式驗證通過")
    return True
