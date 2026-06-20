"""
run_index.py
============
對一或多場會議執行完整 GraphRAG 索引流程。

流程
----
  1. meeting_loader  — 從 database/meet_origin_data/{meeting_name}/ 載入各場資料
  2. document_loader — 合併所有記錄，轉換為 TXT + source_map.json
                       （寫入 graphrag/input/_combined/ 或單場名稱的子資料夾）
  3. indexer         — 對合併後的輸入執行 GraphRAG 索引（實體、關係、社群、向量）

使用方式
--------
  # 單場會議
  MEETING_NAMES = ["IETF 125_ IAB Open"]

  # 多場會議（合併建庫）
  MEETING_NAMES = ["IETF 124_ IAB Open", "IETF 125_ IAB Open"]

  python qa_Module/graphrag/run_index.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# 設定區（修改這裡）
# ══════════════════════════════════════════════════════════════════════════════

# 要索引的會議名稱清單（與 database/meet_origin_data/ 下的資料夾名稱完全一致）
# 執行 python qa_Module/graphrag/meeting_loader.py 可列出所有可用會議
# 單場：["IETF 125_ IAB Open"]
# 多場：["IETF 124_ IAB Open", "IETF 125_ IAB Open"]
MEETING_NAMES = [
    "IETF 124_ IAB Open",
    "IETF 125_ IAB Open",
]

# 輸出路徑（相對於專案根目錄）
INPUT_DIR  = "qa_Module/graphrag/input"    # TXT 中間檔輸出位置
OUTPUT_DIR = "qa_Module/graphrag/output"   # GraphRAG 索引輸出位置
SETTINGS   = "qa_Module/graphrag/settings.yaml"

# 是否清除舊的 input/ TXT 檔案後再寫入（建議 True，避免舊資料殘留）
CLEAR_INPUT_DIR = True

# 日誌
VERBOSE = False


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from qa_Module.graphrag.meeting_loader import list_meetings, load_meeting
    from qa_Module.graphrag.document_loader import export_to_input_dir
    from qa_Module.graphrag.indexer import run_indexing

    if not MEETING_NAMES:
        print("錯誤：MEETING_NAMES 不可為空，請至少填入一場會議名稱。")
        sys.exit(1)

    base_input_dir = _ROOT / INPUT_DIR
    output_dir     = _ROOT / OUTPUT_DIR
    settings       = _ROOT / SETTINGS

    # 單場：input_dir = input/{meeting_name}/
    # 多場：input_dir = input/（各場資料放在子資料夾，indexer 以 rglob 掃描）
    if len(MEETING_NAMES) == 1:
        index_input_dir = base_input_dir / MEETING_NAMES[0]
    else:
        index_input_dir = base_input_dir

    # ── 確認所有會議都存在 ────────────────────────────────────────────────────
    available = list_meetings()
    missing = [m for m in MEETING_NAMES if m not in available]
    if missing:
        print("找不到以下會議：")
        for m in missing:
            print(f"  - {m!r}")
        print("\n可用會議：")
        for m in available:
            print(f"  - {m}")
        sys.exit(1)

    # ── 逐場匯出 TXT（每場獨立子資料夾）────────────────────────────────────────
    print(f"[1/2] 載入並匯出會議資料（共 {len(MEETING_NAMES)} 場）")
    total_txts = 0
    for meeting_name in MEETING_NAMES:
        meeting_input_dir = base_input_dir / meeting_name

        # 清除該場舊 TXT（可選）
        if CLEAR_INPUT_DIR and meeting_input_dir.exists():
            removed = sum(
                1 for f in meeting_input_dir.glob("*.txt") if f.unlink() is None
            )
            if removed:
                print(f"      [{meeting_name}] 已清除舊 TXT：{removed} 個")

        records = load_meeting(meeting_name)
        source_map = export_to_input_dir(records, meeting_input_dir)
        total_txts += len(source_map)
        print(f"      [{meeting_name}] {len(records)} 筆記錄 → {len(source_map)} 個 TXT")
        print(f"        → {meeting_input_dir}")

    print(f"      合計：{total_txts} 個 TXT 檔案")

    # ── 執行 GraphRAG 索引 ───────────────────────────────────────────────────
    print(f"\n[2/2] 執行 GraphRAG 索引")
    print(f"      input : {index_input_dir}")
    print(f"      output: {output_dir}")
    run_indexing(
        input_dir=index_input_dir,
        output_dir=output_dir,
        settings_path=settings,
    )
    print(f"\n索引完成！結果儲存於：{output_dir}")
    print(f"涵蓋會議：{', '.join(MEETING_NAMES)}")
