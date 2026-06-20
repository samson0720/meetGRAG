"""
view_data.py
============
以互動式 CLI 瀏覽 qa_Module/graphrag/output 中所有 Parquet 檔案的內容與結構。

使用方式
--------
  python frontend/view_data.py              # 互動主選單
  python frontend/view_data.py --summary    # 只顯示所有表格摘要後離開
"""
from __future__ import annotations

import sys
import json
import argparse
import textwrap
from pathlib import Path

import pandas as pd

_ROOT   = Path(__file__).resolve().parents[1]
OUTPUT  = _ROOT / "qa_Module/graphrag/output"

PARQUET_FILES = {
    "text_units":    OUTPUT / "text_units.parquet",
    "entities":      OUTPUT / "entities.parquet",
    "relationships": OUTPUT / "relationships.parquet",
    "graph_edges":   OUTPUT / "graph_edges.parquet",
    "communities":   OUTPUT / "communities.parquet",
}

# ── 欄位說明 ──────────────────────────────────────────────────────────────────

FIELD_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "text_units": {
        "id":          "全域唯一識別碼（UUID hex）",
        "text":        "逐字稿片段（ASR 輸出）",
        "source_file": "原始 JSON 檔路徑",
        "source_video":"對應影片檔名",
        "start_time":  "片段開始時間（秒）",
        "end_time":    "片段結束時間（秒）",
        "slide_image": "對應投影片截圖路徑（可為空）",
        "entity_ids":  "此片段中提取的實體 ID 列表（JSON array string）",
    },
    "entities": {
        "id":           "全域唯一識別碼",
        "name":         "實體名稱（如 QUIC、Glenn Deen）",
        "type":         "實體類型（PERSON/ORGANIZATION/PROTOCOL/RFC/...）",
        "description":  "LLM 生成的實體描述",
        "text_unit_ids":"出現在哪些 text_unit ID 中（JSON array string）",
    },
    "relationships": {
        "id":            "全域唯一識別碼",
        "source":        "關係起點實體名稱",
        "target":        "關係終點實體名稱",
        "rel_type":      "關係類型（CHAIRS、EXTENDS、USES 等）",
        "description":   "最終合併描述",
        "weight":        "關係強度（出現次數 / 共現次數）",
        "is_directional":"是否有方向性",
        "text_unit_ids": "支持此關係的 text_unit ID（JSON array string）",
        "descriptions":  "所有原始描述合集（JSON array string）",
    },
    "graph_edges": {
        "source":        "起點實體名稱",
        "target":        "終點實體名稱",
        "rel_type":      "關係類型",
        "weight":        "關係強度",
        "is_directional":"是否有方向性",
        "description":   "關係描述",
    },
    "communities": {
        "id":           "全域唯一識別碼",
        "level":        "Leiden 演算法層級（0=最細粒度）",
        "title":        "LLM 生成的社群主題標題",
        "summary":      "LLM 生成的社群摘要",
        "entity_names": "社群中所有實體名稱（JSON array string）",
        "text_unit_ids":"社群覆蓋的 text_unit ID（JSON array string）",
        "entity_count": "社群中實體數量",
        "entities":     "實體詳情列表（JSON array string）",
        "relationships":"社群內關係列表（JSON array string）",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 顯示函式
# ══════════════════════════════════════════════════════════════════════════════

def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _load(name: str) -> pd.DataFrame:
    path = PARQUET_FILES[name]
    if not path.exists():
        print(f"  [!] 檔案不存在：{path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def show_summary_all() -> None:
    """顯示所有表格的摘要資訊。"""
    print(_hr("═"))
    print("  GraphRAG Output — 全表摘要")
    print(_hr("═"))
    for name in PARQUET_FILES:
        df = _load(name)
        if df.empty:
            continue
        print(f"\n  [{name}]  {df.shape[0]} 列 × {df.shape[1]} 欄")
        descs = FIELD_DESCRIPTIONS.get(name, {})
        for col in df.columns:
            dtype = str(df[col].dtype)
            desc  = descs.get(col, "")
            null_n = df[col].isna().sum()
            null_s = f"  (null: {null_n})" if null_n else ""
            print(f"    {col:<20} {dtype:<12} {desc}{null_s}")
    print()


def show_table(name: str, n: int = 5) -> None:
    """顯示表格前 n 列（截斷長文字）。"""
    df = _load(name)
    if df.empty:
        return

    print(_hr("═"))
    print(f"  {name}  ── 前 {n} 列（共 {len(df)} 列）")
    print(_hr("═"))

    # 截斷過長欄位
    display = df.head(n).copy()
    for col in display.columns:
        if display[col].dtype == object:
            display[col] = display[col].apply(
                lambda v: (str(v)[:80] + "...") if isinstance(v, str) and len(str(v)) > 80 else v
            )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 60)
    print(display.to_string(index=True))
    print()


def show_row(name: str, idx: int) -> None:
    """完整顯示單一列的所有欄位。"""
    df = _load(name)
    if df.empty:
        return
    if idx < 0 or idx >= len(df):
        print(f"  [!] 索引超界（0 ~ {len(df)-1}）")
        return

    row = df.iloc[idx]
    print(_hr("═"))
    print(f"  {name}  ── 第 {idx} 列")
    print(_hr("═"))
    descs = FIELD_DESCRIPTIONS.get(name, {})
    for col, val in row.items():
        desc = descs.get(col, "")
        val_str = str(val)
        # 嘗試格式化 JSON 欄位
        if val_str.startswith("[") or val_str.startswith("{"):
            try:
                parsed = json.loads(val_str)
                val_str = json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                pass
        # 長文字換行縮排
        if "\n" in val_str or len(val_str) > 80:
            wrapped = textwrap.indent(val_str, "    ")
            print(f"  {col}  ({desc})\n{wrapped}")
        else:
            print(f"  {col:<20} {val_str}   ({desc})")
    print()


def show_schema(name: str) -> None:
    """顯示表格欄位詳細說明。"""
    df = _load(name)
    if df.empty:
        return
    descs = FIELD_DESCRIPTIONS.get(name, {})

    print(_hr("═"))
    print(f"  {name}  ── Schema（{len(df)} 列）")
    print(_hr("═"))
    for col in df.columns:
        dtype   = str(df[col].dtype)
        desc    = descs.get(col, "")
        null_n  = df[col].isna().sum()
        uniq_n  = df[col].nunique()
        print(f"  {col}")
        print(f"    dtype    : {dtype}")
        print(f"    唯一值   : {uniq_n}")
        if null_n:
            print(f"    null 數  : {null_n}")
        if desc:
            print(f"    說明     : {desc}")
        # 數值欄統計
        if df[col].dtype in ("float64", "int64"):
            s = df[col].describe()
            print(f"    min/max  : {s['min']:.3g} / {s['max']:.3g}")
            print(f"    mean±std : {s['mean']:.3g} ± {s['std']:.3g}")
        # 類別欄值分布
        elif df[col].dtype == object and uniq_n <= 20:
            vc = df[col].value_counts().head(10)
            items = ", ".join(f"{k}({v})" for k, v in vc.items())
            print(f"    分布     : {items}")
        print()


def search_entities(keyword: str) -> None:
    """在 entities 表搜尋名稱或描述含關鍵字的實體。"""
    df = _load("entities")
    if df.empty:
        return
    kw = keyword.lower()
    mask = (
        df["name"].str.lower().str.contains(kw, na=False) |
        df["description"].str.lower().str.contains(kw, na=False)
    )
    results = df[mask]
    print(f"\n  搜尋 '{keyword}'：找到 {len(results)} 筆")
    if results.empty:
        return
    for _, row in results.iterrows():
        desc = (row["description"] or "")[:100]
        print(f"  [{row['type']}] {row['name']}")
        if desc:
            print(f"    {desc}...")
    print()


def show_community_detail(idx: int) -> None:
    """顯示單一 community 的結構化內容。"""
    df = _load("communities")
    if df.empty:
        return
    if idx < 0 or idx >= len(df):
        print(f"  [!] 索引超界（0 ~ {len(df)-1}）")
        return

    row = df.iloc[idx]
    print(_hr("═"))
    print(f"  Community [{idx}]  id={row['id']}  level={row['level']}")
    print(_hr("═"))
    print(f"  Title   : {row['title']}")
    print(f"  Entities: {row['entity_count']}")
    print()
    print("  Summary:")
    print(textwrap.indent(textwrap.fill(row["summary"], width=70), "    "))
    print()

    # 實體列表
    try:
        entities = json.loads(row["entities"])
        print(f"  Entities ({len(entities)}):")
        for e in entities:
            print(f"    [{e.get('type','?')}] {e.get('name','?')}")
    except Exception:
        print(f"  entity_names: {row['entity_names'][:200]}")

    # 關係列表
    try:
        rels = json.loads(row["relationships"])
        print(f"\n  Relationships ({len(rels)}):")
        for r in rels[:10]:
            print(f"    {r.get('source','?')} --[{r.get('rel_type','?')}]--> {r.get('target','?')}")
        if len(rels) > 10:
            print(f"    ... 還有 {len(rels)-10} 筆")
    except Exception:
        pass
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 互動選單
# ══════════════════════════════════════════════════════════════════════════════

MENU = """
  ┌─────────────────────────────────────────────────────────┐
  │  GraphRAG Parquet Viewer                                │
  ├─────────────────────────────────────────────────────────┤
  │  s   — 全表摘要（schema + 列數）                        │
  │  1   — text_units    前幾列 / schema / 單列             │
  │  2   — entities      前幾列 / schema / 單列             │
  │  3   — relationships 前幾列 / schema / 單列             │
  │  4   — graph_edges   前幾列 / schema / 單列             │
  │  5   — communities   前幾列 / schema / 社群詳情         │
  │  f   — 搜尋 entities（關鍵字）                          │
  │  q   — 離開                                             │
  └─────────────────────────────────────────────────────────┘
"""

TABLE_NAMES = {
    "1": "text_units",
    "2": "entities",
    "3": "relationships",
    "4": "graph_edges",
    "5": "communities",
}


def _table_submenu(name: str) -> None:
    df = _load(name)
    if df.empty:
        return
    print(f"\n  [{name}] 共 {len(df)} 列")
    print("  h — 前 10 列  |  s — Schema  |  r<n> — 第 n 列詳情", end="")
    if name == "communities":
        print("  |  c<n> — 社群詳情", end="")
    print()
    cmd = input("  > ").strip()

    if cmd == "h":
        show_table(name, n=10)
    elif cmd == "s":
        show_schema(name)
    elif cmd.startswith("r"):
        try:
            show_row(name, int(cmd[1:]))
        except ValueError:
            print("  [!] 請輸入數字，例如 r0")
    elif name == "communities" and cmd.startswith("c"):
        try:
            show_community_detail(int(cmd[1:]))
        except ValueError:
            print("  [!] 請輸入數字，例如 c0")
    else:
        print("  [!] 未知指令")


def interactive() -> None:
    print(MENU)
    while True:
        try:
            cmd = input("  指令 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if cmd == "q":
            break
        elif cmd == "s":
            show_summary_all()
        elif cmd in TABLE_NAMES:
            _table_submenu(TABLE_NAMES[cmd])
        elif cmd == "f":
            kw = input("  搜尋關鍵字：").strip()
            if kw:
                search_entities(kw)
        else:
            print(MENU)


# ══════════════════════════════════════════════════════════════════════════════
# 進入點
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="GraphRAG Parquet Viewer")
    parser.add_argument("--summary", action="store_true", help="只顯示全表摘要後離開")
    args = parser.parse_args()

    if args.summary:
        show_summary_all()
    else:
        interactive()
