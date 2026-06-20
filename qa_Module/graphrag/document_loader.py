"""
document_loader.py
==================
從 database/ 資料夾讀取所有 JSON 記錄，轉換為 GraphRAG 索引管線所需的
TXT 輸入格式，並在每份文件開頭嵌入可溯源元資料標注 [SOURCE: ...]。

輸入：database/*.json
輸出：
  graphrag/input/{video_name}_{id}.txt   每筆記錄一個 TXT 檔
  graphrag/input/source_map.json         TXT 檔名 → 溯源元資料的映射

執行方式（在 meetGRAG 根目錄）：
    python -m qa_Module.graphrag.document_loader
    python -m qa_Module.graphrag.document_loader --database database --output qa_Module/graphrag/input
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 預設路徑（相對於 meetGRAG 根目錄）──────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_DIR = _ROOT / "database" / "meet_origin_data"
DEFAULT_OUTPUT_DIR = _ROOT / "qa_Module" / "graphrag" / "input"


# ══════════════════════════════════════════════════════════════════════════════
# 核心函式
# ══════════════════════════════════════════════════════════════════════════════

def load_records(database_dir: str | Path) -> list[dict]:
    """
    讀取 database_dir 下所有 *.json 檔案，合併為記錄列表。

    每個 JSON 檔案可以是：
      - 單筆 dict：{ "id": ..., "video_name": ..., ... }
      - 多筆 list：[{ ... }, { ... }]

    Returns
    -------
    list[dict]
        所有記錄的合併列表，每筆記錄包含來源檔案路徑 _source_file（內部用）。
    """
    database_dir = Path(database_dir)
    if not database_dir.exists():
        raise FileNotFoundError(f"database 目錄不存在：{database_dir}")

    json_files = [p for p in sorted(database_dir.glob("*.json")) if p.name != "link.json"]
    if not json_files:
        logger.warning("database 目錄中沒有找到任何 JSON 檔案：%s", database_dir)
        return []

    all_records: list[dict] = []
    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("無法讀取 %s：%s", json_path.name, exc)
            continue

        records = data if isinstance(data, list) else [data]
        for rec in records:
            if not isinstance(rec, dict):
                logger.warning("跳過非 dict 記錄（來源：%s）", json_path.name)
                continue
            rec["_source_file"] = json_path.name
            all_records.append(rec)

    logger.info("共載入 %d 筆記錄（%d 個 JSON 檔案）", len(all_records), len(json_files))
    return all_records


def record_to_txt(record: dict) -> str:
    """
    將單筆記錄轉換為帶 [SOURCE] 標注的 TXT 內容。

    格式：
        [SOURCE: {video_name}, START: {start_time}, END: {end_time}, ID: {id}]

        {ocr_text}

        {chart_description}   （若有）

        {transcript}

    Parameters
    ----------
    record : dict
        單筆資料記錄，必須包含 video_name、start_time、end_time。

    Returns
    -------
    str
        格式化後的 TXT 字串。
    """
    video_name = record.get("video_name", "unknown")
    start_time = record.get("start_time", 0.0)
    end_time = record.get("end_time", 0.0)
    record_id = record.get("id", "unknown")

    source_header = (
        f"[SOURCE: {video_name}, "
        f"START: {start_time}, "
        f"END: {end_time}, "
        f"ID: {record_id}]"
    )

    parts = [source_header]

    ocr_text = (record.get("ocr_text") or "").strip()
    if ocr_text:
        parts.append(ocr_text)

    chart_description = (record.get("chart_description") or "").strip()
    if chart_description:
        parts.append(chart_description)

    transcript = (record.get("transcript") or "").strip()
    if transcript:
        parts.append(transcript)

    return "\n\n".join(parts)


def _safe_filename(video_name: str, record_id: str) -> str:
    """
    產生安全的檔案名稱，移除或替換不合法字元。

    格式：{video_name_stem}_{record_id}.txt
    例：RFC9114_HTTP3_rfc9114_001.txt
    """
    stem = Path(video_name).stem  # 去掉副檔名
    # 只保留英數字、底線、連字號
    stem = re.sub(r"[^\w\-]", "_", stem)
    safe_id = re.sub(r"[^\w\-]", "_", str(record_id))
    return f"{stem}_{safe_id}.txt"


def export_to_input_dir(
    records: list[dict],
    output_dir: str | Path,
) -> dict[str, dict]:
    """
    將每筆記錄寫出為獨立的 TXT 檔案，並輸出 source_map.json。

    Parameters
    ----------
    records : list[dict]
        load_records() 回傳的記錄列表。
    output_dir : str | Path
        輸出目錄（若不存在則自動建立）。

    Returns
    -------
    dict[str, dict]
        source_map：{ "txt 檔名" → { source_video, start_time, end_time, id } }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_map: dict[str, dict] = {}
    written = 0
    skipped = 0

    for record in records:
        record_id = record.get("id")
        video_name = record.get("video_name")

        if not record_id or not video_name:
            logger.warning(
                "記錄缺少 id 或 video_name，跳過：%s",
                {k: record.get(k) for k in ("id", "video_name", "_source_file")},
            )
            skipped += 1
            continue

        filename = _safe_filename(video_name, record_id)
        txt_path = output_dir / filename

        txt_content = record_to_txt(record)
        try:
            txt_path.write_text(txt_content, encoding="utf-8")
            written += 1
        except OSError as exc:
            logger.error("寫入 %s 失敗：%s", filename, exc)
            skipped += 1
            continue

        source_map[filename] = {
            "source_video": video_name,
            "start_time": record.get("start_time", 0.0),
            "end_time": record.get("end_time", 0.0),
            "id": record_id,
            "slide_image": record.get("slide_image", ""),
        }

    # 輸出 source_map.json
    source_map_path = output_dir / "source_map.json"
    source_map_path.write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "完成：寫出 %d 個 TXT 檔案，跳過 %d 筆，source_map.json → %s",
        written, skipped, source_map_path,
    )
    return source_map


def load_source_map(output_dir: str | Path) -> dict[str, dict]:
    """
    讀取已產生的 source_map.json。
    供 indexer.py 的 inject_traceability() 使用。
    """
    path = Path(output_dir) / "source_map.json"
    if not path.exists():
        raise FileNotFoundError(f"source_map.json 不存在，請先執行 export_to_input_dir()：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 主程式入口（直接執行 / CLI）
# ══════════════════════════════════════════════════════════════════════════════

def run(database_dir: Path, output_dir: Path, verbose: bool = False) -> dict[str, dict]:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s  %(message)s", level=level)

    print(f"[document_loader] database 目錄：{database_dir}")
    print(f"[document_loader] 輸出目錄　　：{output_dir}")

    records = load_records(database_dir)
    if not records:
        print("[document_loader] 沒有記錄可處理，結束。")
        return {}

    source_map = export_to_input_dir(records, output_dir)

    # 印出摘要
    print(f"\n[document_loader] 完成！共輸出 {len(source_map)} 筆")
    print(f"  TXT 檔案 → {output_dir}/")
    print(f"  source_map → {output_dir / 'source_map.json'}")
    return source_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="將 database/*.json 轉換為 GraphRAG 輸入 TXT"
    )
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DATABASE_DIR),
        help=f"JSON 資料庫目錄（預設：{DEFAULT_DATABASE_DIR}）",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"TXT 輸出目錄（預設：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="顯示 DEBUG 詳細日誌",
    )
    args = parser.parse_args()

    source_map = run(Path(args.database), Path(args.output), args.verbose)
    sys.exit(0 if source_map else 1)


# if __name__ == "__main__":
#     main()
