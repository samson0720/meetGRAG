"""
meeting_loader.py
=================
依據會議名稱，從 database/meet_origin_data/{meeting_name}/ 載入完整會議資料，
並回傳與 document_loader.export_to_input_dir() 相容的 list[dict]。

資料夾結構（每場會議）
-----------------------
  database/meet_origin_data/{meeting_name}/
  ├── slides/
  │   └── slide_{id}_{hr}-{min}-{sec}.jpg   # id 對應 transcript.json 的 id
  ├── transcript.json
  └── {meeting_name}.mp4

transcript.json 格式
--------------------
  {
    "status": "done",
    "slides": [
      {
        "id": "001",
        "video_name": "IETF 125_ IAB Open.mp4",
        "start_time": 0.0,
        "end_time": 26.5,
        "ocr_text": "...",
        "chart_description": "...",
        "transcript": "..."
      },
      ...
    ]
  }

使用方式
--------
  from qa_Module.graphrag.meeting_loader import list_meetings, load_meeting

  # 列出所有可用會議
  meetings = list_meetings()

  # 載入特定會議的記錄（供 document_loader 使用）
  records = load_meeting("IETF 125_ IAB Open")
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 預設路徑（相對於 meetGRAG 根目錄）──────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEET_DATA_DIR = _ROOT / "database" / "meet_origin_data"


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API
# ══════════════════════════════════════════════════════════════════════════════

def list_meetings(meet_data_dir: str | Path = DEFAULT_MEET_DATA_DIR) -> list[str]:
    """
    列出 meet_data_dir 下所有可用的會議名稱（資料夾名稱）。

    Returns
    -------
    list[str]
        依字母排序的會議名稱清單。
    """
    meet_data_dir = Path(meet_data_dir)
    if not meet_data_dir.exists():
        raise FileNotFoundError(f"會議資料目錄不存在：{meet_data_dir}")

    meetings = sorted(
        d.name for d in meet_data_dir.iterdir()
        if d.is_dir() and (d / "transcript.json").exists()
    )
    return meetings


def load_meeting(
    meeting_name: str,
    meet_data_dir: str | Path = DEFAULT_MEET_DATA_DIR,
) -> list[dict]:
    """
    載入指定會議的所有 slide 記錄，並自動匹配對應的投影片圖片路徑。

    Parameters
    ----------
    meeting_name : str
        會議名稱，與 meet_data_dir 下的資料夾名稱完全一致。
        例如："IETF 125_ IAB Open"
    meet_data_dir : str | Path
        會議資料根目錄（預設 database/meet_origin_data/）。

    Returns
    -------
    list[dict]
        每筆記錄包含以下欄位（與 document_loader.export_to_input_dir() 相容）：
          - id            : str，例如 "001"
          - video_name    : str，例如 "IETF 125_ IAB Open.mp4"
          - start_time    : float，單位秒
          - end_time      : float，單位秒
          - ocr_text      : str
          - chart_description : str
          - transcript    : str
          - slide_image   : str，投影片圖片的絕對路徑（若無則為空字串）
          - meeting_name  : str，所屬會議名稱

    Raises
    ------
    FileNotFoundError
        若會議資料夾或 transcript.json 不存在。
    ValueError
        若 transcript.json 格式不符預期。
    """
    meet_data_dir = Path(meet_data_dir)
    meeting_dir = meet_data_dir / meeting_name

    if not meeting_dir.exists():
        available = list_meetings(meet_data_dir)
        raise FileNotFoundError(
            f"找不到會議資料夾：{meeting_dir}\n"
            f"可用會議：{available}"
        )

    transcript_path = meeting_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"transcript.json 不存在：{transcript_path}")

    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    slides_data: list[dict] = data.get("slides", [])
    if not slides_data:
        raise ValueError(f"transcript.json 中沒有 slides 資料：{transcript_path}")

    # 建立 id → 投影片圖片路徑的對應表
    slide_image_map = _build_slide_image_map(meeting_dir / "slides")

    records: list[dict] = []
    for entry in slides_data:
        slide_id = str(entry.get("id", "")).zfill(3)   # 統一補零到三位，如 "4" → "004"
        records.append({
            "id":                entry.get("id", slide_id),
            "video_name":        entry.get("video_name", f"{meeting_name}.mp4"),
            "start_time":        float(entry.get("start_time") or 0.0),
            "end_time":          float(entry.get("end_time") or 0.0),
            "ocr_text":          entry.get("ocr_text") or "",
            "chart_description": _clean_chart_description(entry.get("chart_description")),
            "transcript":        entry.get("transcript") or "",
            "slide_image":       slide_image_map.get(slide_id, ""),
            "meeting_name":      meeting_name,
        })

    logger.info(
        "載入會議 '%s'：%d 筆記錄，%d 張投影片已匹配",
        meeting_name,
        len(records),
        sum(1 for r in records if r["slide_image"]),
    )
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 內部輔助函式
# ══════════════════════════════════════════════════════════════════════════════

def _build_slide_image_map(slides_dir: Path) -> dict[str, str]:
    """
    掃描 slides/ 資料夾，建立 slide_id → 絕對圖片路徑 的映射。

    檔名格式：slide_{id}_{hr}-{min}-{sec}.jpg
    例：slide_004_00-01-17.jpg → id = "004"
    """
    if not slides_dir.exists():
        logger.warning("slides 資料夾不存在：%s", slides_dir)
        return {}

    pattern = re.compile(r"^slide_(\d+)_.*\.jpe?g$", re.IGNORECASE)
    image_map: dict[str, str] = {}

    for img_path in slides_dir.iterdir():
        if not img_path.is_file():
            continue
        match = pattern.match(img_path.name)
        if match:
            slide_id = match.group(1).zfill(3)   # 統一三位補零
            image_map[slide_id] = str(img_path.resolve())

    logger.debug("slides 資料夾掃描完成：找到 %d 張圖片", len(image_map))
    return image_map


def _clean_chart_description(value) -> str:
    """移除佔位字串 'String or null'，回傳乾淨的描述文字。"""
    if not value or str(value).strip().lower() in ("string or null", "null", "none"):
        return ""
    return str(value).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 快速驗證（直接執行此檔案）
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    meetings = list_meetings()
    print(f"可用會議（共 {len(meetings)} 場）：")
    for i, m in enumerate(meetings, 1):
        print(f"  [{i}] {m}")

    if not meetings:
        print("找不到任何會議資料，請確認 database/meet_origin_data/ 目錄")
        sys.exit(1)

    # 載入第一場會議做驗證
    target = sys.argv[1] if len(sys.argv) > 1 else meetings[0]
    print(f"\n載入會議：{target}")
    records = load_meeting(target)

    print(f"共 {len(records)} 筆記錄，前 3 筆：")
    for r in records[:3]:
        img = "有" if r["slide_image"] else "無"
        print(
            f"  id={r['id']}  "
            f"time={r['start_time']:.1f}~{r['end_time']:.1f}s  "
            f"slide={img}  "
            f"ocr={r['ocr_text'][:30]!r}"
        )
