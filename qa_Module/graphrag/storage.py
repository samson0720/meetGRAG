"""
storage.py
==========
GraphRAG 索引結果的讀寫模組。

混合儲存策略：
  Parquet（大型表格資料，columnar + zstd 壓縮，快速欄位過濾）
    text_units.parquet     — TextUnit（含溯源 source_video/start_time/end_time）
    entities.parquet       — Entity（type/description/text_unit_ids）
    relationships.parquet  — Relationship（rel_type/source/target/weight）
    graph_edges.parquet    — 純圖結構（src/rel_type/tgt/weight，NetworkX 建圖用）

  JSON（小型 / 需人工閱讀的資料）
    community_reports.json — Community id/title/summary（Global Search 查詢）
    index_meta.json        — 索引統計與時間戳

若 pandas/pyarrow 未安裝，自動降級為全 JSON 輸出。

公開 API
--------
  save(text_units, entity_map, rel_map, communities, output_dir)
  load(output_dir)  → dict[str, list|dict]
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qa_Module.graphrag.indexer import (
        Community, Entity, Relationship, TextUnit,
    )

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 內部工具
# ══════════════════════════════════════════════════════════════════════════════

def _to_dict(obj) -> dict:
    """dataclass → plain dict，確保所有欄位可 JSON / Parquet 序列化。"""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return vars(obj)


def _has_pandas() -> bool: 
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def _write_parquet(path: Path, records: list[dict]) -> None:
    """
    寫出 Parquet（zstd 壓縮）。
    Parquet 不支援 nested list，因此 list 欄位先序列化為 JSON 字串。
    """
    import pandas as pd

    if not records:
        return  # 不寫空檔，讀取端以 .exists() 判斷

    serialized = [
        {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
         for k, v in row.items()}
        for row in records
    ]
    pd.DataFrame(serialized).to_parquet(path, index=False, compression="zstd")


def _read_parquet(path: Path) -> list[dict]:
    """
    讀取 Parquet，並將被序列化為 JSON 字串的 list 欄位還原。
    """
    import pandas as pd

    df = pd.read_parquet(path)
    records = df.to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, str) and v.startswith("["):
                try:
                    row[k] = json.loads(v)
                except json.JSONDecodeError:
                    pass
    return records


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _log_file(path: Path, count: int) -> None:
    size_kb = path.stat().st_size // 1024 if path.exists() else 0
    logger.info("  %-35s %5d 項  %4d KB", path.name, count, size_kb)


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API — 分階段儲存
# ══════════════════════════════════════════════════════════════════════════════

def save_text_units(text_units: list[TextUnit], output_dir: Path) -> None:
    """Step 2 完成後：儲存 text_units.parquet"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "text_units"
    records = [_to_dict(tu) for tu in text_units]
    if _has_pandas():
        p = output_dir / f"{stem}.parquet"
        _write_parquet(p, records)
    else:
        p = output_dir / f"{stem}.json"
        _write_json(p, records)
    _log_file(p, len(records))


def save_entities(entity_map: dict[str, Entity], output_dir: Path) -> None:
    """Step 4 完成後：儲存 entities.parquet"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "entities"
    records = [_to_dict(e) for e in entity_map.values()]
    if _has_pandas():
        p = output_dir / f"{stem}.parquet"
        _write_parquet(p, records)
    else:
        p = output_dir / f"{stem}.json"
        _write_json(p, records)
    _log_file(p, len(records))


def save_relationships(rel_map: dict[str, Relationship], output_dir: Path) -> None:
    """Step 4 完成後：儲存 relationships.parquet + graph_edges.parquet"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_parquet = _has_pandas()

    rel_records = [_to_dict(r) for r in rel_map.values()]
    stem = "relationships"
    if use_parquet:
        p = output_dir / f"{stem}.parquet"
        _write_parquet(p, rel_records)
    else:
        p = output_dir / f"{stem}.json"
        _write_json(p, rel_records)
    _log_file(p, len(rel_records))

    edge_records = [
        {
            "source":         r.source,
            "target":         r.target,
            "rel_type":       r.rel_type,
            "weight":         r.weight,
            "is_directional": r.is_directional,
            "description":    r.description,
        }
        for r in rel_map.values()
    ]
    stem = "graph_edges"
    if use_parquet:
        p = output_dir / f"{stem}.parquet"
        _write_parquet(p, edge_records)
    else:
        p = output_dir / f"{stem}.json"
        _write_json(p, edge_records)
    _log_file(p, len(edge_records))


def save_communities(
    communities: list[Community],
    entity_map: dict[str, Entity],
    rel_map: dict[str, Relationship],
    output_dir: Path,
) -> None:
    """
    Step 5（圖結構）或 Step 6（含 LLM 摘要）完成後呼叫，覆蓋寫入 communities.json。

    每筆 community 記錄包含：
      - id, level, title, summary          — 識別與 LLM 摘要（Global Search）
      - entity_names, text_unit_ids        — 成員索引（Local Search）
      - entities                           — 成員實體完整資料
      - relationships                      — 社群內部關係完整資料
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from qa_Module.graphrag.indexer import _canonical_name

    records = []
    for c in communities:
        # 成員實體詳細資料
        member_entities = []
        member_canonical: set[str] = set()
        for em_key in c.entity_names:
            e = entity_map.get(em_key)
            if e:
                member_entities.append({
                    "id":          e.id,
                    "name":        e.name,
                    "type":        e.type,
                    "description": e.description,
                })
                member_canonical.add(_canonical_name(e.name))

        # 社群內部關係：兩端都在成員中
        member_rels = []
        for r in rel_map.values():
            if (_canonical_name(r.source) in member_canonical and
                    _canonical_name(r.target) in member_canonical):
                member_rels.append({
                    "source":      r.source,
                    "target":      r.target,
                    "rel_type":    r.rel_type,
                    "description": r.description,
                    "weight":      r.weight,
                })

        records.append({
            "id":            c.id,
            "level":         c.level,
            "title":         c.title,
            "summary":       c.summary,
            "entity_names":  c.entity_names,
            "text_unit_ids": c.text_unit_ids,
            "entity_count":  len(c.entity_names),
            "entities":      member_entities,
            "relationships": member_rels,
        })

    stem = "communities"
    if _has_pandas():
        # entities / relationships 是 list[dict]，需額外序列化為 JSON 字串後才能存 Parquet
        parquet_records = [
            {**r,
             "entities":      json.dumps(r["entities"],      ensure_ascii=False),
             "relationships": json.dumps(r["relationships"], ensure_ascii=False)}
            for r in records
        ]
        p = output_dir / f"{stem}.parquet"
        _write_parquet(p, parquet_records)
    else:
        p = output_dir / f"{stem}.json"
        _write_json(p, records)
    _log_file(p, len(records))


def save_meta(
    text_units: list[TextUnit],
    entity_map: dict[str, Entity],
    rel_map: dict[str, Relationship],
    communities: list[Community],
    output_dir: Path,
    extra: dict | None = None,
) -> None:
    """最終：儲存 index_meta.json（統計摘要）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    use_parquet = _has_pandas()
    ext = ".parquet" if use_parquet else ".json"
    written = [
        f"text_units{ext}", f"entities{ext}",
        f"relationships{ext}", f"graph_edges{ext}",
        f"communities{ext}", "index_meta.json",
    ]
    meta = {
        "indexed_at":         datetime.now(timezone.utc).isoformat(),
        "storage_format":     "parquet+json" if use_parquet else "json",
        "text_unit_count":    len(text_units),
        "entity_count":       len(entity_map),
        "relationship_count": len(rel_map),
        "community_count":    len(communities),
        "entity_types":       sorted({e.type for e in entity_map.values()}),
        "rel_types":          sorted({r.rel_type for r in rel_map.values()}),
        "output_files":       written,
        **(extra or {}),
    }
    meta_path = output_dir / "index_meta.json"
    _write_json(meta_path, meta)
    logger.info("  %-35s 已更新", meta_path.name)


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API — 一次性儲存（向後相容）
# ══════════════════════════════════════════════════════════════════════════════

def save(
    text_units: list[TextUnit],
    entity_map: dict[str, Entity],
    rel_map: dict[str, Relationship],
    communities: list[Community],
    output_dir: Path,
) -> None:
    """
    將 GraphRAG 索引結果一次性持久化至 output_dir/（向後相容）。
    等同於依序呼叫 save_text_units → save_entities → save_relationships
    → save_communities → save_meta。
    """
    if not _has_pandas():
        logger.warning("pandas 未安裝，降級為全 JSON 輸出（pip install pandas pyarrow）")
    logger.info("=== 儲存索引結果 → %s ===", output_dir)
    save_text_units(text_units, output_dir)
    save_entities(entity_map, output_dir)
    save_relationships(rel_map, output_dir)
    save_communities(communities, entity_map, rel_map, output_dir)
    save_meta(text_units, entity_map, rel_map, communities, output_dir)
    logger.info("儲存完成")


def load_table(stem: str, output_dir: Path) -> list[dict]:
    """
    從 output_dir 載入單張資料表（text_units / entities / relationships /
    graph_edges / communities），自動偵測 parquet / json 格式。

    供只需要部分表格的呼叫端使用，避免一次載入所有資料。
    """
    output_dir = Path(output_dir)
    parquet_path = output_dir / f"{stem}.parquet"
    json_path    = output_dir / f"{stem}.json"
    if _has_pandas() and parquet_path.exists() and parquet_path.stat().st_size > 0:
        return _read_parquet(parquet_path)
    elif json_path.exists():
        return _read_json(json_path)
    logger.warning("找不到 %s（.parquet 或 .json）", stem)
    return []


def load(output_dir: Path) -> dict[str, Any]:
    """
    從 output_dir/ 載入索引結果，自動偵測 parquet / json 格式。

    Returns
    -------
    dict with keys:
      "text_units"    — list[dict]   TextUnit（含溯源）
      "entities"      — list[dict]   Entity（type/description/text_unit_ids）
      "relationships" — list[dict]   Relationship（rel_type/weight/is_directional）
      "graph_edges"   — list[dict]   精簡圖結構（NetworkX 建圖用）
      "communities"   — list[dict]   社群完整資料（圖結構 + 成員實體/關係 + LLM 摘要）
      "meta"          — dict         索引統計與時間戳

    讀取時 list 欄位（entity_ids、text_unit_ids 等）
    會從 JSON 字串自動還原為 Python list。
    """
    output_dir = Path(output_dir)
    use_parquet = _has_pandas()

    def _load_table(stem: str) -> list[dict]:
        parquet_path = output_dir / f"{stem}.parquet"
        json_path    = output_dir / f"{stem}.json"
        if use_parquet and parquet_path.exists() and parquet_path.stat().st_size > 0:
            return _read_parquet(parquet_path)
        elif json_path.exists():
            return _read_json(json_path)
        logger.warning("找不到 %s（.parquet 或 .json）", stem)
        return []

    return {
        "text_units":        _load_table("text_units"),
        "entities":          _load_table("entities"),
        "relationships":     _load_table("relationships"),
        "graph_edges":       _load_table("graph_edges"),
        "communities": _load_table("communities"),
        "meta":        _read_json(output_dir / "index_meta.json"),
    }
