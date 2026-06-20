"""
vectorizer.py
=============
GraphRAG 向量化模組。

從本機 output/ 目錄讀取已索引的資料（text_units / entities / communities），
產生嵌入向量並存入 LanceDB，供查詢模組進行語意搜尋。

嵌入模型支援
-----------
  OpenAI   text-embedding-3-small / text-embedding-3-large
  Ollama   nomic-embed-text 等本機模型（不需 API key）
  設定來源：settings.yaml → embeddings.provider / embeddings.model

公開 API
--------
  build_index(output_dir, settings_path, *, force) -> Path
      讀取 storage.load() 資料，嵌入並寫入 LanceDB。
      force=True 時強制重新嵌入（即使索引已存在）。

  VectorSearcher(lancedb_path, settings_path)
      封裝 LanceDB 向量搜尋。
      .search_text_units(query, top_k)   -> list[dict]  含 source_video/start_time
      .search_entities(query, top_k)     -> list[dict]  含 name/type/description
      .search_communities(query, top_k)  -> list[dict]  含 title/summary
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]   # meetGRAG 專案根目錄


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def _load_settings(settings_path: Path | str | None, ref_dir: Path) -> dict:
    """讀取 settings.yaml；若未指定路徑則從 ref_dir/../settings.yaml 嘗試。"""
    candidates = []
    if settings_path:
        candidates.append(Path(settings_path))
    candidates += [
        ref_dir.parent / "settings.yaml",
        _ROOT / "qa_Module" / "graphrag" / "settings.yaml",
    ]
    for p in candidates:
        if p.exists():
            try:
                import yaml
                with open(p, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                logger.warning("PyYAML 未安裝（pip install pyyaml），使用預設設定")
            except Exception as exc:
                logger.warning("讀取 settings.yaml 失敗：%s", exc)
    return {}


def _resolve_lancedb_path(settings: dict, output_dir: Path) -> Path:
    """從 settings 取得 lancedb 路徑，相對路徑以專案根目錄為基準。"""
    raw = settings.get("storage", {}).get("lancedb_path", "")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (_ROOT / p).resolve()
    return output_dir.parent / "lancedb"


# ══════════════════════════════════════════════════════════════════════════════
# 嵌入模型客戶端
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingClient:
    """嵌入客戶端基類。"""

    def embed(self, text: str) -> list[float]:
        return self._embed_batch_raw([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 96) -> list[list[float]]:
        """分批送出嵌入請求，避免超過 API token 上限。"""
        results: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            logger.debug("  嵌入 batch %d–%d / %d", i + 1, min(i + batch_size, total), total)
            results.extend(self._embed_batch_raw(batch))
            if i + batch_size < total:
                time.sleep(0.1)
        return results

    async def embed_batch_async(
        self,
        texts: list[str],
        batch_size: int = 96,
        concurrency: int = 4,
    ) -> list[list[float]]:
        """
        非同步分批嵌入。

        將 texts 切成 batch_size 的小批次，最多同時送出 concurrency 個批次，
        全部完成後依原始順序拼回結果。

        Parameters
        ----------
        texts       待嵌入的文字列表
        batch_size  每批次的文字數量（預設 96）
        concurrency 同時進行的 API 批次上限（預設 4）
        """
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        total   = len(batches)
        sem     = asyncio.Semaphore(concurrency)

        async def _one(idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            async with sem:
                logger.debug("  嵌入 batch %d / %d（%d 筆）", idx + 1, total, len(batch))
                vecs = await asyncio.to_thread(self._embed_batch_raw, batch)
            return idx, vecs

        raw = await asyncio.gather(*[_one(i, b) for i, b in enumerate(batches)])
        # 依 idx 排序後拼接（gather 不保證順序）
        ordered: list[list[float]] = []
        for _, vecs in sorted(raw, key=lambda x: x[0]):
            ordered.extend(vecs)
        return ordered

    def _embed_batch_raw(self, _texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddingClient(EmbeddingClient):
    """
    使用 OpenAI Embedding API（text-embedding-3-small 等）。
    API key 優先從環境變數讀取：OPENAPI_API_KEY 或 OPENAI_API_KEY。
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("請安裝 openai：pip install openai") from exc

        key = api_key or os.getenv("OPENAPI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key 未設定。"
                "請在 .env 中設定 OPENAPI_API_KEY 或 OPENAI_API_KEY。"
            )
        self._client = OpenAI(api_key=key)
        self.model = model
        logger.info("OpenAI embedding 客戶端：model=%s", model)

    def _embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class OllamaEmbeddingClient(EmbeddingClient):
    """
    使用 Ollama 本機嵌入（nomic-embed-text / mxbai-embed-large 等）。
    不需要 API key，僅需本機 Ollama 服務運行中。
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        logger.info("Ollama embedding 客戶端：model=%s  base_url=%s", model, base_url)

    def _embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        import urllib.request
        results: list[list[float]] = []
        for text in texts:
            payload = json.dumps({"model": self.model, "prompt": text}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            results.append(data["embedding"])
        return results


def create_embedding_client(settings: dict) -> EmbeddingClient:
    """
    根據 settings 建立嵌入客戶端。

    settings.yaml 範例：
      embeddings:
        provider: openai        # openai | ollama（省略時繼承 llm.provider）
        model: text-embedding-3-small
        base_url: http://localhost:11434  # ollama 專用
    """
    emb_cfg = settings.get("embeddings", {})
    llm_cfg = settings.get("llm", {})

    provider = emb_cfg.get("provider") or llm_cfg.get("provider", "openai")
    model    = emb_cfg.get("model", "text-embedding-3-small")

    if provider == "ollama":
        base_url = (
            emb_cfg.get("base_url")
            or llm_cfg.get("base_url", "http://localhost:11434")
        )
        return OllamaEmbeddingClient(model=model, base_url=base_url)

    from dotenv import load_dotenv
    load_dotenv()
    return OpenAIEmbeddingClient(model=model)


# ══════════════════════════════════════════════════════════════════════════════
# LanceDB 索引建立
# ══════════════════════════════════════════════════════════════════════════════

def _build_lancedb(
    lancedb_path: Path,
    text_units: list[dict],
    entities: list[dict],
    communities: list[dict],
    client: EmbeddingClient,
    force: bool,
) -> None:
    import lancedb

    lancedb_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lancedb_path))

    def _exists(name: str) -> bool:
        try:
            db.open_table(name)
            return True
        except Exception:
            return False

    # ── text_units ──────────────────────────────────────────────────────────
    if not _exists("text_units") or force:
        logger.info("嵌入 text_units（%d 筆）...", len(text_units))
        vectors = client.embed_batch([r["text"] for r in text_units])
        rows = [
            {
                "id":           r["id"],
                "text":         r["text"],
                "source_video": r.get("source_video", ""),
                "start_time":   float(r.get("start_time", 0.0)),
                "end_time":     float(r.get("end_time", 0.0)),
                "slide_image":  r.get("slide_image", ""),
                "entity_ids":   json.dumps(r.get("entity_ids", []), ensure_ascii=False),
                "vector":       v,
            }
            for r, v in zip(text_units, vectors)
        ]
        db.create_table("text_units", data=rows, mode="overwrite")
        logger.info("  text_units：%d 筆已索引", len(rows))
    else:
        logger.info("  text_units：索引已存在，跳過（force=True 可強制重建）")

    # ── entities ────────────────────────────────────────────────────────────
    if not _exists("entities") or force:
        logger.info("嵌入 entities（%d 筆）...", len(entities))
        embed_texts = [f"{r['name']}: {r.get('description', '')}" for r in entities]
        vectors = client.embed_batch(embed_texts)
        rows = [
            {
                "id":          r["id"],
                "name":        r["name"],
                "type":        r.get("type", ""),
                "description": r.get("description", ""),
                "embed_text":  t,
                "vector":      v,
            }
            for r, t, v in zip(entities, embed_texts, vectors)
        ]
        db.create_table("entities", data=rows, mode="overwrite")
        logger.info("  entities：%d 筆已索引", len(rows))
    else:
        logger.info("  entities：索引已存在，跳過（force=True 可強制重建）")

    # ── communities（只嵌入有 LLM 摘要的）──────────────────────────────────
    c_with_summary = [r for r in communities if r.get("summary")]
    if (not _exists("communities") or force) and c_with_summary:
        logger.info("嵌入 communities（%d 筆）...", len(c_with_summary))
        embed_texts = [f"{r.get('title', '')}\n{r['summary']}" for r in c_with_summary]
        vectors = client.embed_batch(embed_texts)
        rows = [
            {
                "id":      r["id"],
                "title":   r.get("title", ""),
                "summary": r["summary"],
                "level":   int(r.get("level", 0)),
                "vector":  v,
            }
            for r, v in zip(c_with_summary, vectors)
        ]
        db.create_table("communities", data=rows, mode="overwrite")
        logger.info("  communities：%d 筆已索引", len(rows))
    elif not c_with_summary:
        logger.info("  communities：無 LLM 摘要，跳過（請先執行 Step 6）")
    else:
        logger.info("  communities：索引已存在，跳過（force=True 可強制重建）")


async def _build_lancedb_async(
    lancedb_path: "Path",
    text_units: list[dict],
    entities: list[dict],
    communities: list[dict],
    client: "EmbeddingClient",
    force: bool,
    concurrency: int,
) -> None:
    """非同步版 _build_lancedb，使用 embed_batch_async() 並發嵌入。"""
    import lancedb

    lancedb_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lancedb_path))

    def _exists(name: str) -> bool:
        try:
            db.open_table(name)
            return True
        except Exception:
            return False

    # ── text_units ──────────────────────────────────────────────────────────
    if not _exists("text_units") or force:
        logger.info("非同步嵌入 text_units（%d 筆，concurrency=%d）...",
                    len(text_units), concurrency)
        vectors = await client.embed_batch_async(
            [r["text"] for r in text_units], concurrency=concurrency
        )
        rows = [
            {
                "id":           r["id"],
                "text":         r["text"],
                "source_video": r.get("source_video", ""),
                "start_time":   float(r.get("start_time", 0.0)),
                "end_time":     float(r.get("end_time", 0.0)),
                "slide_image":  r.get("slide_image", ""),
                "entity_ids":   json.dumps(r.get("entity_ids", []), ensure_ascii=False),
                "vector":       v,
            }
            for r, v in zip(text_units, vectors)
        ]
        db.create_table("text_units", data=rows, mode="overwrite")
        logger.info("  text_units：%d 筆已索引", len(rows))
    else:
        logger.info("  text_units：索引已存在，跳過（force=True 可強制重建）")

    # ── entities ────────────────────────────────────────────────────────────
    if not _exists("entities") or force:
        logger.info("非同步嵌入 entities（%d 筆，concurrency=%d）...",
                    len(entities), concurrency)
        embed_texts_ent = [f"{r['name']}: {r.get('description', '')}" for r in entities]
        vectors = await client.embed_batch_async(embed_texts_ent, concurrency=concurrency)
        rows = [
            {
                "id":          r["id"],
                "name":        r["name"],
                "type":        r.get("type", ""),
                "description": r.get("description", ""),
                "embed_text":  t,
                "vector":      v,
            }
            for r, t, v in zip(entities, embed_texts_ent, vectors)
        ]
        db.create_table("entities", data=rows, mode="overwrite")
        logger.info("  entities：%d 筆已索引", len(rows))
    else:
        logger.info("  entities：索引已存在，跳過（force=True 可強制重建）")

    # ── communities ─────────────────────────────────────────────────────────
    c_with_summary = [r for r in communities if r.get("summary")]
    if (not _exists("communities") or force) and c_with_summary:
        logger.info("非同步嵌入 communities（%d 筆，concurrency=%d）...",
                    len(c_with_summary), concurrency)
        embed_texts_com = [
            f"{r.get('title', '')}\n{r['summary']}" for r in c_with_summary
        ]
        vectors = await client.embed_batch_async(embed_texts_com, concurrency=concurrency)
        rows = [
            {
                "id":      r["id"],
                "title":   r.get("title", ""),
                "summary": r["summary"],
                "level":   int(r.get("level", 0)),
                "vector":  v,
            }
            for r, v in zip(c_with_summary, vectors)
        ]
        db.create_table("communities", data=rows, mode="overwrite")
        logger.info("  communities：%d 筆已索引", len(rows))
    elif not c_with_summary:
        logger.info("  communities：無 LLM 摘要，跳過（請先執行 Step 6）")
    else:
        logger.info("  communities：索引已存在，跳過（force=True 可強制重建）")


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API — 建立向量索引
# ══════════════════════════════════════════════════════════════════════════════

async def build_index_async(
    output_dir: "Path | str",
    settings_path: "Path | str | None" = None,
    *,
    force: bool = False,
    concurrency: int = 4,
) -> "Path":
    """
    非同步版 build_index。從 output_dir 讀取索引資料，並發嵌入並建立 LanceDB。

    Parameters
    ----------
    output_dir      storage.save() 的輸出目錄
    settings_path   settings.yaml 路徑；None → 自動搜尋
    force           True → 強制重新嵌入
    concurrency     同時進行的嵌入 batch 上限

    Returns
    -------
    LanceDB 資料夾路徑
    """
    output_dir = Path(output_dir)
    settings   = _load_settings(settings_path, output_dir)

    from qa_Module.graphrag.storage import load_table

    text_units  = load_table("text_units",  output_dir)
    entities    = load_table("entities",    output_dir)
    communities = load_table("communities", output_dir)
    logger.info(
        "載入：%d text_units / %d entities / %d communities",
        len(text_units), len(entities), len(communities),
    )

    client       = create_embedding_client(settings)
    lancedb_path = _resolve_lancedb_path(settings, output_dir)
    logger.info("LanceDB → %s", lancedb_path)
    await _build_lancedb_async(
        lancedb_path, text_units, entities, communities, client, force, concurrency
    )
    return lancedb_path


def build_index(
    output_dir: Path | str,
    settings_path: Path | str | None = None,
    *,
    force: bool = False,
) -> Path:
    """
    從 output_dir 讀取索引資料，嵌入並建立 LanceDB 向量索引。

    Parameters
    ----------
    output_dir      storage.save() 的輸出目錄（含 text_units.parquet 等）
    settings_path   settings.yaml 路徑；None → 自動搜尋
    force           True → 強制重新嵌入（即使索引已存在）

    Returns
    -------
    LanceDB 資料夾路徑
    """
    output_dir = Path(output_dir)
    settings = _load_settings(settings_path, output_dir)

    from qa_Module.graphrag.storage import load_table

    text_units  = load_table("text_units",  output_dir)
    entities    = load_table("entities",    output_dir)
    communities = load_table("communities", output_dir)
    logger.info(
        "載入：%d text_units / %d entities / %d communities",
        len(text_units), len(entities), len(communities),
    )

    client = create_embedding_client(settings)
    lancedb_path = _resolve_lancedb_path(settings, output_dir)
    logger.info("LanceDB → %s", lancedb_path)
    _build_lancedb(lancedb_path, text_units, entities, communities, client, force)
    return lancedb_path


# ══════════════════════════════════════════════════════════════════════════════
# 公開 API — 向量搜尋
# ══════════════════════════════════════════════════════════════════════════════

class VectorSearcher:
    """
    封裝 LanceDB 向量搜尋。

    使用方式
    --------
    searcher = VectorSearcher(lancedb_path, settings_path)

    chunks = searcher.search_text_units("HTTP/3 QUIC implementation", top_k=10)
    ents   = searcher.search_entities("QUIC transport protocol", top_k=5)
    comms  = searcher.search_communities("What are the key protocol improvements?", top_k=3)

    每筆回傳 dict 包含原始欄位 + "score"（相似度，越高越相關）。
    """

    def __init__(
        self,
        lancedb_path: Path | str,
        settings_path: Path | str | None = None,
    ):
        self._lancedb_path = Path(lancedb_path)
        settings_path = (
            Path(settings_path) if settings_path
            else _ROOT / "qa_Module" / "graphrag" / "settings.yaml"
        )
        _settings = _load_settings(settings_path, self._lancedb_path)
        self._client = create_embedding_client(_settings)
        self._db: Any = None

    def _get_db(self) -> Any:
        if self._db is None:
            import lancedb
            self._db = lancedb.connect(str(self._lancedb_path))
        return self._db

    def _search(self, table_name: str, query: str, top_k: int) -> list[dict]:
        db = self._get_db()
        try:
            tbl = db.open_table(table_name)
        except Exception:
            logger.warning("LanceDB 找不到資料表 '%s'", table_name)
            return []
        query_vec = self._client.embed(query)
        results = tbl.search(query_vec).limit(top_k).to_list()
        for row in results:
            dist = row.pop("_distance", None)
            row["score"] = round(1.0 - dist, 6) if dist is not None else None
        return results

    def search_text_units(self, query: str, top_k: int = 10) -> list[dict]:
        """
        語意搜尋文字塊（TextUnit）。
        回傳欄位：id / text / source_video / start_time / end_time / slide_image / entity_ids / score
        """
        return self._search("text_units", query, top_k)

    def search_entities(self, query: str, top_k: int = 10) -> list[dict]:
        """
        語意搜尋實體（Entity）。
        回傳欄位：id / name / type / description / embed_text / score
        """
        return self._search("entities", query, top_k)

    def search_communities(self, query: str, top_k: int = 5) -> list[dict]:
        """
        語意搜尋社群摘要（Community Report），適用於 Global Search。
        回傳欄位：id / title / summary / level / score
        """
        return self._search("communities", query, top_k)


# ══════════════════════════════════════════════════════════════════════════════
# 命令列快速測試
# ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     import sys

#     # 直接執行時（python vectorizer.py）將專案根目錄加入 sys.path
#     # 以模組方式執行時（python -m ...）則不需要
#     if str(_ROOT) not in sys.path:
#         sys.path.insert(0, str(_ROOT))

#     if sys.platform == "win32":
#         import io
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

#     logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

#     # ── 測試參數（直接修改這裡）────────────────────────────────────────────────
#     CMD        = "build"                        # "build" | "search"

#     # build 參數
#     OUTPUT_DIR = _ROOT / "qa_Module/graphrag/output"
#     SETTINGS   = None                           # None = 自動搜尋 settings.yaml
#     FORCE      = False                          # True = 強制重新嵌入

#     # search 參數
#     QUERY      = "Who is ADD working group chair"
#     LANCEDB    = _ROOT / "qa_Module/graphrag/lancedb"
#     TABLE      = "text_units"                   # "text_units" | "entities" | "communities"
#     TOP_K      = 5
#     # ─────────────────────────────────────────────────────────────────────────

#     if CMD == "build":
#         index_path = build_index(OUTPUT_DIR, SETTINGS, force=FORCE)
#         print(f"[DONE] 向量索引建立完成：{index_path}")

#     elif CMD == "search":
#         searcher = VectorSearcher(LANCEDB, SETTINGS)
        
#         for table in ("text_units", "entities", "communities"):
#             method = {
#                 "text_units":  searcher.search_text_units,
#                 "entities":    searcher.search_entities,
#                 "communities": searcher.search_communities,
#             }[table]
#             results = method(QUERY, top_k=TOP_K)
#             print(f"\n--- [{table}] Top-{TOP_K} ---")
#             for rank, r in enumerate(results, 1):
#                 score = r.get("score", "?")
#                 if table == "text_units":
#                     print(f"\n[{rank}] score={score:.4f}  {r.get('source_video','')} "
#                         f"{r.get('start_time',0):.1f}s-{r.get('end_time',0):.1f}s")
#                     print(r.get("text", "")[:200])
#                 elif table == "entities":
#                     print(f"[{rank}] score={score:.4f}  [{r.get('type','')}] {r.get('name','')}")
#                     print(f"     {r.get('description','')[:120]}")
#                 else:
#                     print(f"[{rank}] score={score:.4f}  {r.get('title','')}")
#                     print(f"     {r.get('summary','')[:200]}")
                
