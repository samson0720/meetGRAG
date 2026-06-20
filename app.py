"""
app.py
======
meetGRAG FastAPI 後端，將 qa_Module 問答管線包裝為 HTTP API。

啟動方式
--------
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
  uvicorn app:app --host 127.0.0.1 --port 9000 --reload

端點
----
  POST /api/v1/query         — 主問答端點（接收查詢，回傳回覆 + 引用來源）
  GET  /api/v1/health        — 健康檢查
  GET  /api/v1/graph         — 回傳知識圖譜資料（nodes / links / communities）
  POST /api/v1/index         — 觸發離線索引管線（背景執行）
  GET  /api/v1/index/status  — 查詢索引進度

OpenAPI 互動文件：http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import re
import urllib.parse

from fastapi import FastAPI, HTTPException, Request

# Strip the "[Meeting: ... | ...s ~ ...s]" prefix that indexer embeds for embedding context
_SOURCE_PREFIX_RE = re.compile(r'^\[Meeting:[^\]]*\]\s*')
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# ── 路徑設定 ──────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUTPUT_DIR    = _ROOT / "qa_Module/graphrag/output"
LANCEDB_PATH  = _ROOT / "qa_Module/graphrag/lancedb"
SETTINGS_PATH = _ROOT / "qa_Module/graphrag/settings.yaml"
LINK_JSON     = _ROOT / "database/meet_origin_data/link.json"

# Build meeting-name → YouTube URL mapping from link.json at import time
# key = meeting name (without .mp4), e.g. "IETF 125_ IAB Open"
# value = YouTube URL
def _load_meet_links() -> dict[str, str]:
    if not LINK_JSON.exists():
        return {}
    import json as _json
    try:
        entries = _json.loads(LINK_JSON.read_text(encoding="utf-8"))
        return {e["meet"]: e["link"] for e in entries if "meet" in e and "link" in e}
    except Exception:
        return {}

_MEET_LINK_MAP: dict[str, str] = _load_meet_links()


def _refresh_meet_links() -> None:
    global _MEET_LINK_MAP
    _MEET_LINK_MAP = _load_meet_links()

# ── 日誌 ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("meetGRAG.api")


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic 模型（Request / Response Schema）
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="使用者查詢字串")
    top_k: int = Field(5, ge=1, le=20, description="搜尋結果數量上限")
    score_cutoff: float = Field(-0.5, description="相關性分數門檻，低於此值的結果將被過濾")
    max_chunks: int = Field(5, ge=1, le=10, description="傳入 LLM 的最大片段數")
    dedup_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Jaccard 去重門檻（0~1，越小越嚴格）")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="LLM 生成溫度")
    max_tokens: int = Field(8192, ge=64, le=16384, description="LLM 最大生成 token 數")
    current_meeting: str | None = Field(None, description="使用者目前正在觀看的會議名稱（None = 不限定會議）")


class SourceRefOut(BaseModel):
    source_video: str
    start_time:   float
    end_time:     float
    slide_image:  str = ""
    text_snippet: str = ""
    video_url:    str = ""   # player page URL, e.g. http://localhost:9000/player?video=...


class CitationOut(BaseModel):
    chunk_id:     str
    chunk_type:   str                          # "text_unit" | "community"
    title:        str
    source_refs:  list[SourceRefOut]
    entity_names: list[str] = []              # 此 chunk 包含的實體名稱，供前端 highlight 圖譜節點


class QueryResponse(BaseModel):
    answer:      str
    query_type:  str                          # "local" | "global"
    is_fallback: bool
    citations:   list[CitationOut]
    usage:       dict[str, int]
    cited_entity_names: list[str] = Field(
        default_factory=list,
        description="被引用的 chunk 中包含的實體名稱列表，供前端建構知識子圖使用",
    )


class HealthResponse(BaseModel):
    status: str
    model:  str


class AnalyzeYoutubeRequest(BaseModel):
    url: str = Field(..., min_length=1, description="YouTube URL to analyze")
    meeting_name: str | None = Field(None, description="Optional meeting name; defaults to YouTube title")
    auto_index: bool = Field(True, description="Run GraphRAG indexing after JSON export")


class AnalyzeYoutubeResponse(BaseModel):
    task_id: str
    status: str
    stage: str | None = None
    progress: int = 0
    message: str = ""
    meeting_name: str | None = None
    video_url: str | None = None
    output_json: str | None = None
    total_slides: int | None = None
    error: str | None = None


class IndexRequest(BaseModel):
    run_document_loader: bool = Field(True,  description="重新從 database/ 載入並轉換 JSON → TXT")
    run_indexer:         bool = Field(True,  description="重新執行 GraphRAG 實體/關係/社群索引")
    run_vectorizer:      bool = Field(True,  description="重新建立 LanceDB 向量索引")
    force_vector:        bool = Field(False, description="強制重新嵌入（即使向量索引已存在）")


class IndexStatusResponse(BaseModel):
    status:      str            # "idle" | "running" | "done" | "failed"
    stage:       str | None     # "document_loading" | "indexing" | "vectorizing" | None
    started_at:  str | None     # ISO 8601
    finished_at: str | None     # ISO 8601
    stats:       dict | None    # 完成後的統計（text_units, entities, ...）
    error:       str | None     # 失敗時的錯誤訊息


# ══════════════════════════════════════════════════════════════════════════════
# 轉換工具
# ══════════════════════════════════════════════════════════════════════════════

def _to_response(ans, org=None) -> QueryResponse:
    """
    將 GeneratedAnswer dataclass 轉為 Pydantic 回應模型。

    Parameters
    ----------
    ans     GeneratedAnswer（generate() 的回傳值）
    org     OrganizedContext（organize() 的回傳值，可選）
            若提供，會從被引用的 chunk 中提取實體名稱，
            填入 cited_entity_names 供前端建構知識子圖。
    """
    # 先建立 chunk_id → entity names 的查找表
    chunk_entity_map: dict[str, list[str]] = {}
    if org is not None:
        for chunk in org.chunks:
            chunk_entity_map[chunk.chunk_id] = [
                e.name for e in chunk.entities if e.name
            ]

    citations_out: list[CitationOut] = []
    for c in ans.citations:
        refs_out = []
        for ref in c.source_refs:
            # source_video e.g. "IETF 125_ IAB Open.mp4" → strip ".mp4" → look up link.json
            meet_name = Path(ref.source_video).stem if ref.source_video else ""
            video_url = _MEET_LINK_MAP.get(meet_name, "")
            refs_out.append(SourceRefOut(
                source_video = ref.source_video,
                start_time   = ref.start_time,
                end_time     = ref.end_time,
                slide_image  = getattr(ref, "slide_image",  ""),
                text_snippet = _SOURCE_PREFIX_RE.sub('', getattr(ref, "text_snippet", "")).strip(),
                video_url    = video_url,
            ))
        citations_out.append(CitationOut(
            chunk_id     = c.chunk_id,
            chunk_type   = c.chunk_type,
            title        = c.title,
            source_refs  = refs_out,
            entity_names = chunk_entity_map.get(c.chunk_id, []),
        ))

    # ── 從被引用的 chunk 中收集精確的實體名稱 ──────────────────────────
    cited_entity_names: list[str] = []
    if org is not None:
        cited_ids = {c.chunk_id for c in ans.citations}
        seen: set[str] = set()
        for chunk in org.chunks:
            if chunk.chunk_id not in cited_ids:
                continue
            for e in chunk.entities:
                name = e.name
                if name and name not in seen:
                    seen.add(name)
                    cited_entity_names.append(name)

    return QueryResponse(
        answer              = ans.answer,
        query_type          = ans.query_type,
        is_fallback         = ans.is_fallback,
        citations           = citations_out,
        usage               = ans.usage or {},
        cited_entity_names  = cited_entity_names,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan（應用程式啟動 / 關閉）
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時初始化 LLM 與 GraphRAGSearcher（單例），關閉時清理。"""
    logger.info("初始化 meetGRAG 服務...")

    from qa_Module.graphrag.vectorizer import _load_settings
    from qa_Module.llm import create_llm_from_config
    from qa_Module.graphrag.searcher import GraphRAGSearcher

    settings = _load_settings(SETTINGS_PATH, SETTINGS_PATH.parent)
    llm_conf = dict(settings.get("llm", {}))

    app.state.llm      = create_llm_from_config(llm_conf)
    app.state.searcher = GraphRAGSearcher(OUTPUT_DIR, LANCEDB_PATH, SETTINGS_PATH)
    app.state.model    = getattr(app.state.llm, "model", "unknown")

    # 索引狀態（執行緒安全：用 threading.Lock 保護寫入）
    app.state.index_status = {
        "status":      "idle",
        "stage":       None,
        "started_at":  None,
        "finished_at": None,
        "stats":       None,
        "error":       None,
    }
    app.state.index_lock = threading.Lock()
    app.state.analysis_status = {}
    app.state.analysis_lock = threading.Lock()

    logger.info("服務就緒：LLM model=%s", app.state.model)
    yield
    logger.info("服務關閉")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 應用程式
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "meetGRAG QA API",
    description = "GraphRAG-based question answering with traceable source citations.",
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # 開發階段全開；正式部署改為指定域名
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# 端點
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request):
    """確認服務是否正常運作。"""
    return HealthResponse(
        status = "ok",
        model  = getattr(request.app.state, "model", "unknown"),
    )


@app.get("/api/v1/meetings", tags=["meetings"])
async def list_meetings():
    """
    回傳 link.json 中所有會議的名稱與 YouTube URL，
    供 Extension 側邊欄偵測目前觀看的會議。
    """
    return [{"name": name, "url": url} for name, url in _MEET_LINK_MAP.items()]


@app.post("/api/v1/query", response_model=QueryResponse, tags=["qa"])
async def query_endpoint(req: QueryRequest, request: Request):
    """
    接收使用者查詢，執行 GraphRAG 問答管線，回傳回覆與可溯源的引用來源。

    **回應說明**
    - `answer`：LLM 生成的回覆，含行內 `[REF:chunk_id]` 標注
    - `citations`：解析後的引用列表；每筆的 `source_refs` 含影片名稱與起止時間戳，
      可供前端生成「跳轉到第 N 秒」功能
    - `is_fallback`：若 `true` 表示找不到相關資料或 LLM 呼叫失敗
    """
    llm      = request.app.state.llm
    searcher = request.app.state.searcher

    if llm is None or searcher is None:
        raise HTTPException(status_code=503, detail="服務尚未初始化，請稍後再試")

    from qa_Module.query_processor import process_query
    from qa_Module.retriever import retrieve
    from qa_Module.organizer import organize
    from qa_Module.generator import generate

    try:
        # 管線皆為同步函式，用 asyncio.to_thread 避免阻塞 event loop
        def _run_pipeline():
            # 若使用者目前正在觀看特定會議，在查詢前綴中注入該會議的上下文，
            # 讓 query_processor 生成會議聚焦的 expanded_query 以導引向量檢索
            if req.current_meeting:
                effective_query = (
                    f'[Context: The user is currently watching the meeting '
                    f'"{req.current_meeting}". Prioritize content from this '
                    f'specific meeting unless the question clearly requires '
                    f'information from other meetings.]\n'
                    f'Question: {req.query}'
                )
            else:
                effective_query = req.query

            if req.current_meeting:
                logger.info("查詢範圍：%s", req.current_meeting)
            else:
                logger.info("查詢範圍：全部會議")
            qr = process_query(effective_query, llm)
            qr.query = req.query   # 還原原始查詢，供 generate() 用於回覆生成
            meeting_filter = [req.current_meeting] if req.current_meeting else None
            ctx = retrieve(qr, searcher, top_k=req.top_k, meeting_filter=meeting_filter)
            org = organize(
                ctx,
                query            = req.query,
                score_cutoff     = req.score_cutoff,
                max_chunks       = req.max_chunks,
                dedup_threshold  = req.dedup_threshold,
            )
            ans = generate(
                org,
                query       = req.query,
                llm         = llm,
                temperature = req.temperature,
                max_tokens  = req.max_tokens,
            )
            return ans, org     # 同時回傳 org，供提取引用實體名稱

        ans, org = await asyncio.to_thread(_run_pipeline)

    except Exception as exc:
        logger.exception("管線執行失敗：%s", exc)
        raise HTTPException(status_code=502, detail=f"問答管線執行失敗：{exc}")

    return _to_response(ans, org)


# ══════════════════════════════════════════════════════════════════════════════
# 圖譜資料端點
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/graph", tags=["graph"])
async def get_graph():
    """
    回傳完整知識圖譜資料，供前端 D3.js 可視化使用。

    **回應結構**
    - `nodes`  — 實體節點（id, name, type, description, community_id）
    - `links`  — 關係邊（source, target, weight, description）
    - `communities` — 社群列表（id, title, level）
    """
    from qa_Module.graphrag.storage import load_table
    import json as _json

    def _parse_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                r = _json.loads(v)
                return r if isinstance(r, list) else []
            except Exception:
                return []
        return []

    entities    = load_table("entities",      OUTPUT_DIR)
    rels        = load_table("relationships", OUTPUT_DIR)
    communities = load_table("communities",   OUTPUT_DIR)
    text_units  = load_table("text_units",   OUTPUT_DIR)

    # tu_id → meeting name（strip .mp4 副檔名）
    tu_meeting: dict[str, str] = {
        tu["id"]: Path(tu.get("source_video", "")).stem
        for tu in text_units
        if tu.get("id") and tu.get("source_video")
    }

    # 建立 entity_name → community_id 對照表（取最低 level 社群）
    name_to_comm: dict[str, str] = {}
    for comm in sorted(communities, key=lambda c: int(c.get("level", 0))):
        for key in _parse_list(comm.get("entity_names", [])):
            pure_name = key.split("|")[0].strip().lower()
            if pure_name not in name_to_comm:
                name_to_comm[pure_name] = str(comm["id"])

    nodes = [
        {
            "id":           e["name"],
            "type":         e.get("type", "UNKNOWN"),
            "description":  e.get("description", ""),
            "community_id": name_to_comm.get(e["name"].lower(), ""),
            # 該實體出現在哪些會議（透過 text_unit_ids 追溯 source_video）
            "meetings":     sorted({
                tu_meeting[tid]
                for tid in _parse_list(e.get("text_unit_ids", []))
                if tid in tu_meeting
            }),
        }
        for e in entities
    ]

    # 只保留兩端點都存在於節點集合中的關係邊，避免 D3 forceLink 找不到節點而崩潰
    node_ids = {n["id"] for n in nodes}
    links = [
        {
            "source":      r["source"],
            "target":      r["target"],
            "weight":      float(r.get("weight", 1.0)),
            "description": r.get("description", ""),
        }
        for r in rels
        if r.get("source") and r.get("target")
        and r["source"] in node_ids and r["target"] in node_ids
    ]

    comms_out = [
        {
            "id":    str(c["id"]),
            "title": c.get("title", ""),
            "level": int(c.get("level", 0)),
        }
        for c in communities
    ]

    return {"nodes": nodes, "links": links, "communities": comms_out}


# ══════════════════════════════════════════════════════════════════════════════
# 投影片片段端點（供 Extension 側邊欄使用）
# ══════════════════════════════════════════════════════════════════════════════

class _VisualInfo(BaseModel):
    title:   str
    content: list[str]

class _MultimodalContent(BaseModel):
    visual_info:      _VisualInfo
    audio_transcript: str

class _SlideTimeRange(BaseModel):
    start_sec:         float
    end_sec:           float
    display_timestamp: str

class SlideSegment(BaseModel):
    slide_index:        int
    video_name:         str
    slide_image:        str = ""
    time_range:         _SlideTimeRange
    multimodal_content: _MultimodalContent

class SlidesResponse(BaseModel):
    slides: list[SlideSegment]
    total:  int


def _load_slide_records(database_dir: Path) -> list[dict]:
    """
    掃描 database_dir 下的所有 JSON 檔，支援兩種格式：
      1. 根目錄 *.json          → 內容為 list[dict]
      2. 子目錄 */transcript.json → 內容為 {"slides": list[dict]}
    回傳合併後的原始 record list（未排序）。
    """
    import json as _json

    records: list[dict] = []

    # 根目錄 flat list JSON
    for f in sorted(database_dir.glob("*.json")):
        if f.name == "link.json":
            continue
        try:
            data = _json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(data)
        except Exception as exc:
            logger.warning("無法載入 %s：%s", f.name, exc)

    # 子目錄 transcript.json
    for f in sorted(database_dir.glob("*/transcript.json")):
        try:
            data = _json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("slides"), list):
                records.extend(data["slides"])
            elif isinstance(data, list):
                records.extend(data)
        except Exception as exc:
            logger.warning("無法載入 %s：%s", f, exc)

    deduped: list[dict] = []
    seen: set[tuple[str, str, float]] = set()
    for rec in records:
        try:
            start = float(rec.get("start_time") or 0.0)
        except Exception:
            start = 0.0
        key = (str(rec.get("video_name") or ""), str(rec.get("id") or ""), start)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    return deduped


def _record_to_segment(rec: dict, index: int) -> SlideSegment:
    """將原始 JSON record 轉換為 SlideSegment。"""
    video_name  = rec.get("video_name", "unknown")
    raw_start   = rec.get("start_time")
    start       = float(raw_start) if raw_start is not None else 0.0
    raw_end     = rec.get("end_time")
    end         = float(raw_end)   if raw_end   is not None else start + 1800.0
    slide_image = rec.get("slide_image") or ""

    # OCR 文字：首行作為標題，其餘非空行作為內容 bullet
    ocr    = (rec.get("ocr_text") or "").strip()
    lines  = [l.strip() for l in ocr.splitlines() if l.strip()]
    title   = lines[0] if lines else f"Segment {index}"
    content = lines[1:] if len(lines) > 1 else []

    # chart_description 補充到 content
    chart = (rec.get("chart_description") or "").strip()
    if chart and chart.lower() not in ("string or null", "null", "none"):
        content.append(chart)

    # 時間戳格式化 HH:MM:SS
    h = int(start // 3600)
    m = int((start % 3600) // 60)
    s = int(start % 60)

    return SlideSegment(
        slide_index  = index,
        video_name   = video_name,
        slide_image  = slide_image,
        time_range   = _SlideTimeRange(
            start_sec         = start,
            end_sec           = end,
            display_timestamp = f"{h:02d}:{m:02d}:{s:02d}",
        ),
        multimodal_content = _MultimodalContent(
            visual_info = _VisualInfo(title=title, content=content),
            audio_transcript = (rec.get("transcript") or "").strip(),
        ),
    )


@app.get("/api/v1/slides", response_model=SlidesResponse, tags=["slides"])
async def get_slides(video_name: str | None = None):
    """
    從 database/meet_origin_data/ 載入所有會議片段，回傳給 Extension 側邊欄。

    **支援格式**
    - `database/*.json`               — 根目錄 flat list
    - `database/*/transcript.json`    — 子目錄 `{"slides": [...]}`

    **參數**
    - `video_name`（可選）— 過濾指定影片名稱，省略則回傳全部。

    **回應**
    - `slides` — 依 `(video_name, start_sec)` 排序的 `SlideSegment` 列表
    - `total`  — 片段總數
    """
    DATABASE_DIR = _ROOT / "database/meet_origin_data"

    raw = await asyncio.to_thread(_load_slide_records, DATABASE_DIR)

    segments: list[SlideSegment] = []
    for i, rec in enumerate(raw, start=1):
        if video_name and rec.get("video_name") != video_name:
            continue
        try:
            segments.append(_record_to_segment(rec, i))
        except Exception as exc:
            logger.warning("record %d 轉換失敗：%s", i, exc)

    segments.sort(key=lambda seg: (seg.video_name, seg.time_range.start_sec))

    # 重新指派 slide_index（依排序後順序）
    for idx, seg in enumerate(segments, start=1):
        seg.slide_index = idx

    return SlidesResponse(slides=segments, total=len(segments))


# ══════════════════════════════════════════════════════════════════════════════
# 索引端點
# ══════════════════════════════════════════════════════════════════════════════

def _run_index_pipeline(app_state, req: IndexRequest) -> None:
    """
    在背景執行緒中依序執行三個索引步驟，並即時更新 app_state.index_status。

    步驟
    ----
    1. document_loading  — database/*.json → graphrag/input/*.txt
    2. indexing          — 實體 / 關係 / 社群 GraphRAG 索引
    3. vectorizing       — LanceDB 向量嵌入
    """
    status  = app_state.index_status
    lock    = app_state.index_lock

    def _set(**kwargs):
        with lock:
            status.update(kwargs)

    with lock:
        if status.get("status") == "running":
            raise RuntimeError("GraphRAG indexing is already running")
        status.update(
            status      = "running",
            started_at  = datetime.now(timezone.utc).isoformat(),
            finished_at = None,
            stats       = None,
            error       = None,
        )

    try:
        DATABASE_DIR = _ROOT / "database/meet_origin_data"
        INPUT_DIR    = _ROOT / "qa_Module/graphrag/input"

        # ── Step 1: Document Loading ──────────────────────────────────────────
        if req.run_document_loader:
            _set(stage="document_loading")
            logger.info("[index] Step 1/3 document_loading 開始")
            from qa_Module.graphrag.document_loader import run as dl_run
            dl_run(DATABASE_DIR, INPUT_DIR)
            logger.info("[index] Step 1/3 document_loading 完成")

        # ── Step 2: GraphRAG Indexing ─────────────────────────────────────────
        if req.run_indexer:
            _set(stage="indexing")
            logger.info("[index] Step 2/3 indexing 開始")
            from qa_Module.graphrag.indexer import run_indexing
            stats = run_indexing(
                input_dir     = INPUT_DIR,
                output_dir    = OUTPUT_DIR,
                settings_path = SETTINGS_PATH,
            )
            logger.info("[index] Step 2/3 indexing 完成：%s", stats)
        else:
            stats = {}

        # ── Step 3: Vector Embedding ──────────────────────────────────────────
        if req.run_vectorizer:
            _set(stage="vectorizing")
            logger.info("[index] Step 3/3 vectorizing 開始")
            from qa_Module.graphrag.vectorizer import build_index
            build_index(OUTPUT_DIR, SETTINGS_PATH, force=req.force_vector)
            logger.info("[index] Step 3/3 vectorizing 完成")

        # ── 重置 searcher 的懶載入快取，使下次查詢讀取最新資料 ───────────────
        searcher = app_state.searcher
        if searcher is not None:
            searcher._entity_list  = None
            searcher._entity_by_name = {}
            searcher._tu_by_id     = None
            searcher._adj_graph    = None
            searcher._vector_searcher = None   # 強制重建 VectorSearcher
            logger.info("[index] GraphRAGSearcher 快取已清除，下次查詢將重新載入")

        _set(
            status      = "done",
            stage       = None,
            finished_at = datetime.now(timezone.utc).isoformat(),
            stats       = stats,
        )

    except Exception as exc:
        logger.exception("[index] 索引管線失敗：%s", exc)
        _set(
            status      = "failed",
            stage       = None,
            finished_at = datetime.now(timezone.utc).isoformat(),
            error       = str(exc),
        )


@app.post("/api/v1/index", response_model=IndexStatusResponse, tags=["index"])
async def start_index(req: IndexRequest, request: Request):
    """
    觸發離線索引管線（背景執行，立即回傳）。

    依序執行：
    1. **document_loading** — 將 `database/` 的 JSON 轉為 TXT 輸入
    2. **indexing** — GraphRAG 實體 / 關係 / 社群報告索引
    3. **vectorizing** — 建立 LanceDB 向量索引

    可透過 `GET /api/v1/index/status` 查詢進度。
    """
    with request.app.state.index_lock:
        if request.app.state.index_status["status"] == "running":
            raise HTTPException(status_code=409, detail="索引管線正在執行中，請等待完成後再觸發")

    # 在背景執行緒啟動（不阻塞 event loop）
    t = threading.Thread(
        target   = _run_index_pipeline,
        args     = (request.app.state, req),
        daemon   = True,
        name     = "indexing-pipeline",
    )
    t.start()

    with request.app.state.index_lock:
        s = dict(request.app.state.index_status)
    return IndexStatusResponse(**s)


def _set_analysis_status(app_state, task_id: str, **kwargs) -> None:
    with app_state.analysis_lock:
        current = dict(app_state.analysis_status.get(task_id, {}))
        current.update(kwargs)
        app_state.analysis_status[task_id] = current


def _run_youtube_analysis_pipeline(app_state, task_id: str, req: AnalyzeYoutubeRequest) -> None:
    def progress(stage: str, pct: int, message: str) -> None:
        _set_analysis_status(
            app_state,
            task_id,
            stage=stage,
            progress=max(0, min(100, int(pct))),
            message=message,
        )

    try:
        from qa_Module.multimodal.youtube_analyzer import analyze_youtube_to_meetgrag_json

        _set_analysis_status(
            app_state,
            task_id,
            status="running",
            stage="queued",
            progress=0,
            message="Starting YouTube analysis...",
            video_url=req.url,
        )

        result = analyze_youtube_to_meetgrag_json(
            video_url=req.url,
            meeting_name=req.meeting_name,
            database_dir=_ROOT / "database/meet_origin_data",
            progress=progress,
        )
        _refresh_meet_links()

        _set_analysis_status(
            app_state,
            task_id,
            status="indexing" if req.auto_index else "done",
            stage="indexing" if req.auto_index else "done",
            progress=92 if req.auto_index else 100,
            message="Running GraphRAG indexing..." if req.auto_index else "Analysis complete.",
            meeting_name=result.meeting_name,
            video_url=result.video_url,
            output_json=str(result.json_path),
            total_slides=result.total_slides,
        )

        if req.auto_index:
            _run_index_pipeline(
                app_state,
                IndexRequest(
                    run_document_loader=True,
                    run_indexer=True,
                    run_vectorizer=True,
                    force_vector=True,
                ),
            )
            with app_state.index_lock:
                index_status_copy = dict(app_state.index_status)
            if index_status_copy.get("status") == "failed":
                raise RuntimeError(index_status_copy.get("error") or "GraphRAG indexing failed")
            _set_analysis_status(
                app_state,
                task_id,
                status="done",
                stage="done",
                progress=100,
                message="Analysis and GraphRAG indexing complete.",
            )

    except Exception as exc:
        logger.exception("[analyze-youtube] task %s failed: %s", task_id, exc)
        _set_analysis_status(
            app_state,
            task_id,
            status="failed",
            stage=None,
            message="Analysis failed.",
            error=str(exc),
        )


@app.post("/api/v1/analyze-youtube", response_model=AnalyzeYoutubeResponse, tags=["ingestion"])
async def start_youtube_analysis(req: AnalyzeYoutubeRequest, request: Request):
    with request.app.state.index_lock:
        if req.auto_index and request.app.state.index_status["status"] == "running":
            raise HTTPException(status_code=409, detail="GraphRAG indexing is already running")

    task_id = uuid.uuid4().hex[:8]
    initial = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "Queued.",
        "video_url": req.url,
    }
    with request.app.state.analysis_lock:
        request.app.state.analysis_status[task_id] = initial

    t = threading.Thread(
        target=_run_youtube_analysis_pipeline,
        args=(request.app.state, task_id, req),
        daemon=True,
        name=f"youtube-analysis-{task_id}",
    )
    t.start()
    return AnalyzeYoutubeResponse(**initial)


@app.get("/api/v1/analyze-youtube/{task_id}", response_model=AnalyzeYoutubeResponse, tags=["ingestion"])
async def youtube_analysis_status(task_id: str, request: Request):
    with request.app.state.analysis_lock:
        status = request.app.state.analysis_status.get(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Analysis task not found")
        return AnalyzeYoutubeResponse(**status)


@app.get("/api/v1/index/status", response_model=IndexStatusResponse, tags=["index"])
async def index_status(request: Request):
    """查詢索引管線目前的執行狀態與進度。"""
    with request.app.state.index_lock:
        s = dict(request.app.state.index_status)
    return IndexStatusResponse(**s)
