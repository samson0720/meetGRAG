"""
indexer.py
==========
自定義 GraphRAG 索引管線。功能等同 Microsoft GraphRAG，但完全自行控制，
可在無 Neo4j / LanceDB 的環境下降級為本地 JSON 儲存。

流程：
  1. 解析 graphrag/input/*.txt（含 [SOURCE] 溯源標注）
  2. 文字分塊（TextUnit，~600 tokens）
  3. LLM 實體與關係擷取
  4. 建立 NetworkX 圖譜
  5. Leiden / Louvain 社群偵測
  6. LLM 社群報告生成
  7. 寫入 Neo4j（若可用）
  8. 嵌入向量化 → 寫入 LanceDB（若可用）
  9. 降級備份：所有資料同時寫出至 graphrag/output/*.json

執行方式（meetGRAG 根目錄）：
    python -m qa_Module.graphrag.indexer
    python -m qa_Module.graphrag.indexer --input qa_Module/graphrag/input --verbose
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# ── 路徑常數 ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR  = _ROOT / "qa_Module" / "graphrag" / "input"
DEFAULT_OUTPUT_DIR = _ROOT / "qa_Module" / "graphrag" / "output"
DEFAULT_SETTINGS   = _ROOT / "qa_Module" / "graphrag" / "settings.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# 資料結構
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextUnit:
    id: str
    text: str        # embed_text：source prefix + overlap + chunk body（用於向量嵌入）
    raw_text: str    # 純 chunk body，無 prefix 無 overlap（用於 LLM 實體擷取）
    source_file: str
    source_video: str
    start_time: float
    end_time: float
    slide_image: str = ""
    entity_ids: list[str] = field(default_factory=list)


@dataclass
class Entity:
    id: str
    name: str
    type: str           # PROTOCOL, ORGANIZATION, PERSON, CONCEPT, DOCUMENT, OTHER
    description: str
    text_unit_ids: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    id: str
    source: str             # entity name
    target: str             # entity name
    rel_type: str           # USES, DEFINES, EXTENDS, PART_OF, DISCUSSES, RELATED_TO, ...
    description: str
    weight: float = 1.0
    is_directional: bool = True   # False = symmetric（如 RELATED_TO）
    text_unit_ids: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)  # 累積所有原始描述


@dataclass
class Community:
    id: str
    level: int
    entity_names: list[str]
    text_unit_ids: list[str]
    title: str = ""
    summary: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: TXT 解析
# ══════════════════════════════════════════════════════════════════════════════

_SOURCE_PATTERN = re.compile(
    r"\[SOURCE:\s*(?P<video>[^,]+),\s*"
    r"START:\s*(?P<start>[\d.]+),\s*"
    r"END:\s*(?P<end>[\d.]+)"
    r"(?:,\s*ID:\s*(?P<id>[^\]]+))?\]"
)


def parse_source_header(txt_path: Path, source_map: dict) -> dict:
    """從 [SOURCE] 標注或 source_map 取得溯源資訊。"""
    filename = txt_path.name
    # 先從 source_map 查
    if filename in source_map:
        m = source_map[filename]
        return {
            "source_video": m.get("source_video", ""),
            "start_time": float(m.get("start_time", 0.0)),
            "end_time": float(m.get("end_time", 0.0)),
            "slide_image": m.get("slide_image", ""),
        }
    # 解析檔案第一行的 [SOURCE] 標注
    try:
        first_line = txt_path.read_text(encoding="utf-8").split("\n")[0]
        m = _SOURCE_PATTERN.search(first_line)
        if m:
            return {
                "source_video": m.group("video").strip(),
                "start_time": float(m.group("start")),
                "end_time": float(m.group("end")),
                "slide_image": "",
            }
    except OSError:
        pass
    return {"source_video": filename, "start_time": 0.0, "end_time": 0.0, "slide_image": ""}


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 文字分塊
# ══════════════════════════════════════════════════════════════════════════════

def _approx_tokens(text: str) -> int:
    """粗估 token 數（約 4 字元 = 1 token）。"""
    return len(text) // 4


def _tail_tokens(text: str, n_tokens: int) -> str:
    """取文字末尾約 n_tokens 個 token 的內容（用於 overlap）。"""
    return text[-(n_tokens * 4):]


def _build_chunks(body: str, chunk_size: int) -> list[str]:
    """
    將 body 依段落合併切割為不超過 chunk_size 的文字塊。

    切割策略（優先順序）：
      1. 段落（雙換行）為最小合併單位
      2. 單一段落超過 chunk_size → 按句子切割
      3. 單一句子超過 chunk_size → 按字元硬切
    """
    paragraphs = re.split(r"\n{2,}", body)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = (current + "\n\n" + para) if current else para
        if _approx_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            # 先把 current flush 掉
            if current:
                chunks.append(current)
                current = ""        # ← Bug fix: 必須重置，否則下一輪 candidate 會雙重包含

            # 判斷這個段落本身是否超過 chunk_size
            if _approx_tokens(para) > chunk_size:
                # 按句子切割
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub = ""
                for sent in sentences:
                    trial = (sub + " " + sent).strip() if sub else sent
                    if _approx_tokens(trial) <= chunk_size:
                        sub = trial
                    else:
                        if sub:
                            chunks.append(sub)
                        # 單句仍超過 chunk_size → 硬切
                        if _approx_tokens(sent) > chunk_size:
                            step = chunk_size * 4
                            for k in range(0, len(sent), step):
                                chunks.append(sent[k : k + step])
                            sub = ""
                        else:
                            sub = sent
                if sub:
                    chunks.append(sub)
                # 段落已全數 flush，current 保持 ""
            else:
                current = para      # 段落本身未超過，作為下一輪的起點

    if current:
        chunks.append(current)

    return chunks if chunks else [body]


def chunk_text(
    text: str,
    source_meta: dict,
    source_file: str,
    chunk_size: int = 600,
    overlap: int = 0,
) -> list[TextUnit]:
    """
    將文字依 token 數分塊，每塊建立一個 TextUnit。

    改善項目
    --------
    1. 修正 current 未重置的 bug（原本會導致 chunk 內容重疊）
    2. 實作 overlap：每個 chunk 的開頭附加前一個 chunk 的末尾片段
    3. 每個 chunk 加上 source context prefix，提升向量檢索準確度
    """
    # 移除 [SOURCE] 標頭行
    lines = text.split("\n")
    body_lines = [l for l in lines if not _SOURCE_PATTERN.match(l.strip())]
    body = "\n".join(body_lines).strip()

    if not body:
        return []

    chunks = _build_chunks(body, chunk_size)

    # Source context prefix：讓 embedding 攜帶「哪場會議、哪個時間段」的上下文
    video_name = source_meta.get("source_video", "")
    start_t    = source_meta.get("start_time", 0.0)
    end_t      = source_meta.get("end_time", 0.0)
    source_prefix = f"[Meeting: {video_name} | {start_t:.1f}s ~ {end_t:.1f}s]"

    units: list[TextUnit] = []
    for i, chunk in enumerate(chunks):
        # Overlap：在 chunk 開頭加上前一個 chunk 的末尾片段（不含 prefix 本身）
        if i > 0 and overlap > 0:
            tail = _tail_tokens(chunks[i - 1], overlap)
            chunk_with_overlap = tail + "\n\n" + chunk
        else:
            chunk_with_overlap = chunk

        # embed_text：source prefix + overlap + chunk 本文（向量嵌入用）
        embed_text = source_prefix + "\n\n" + chunk_with_overlap

        unit_id = hashlib.md5(f"{source_file}:{i}:{chunk[:64]}".encode()).hexdigest()[:16]
        units.append(TextUnit(
            id=unit_id,
            text=embed_text,       # 向量嵌入用（含 prefix 與 overlap）
            raw_text=chunk,        # LLM 實體擷取用（純 chunk 本文，無截斷風險）
            source_file=source_file,
            source_video=source_meta["source_video"],
            start_time=source_meta["start_time"],
            end_time=source_meta["end_time"],
            slide_image=source_meta.get("slide_image", ""),
        ))
    return units


# ══════════════════════════════════════════════════════════════════════════════
# 名稱正規化工具
# ══════════════════════════════════════════════════════════════════════════════

def _canonical_name(name: str) -> str:
    """
    正規化實體名稱以處理同義異形。
    步驟：
      1. 小寫
      2. 移除標點與多餘空白（保留 / 和 . 以處理 HTTP/3、TLS 1.3 等）
      3. 壓縮連續空白為單一空白
    例：
      "GPT-4"  → "gpt 4"
      "GPT4"   → "gpt4"       (不同，不強制合併)
      "GPT 4"  → "gpt 4"      (與 "GPT-4" 合併)
      "HTTP/3" → "http/3"     (保留 /)
    注意：語意層面的同義詞（QUIC vs quic-transport）需要 embedding，
          此函式只處理標點/空白變體。
    """
    s = name.lower().strip()
    # 連字號、底線視為空格（GPT-4 → GPT 4）
    s = re.sub(r"[-_]", " ", s)
    # 移除除 / 和 . 以外的標點
    s = re.sub(r"[^\w\s/.]", "", s)
    # 壓縮多餘空白
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: LLM 實體與關係擷取
# ══════════════════════════════════════════════════════════════════════════════

from prompts.indexer import EXTRACT_SYSTEM as _EXTRACT_SYSTEM
from prompts.indexer import EXTRACT_USER as _EXTRACT_USER


def extract_entities_relationships(
    unit: TextUnit,
    llm,
    max_retries: int = 3,
) -> tuple[list[Entity], list[Relationship]]:
    """呼叫 LLM 從單一 TextUnit 擷取實體與關係，回傳解析後的結構。"""
    from qa_Module.llm import Message

    # 使用 raw_text（純 chunk 本文），避免 source prefix + overlap 佔用 context 空間
    # chunk_size=600 tokens ≈ 2400 chars；保留 system prompt 空間，上限設 2400 chars
    prompt = _EXTRACT_USER.format(text=unit.raw_text[:2400])
    messages = [
        Message("system", _EXTRACT_SYSTEM),
        Message("user", prompt),
    ]

    # 抽取實體與關係
    raw = ""
    data = None
    for attempt in range(max_retries + 1):
        try:
            resp = llm.chat(messages, temperature=0.2, max_tokens=1024)
            raw = resp.content.strip()
           
            # 移除可能的 markdown fence
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            break
        except json.JSONDecodeError as exc:
            # LLM 輸出非法 JSON，重試（溫度 / 截斷問題）
            if attempt == max_retries:
                raise RuntimeError(f"TextUnit {unit.id} JSON 解析失敗（已重試 {max_retries} 次）") from exc
            logger.debug("JSON 解析失敗，重試 %d/%d：%s", attempt + 1, max_retries, exc)
            time.sleep(1)
        # 其他例外（RateLimitError、APITimeoutError 等）直接往上拋

    # 分類
    entities: list[Entity] = []
    relationships: list[Relationship] = []

    for e in data.get("entities", []):
        name = (e.get("name") or "").strip()
        etype = e.get("type", "OTHER").strip().upper()
        if not name:
            continue
        # Problem 1 fix: key = (canonical_name, type) — "Apple/ORG" ≠ "apple/CONCEPT"
        canonical = _canonical_name(name)
        eid = hashlib.md5(f"{canonical}|{etype}".encode()).hexdigest()[:12]
        desc = (e.get("description") or "").strip()
        entities.append(Entity(
            id=eid,
            name=name,       # 保留 LLM 原始大小寫作為顯示名稱
            type=etype,
            description=desc,
            text_unit_ids=[unit.id],
        ))

    for r in data.get("relationships", []):
        src = (r.get("source") or "").strip()
        tgt = (r.get("target") or "").strip()
        # Problem 2 fix: rel_type 加入 key，區分不同類型的關係
        rel_type = (r.get("rel_type") or "RELATED_TO").strip().upper()
        if not src or not tgt:
            continue
        rid = hashlib.md5(f"{_canonical_name(src)}|{rel_type}|{_canonical_name(tgt)}".encode()).hexdigest()[:12]
        desc = (r.get("description") or "").strip()
        is_dir = rel_type not in {"RELATED_TO", "PART_OF"}
        relationships.append(Relationship(
            id=rid,
            source=src,
            target=tgt,
            rel_type=rel_type,
            description=desc,
            weight=float(r.get("weight", 1.0)),
            is_directional=is_dir,
            text_unit_ids=[unit.id],
            descriptions=[desc] if desc else [],
        ))

    return entities, relationships


async def _extract_batch_async(
    units: list[TextUnit],
    llm,
    concurrency: int = 1,
    max_retries: int = 3,
) -> tuple[list[Entity], list[Relationship], list[str]]:
    """
    非同步批次擷取所有 TextUnit 的實體與關係。

    Parameters
    ----------
    units       所有待處理的 TextUnit 列表
    llm         LLM client（BaseLLMClient）
    concurrency 同時進行的 LLM API 呼叫上限
    max_retries 每個 TextUnit 的最大重試次數

    Returns
    -------
    (all_entities, all_relationships, failed_ids)
    """
    sem   = asyncio.Semaphore(concurrency)
    total = len(units)
    done  = {"n": 0}   # asyncio 單執行緒，dict 計數器無競爭問題

    async def _one(unit: TextUnit):
        failed = False
        try:
            async with sem:
                ents, rels = await asyncio.to_thread(
                    extract_entities_relationships, unit, llm, max_retries
                )
        except Exception as exc:
            logger.warning("TextUnit %s 擷取失敗（%s）", unit.id, exc)
            ents, rels, failed = [], [], True
        done["n"] += 1
        logger.info("[%d/%d] 擷取完成：%s", done["n"], total, unit.source_file[:40])
        return unit.id, ents, rels, failed

    raw = await asyncio.gather(*[_one(u) for u in units])

    all_entities: list[Entity] = []
    all_rels:     list[Relationship] = []
    failed_ids:   list[str] = []
    for uid, ents, rels, is_failed in raw:
        if is_failed:
            failed_ids.append(uid)
        all_entities.extend(ents)
        all_rels.extend(rels)
    return all_entities, all_rels, failed_ids


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 合併實體 / 關係（去除重複）
# ══════════════════════════════════════════════════════════════════════════════

def merge_entities(all_entities: list[Entity]) -> dict[str, Entity]:
    """
    Problem 1 fix: key = (canonical_name, type)，避免同名不同類的實體被合併。
    Problem 3 fix: 描述改為累積合併（去重），而非取較長者。
    dict key 格式："{canonical_name}|{type}"
    """
    merged: dict[str, Entity] = {}
    for e in all_entities:
        # key 同時含 type，"Apple|ORGANIZATION" ≠ "Apple|CONCEPT"
        key = f"{_canonical_name(e.name)}|{e.type.upper()}"
        if key in merged:
            existing = merged[key]
            existing.text_unit_ids = list(set(existing.text_unit_ids + e.text_unit_ids))
            # 累積描述
            if e.description and e.description not in existing.description:
                existing.description = existing.description + "; " + e.description if existing.description else e.description
        else:
            merged[key] = e
    return merged  # key="{canonical}|{type}", value=Entity


def merge_relationships(
    all_relationships: list[Relationship],
    entity_map: dict[str, Entity],
) -> dict[str, Relationship]:
    """
    key = source|rel_type|target，不同類型的關係保持獨立。
    驗證兩端實體存在於 entity_map（以 canonical|type 比對）。
    """
    # 建立 canonical_name → entity_map key 的快速查找表
    canonical_lookup: dict[str, str] = {
        _canonical_name(e.name): k
        for k, e in entity_map.items()
    }

    merged: dict[str, Relationship] = {}
    for r in all_relationships:
        src_canonical = _canonical_name(r.source)
        tgt_canonical = _canonical_name(r.target)
        # 驗證兩端實體存在
        if src_canonical not in canonical_lookup or tgt_canonical not in canonical_lookup:
            continue
        # key 包含 rel_type
        key = f"{src_canonical}|{r.rel_type}|{tgt_canonical}"
        if key in merged:
            existing = merged[key]
            existing.weight += r.weight
            existing.text_unit_ids = list(set(existing.text_unit_ids + r.text_unit_ids))
            # 累積描述列表
            if r.description and r.description not in existing.descriptions:
                existing.descriptions.append(r.description)
                existing.description = "; ".join(existing.descriptions)
        else:
            merged[key] = r
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: 社群偵測（Leiden / Louvain fallback）
# ══════════════════════════════════════════════════════════════════════════════

def detect_communities(
    entity_map: dict[str, Entity],
    rel_map: dict[str, Relationship],
    resolution: float = 1.0,
) -> list[Community]:
    """
    使用 NetworkX + community-louvain 進行社群偵測。
    若 python-louvain 未安裝則退回連通分量。
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("需要 networkx：pip install networkx")
        return []

    # entity_map key 格式為 "{canonical_name}|{type}"
    # 建立 canonical_name → entity_map key 的查找表（取 type 最靠前的那筆）
    canonical_to_key: dict[str, str] = {}
    for em_key in entity_map:
        canonical = em_key.split("|")[0]
        canonical_to_key.setdefault(canonical, em_key)

    G = nx.Graph()
    for em_key, e in entity_map.items():
        G.add_node(em_key, entity_id=e.id, entity_type=e.type, display_name=e.name)
    for r in rel_map.values():
        src_key = canonical_to_key.get(_canonical_name(r.source))
        tgt_key = canonical_to_key.get(_canonical_name(r.target))
        if src_key and tgt_key and G.has_node(src_key) and G.has_node(tgt_key):
            if G.has_edge(src_key, tgt_key):
                G[src_key][tgt_key]["weight"] += r.weight
            else:
                G.add_edge(src_key, tgt_key, weight=r.weight)

    if G.number_of_nodes() == 0:
        logger.warning("圖譜沒有節點，跳過社群偵測")
        return []

    # 嘗試 Louvain
    partition: dict[str, int] = {}
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight="weight", resolution=resolution)
        logger.info("社群偵測：Louvain，找到 %d 個社群", len(set(partition.values())))
    except ImportError:
        logger.warning("python-louvain 未安裝，退回連通分量（pip install python-louvain）")
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                partition[node] = i
        logger.info("社群偵測：連通分量，找到 %d 個社群", len(set(partition.values())))

    # 組裝 Community 物件
    community_entities: dict[int, list[str]] = {}
    for node, cid in partition.items():
        community_entities.setdefault(cid, []).append(node)

    communities: list[Community] = []
    for cid, node_names in community_entities.items():
        # 收集社群內所有 text_unit_ids
        tu_ids: list[str] = []
        for node in node_names:
            if node in entity_map:
                tu_ids.extend(entity_map[node].text_unit_ids)
        tu_ids = list(set(tu_ids))

        communities.append(Community(
            id=str(uuid.uuid4())[:8],
            level=0,
            entity_names=node_names,
            text_unit_ids=tu_ids,
        ))

    return communities


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: LLM 社群報告生成
# ══════════════════════════════════════════════════════════════════════════════

from prompts.indexer import REPORT_SYSTEM as _REPORT_SYSTEM
from prompts.indexer import REPORT_USER as _REPORT_USER


def generate_community_reports(
    communities: list[Community],
    entity_map: dict[str, Entity],
    llm,
    max_retries: int = 3,
) -> tuple[list[Community], list[str]]:
    """
    為每個社群呼叫 LLM 生成 title 與 summary，循序執行（每次等待完成再呼叫下一個）。

    Returns
    -------
    (communities, failed_ids)
        failed_ids：所有重試耗盡後仍失敗的 community.id 清單。
    """
    from qa_Module.llm import Message

    total = len(communities)
    failed_ids: list[str] = []

    for i, community in enumerate(communities):
        entity_lines = []
        for name in community.entity_names[:20]:
            e = entity_map.get(name)
            if e:
                entity_lines.append(f"- {e.name} ({e.type}): {e.description[:120]}")

        if not entity_lines:
            community.title   = f"Community {community.id}"
            community.summary = "No entity descriptions available."
            continue

        prompt = _REPORT_USER.format(entity_list="\n".join(entity_lines))
        messages = [
            Message("system", _REPORT_SYSTEM),
            Message("user", prompt),
        ]

        last_exc: Exception | None = None
        success = False
        for attempt in range(max_retries + 1):
            try:
                resp = llm.chat(messages, temperature=0.2, max_tokens=512)
                raw = resp.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                community.title   = data.get("title", f"Community {community.id}")
                community.summary = data.get("summary", "")
                logger.debug("社群 %d/%d 完成：%s", i + 1, total, community.title)
                success = True
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.debug("社群 %s 重試 %d/%d（%s），等待 %ds",
                                 community.id, attempt + 1, max_retries, exc, wait)
                    time.sleep(wait)

        if not success:
            logger.warning("社群 %s 報告生成失敗（已重試 %d 次）：%s",
                           community.id, max_retries, last_exc)
            community.title   = f"Community {community.id}"
            community.summary = ", ".join(community.entity_names[:5])
            failed_ids.append(community.id)

    return communities, failed_ids


async def _report_batch_async(
    communities: list[Community],
    entity_map: dict[str, Entity],
    llm,
    concurrency: int = 1,
    max_retries: int = 3,
) -> tuple[list[Community], list[str]]:
    """
    非同步批次生成所有社群的 title 與 summary。

    Parameters
    ----------
    communities 待處理的社群列表
    entity_map  merged entity_map（key = canonical|type）
    llm         LLM client（BaseLLMClient）
    concurrency 同時進行的 LLM API 呼叫上限
    max_retries 每個社群的最大重試次數（指數退避：1s / 2s / 4s）

    Returns
    -------
    (updated_communities, failed_ids)
    """
    from qa_Module.llm import Message
    sem   = asyncio.Semaphore(concurrency)
    total = len(communities)
    done  = {"n": 0}

    async def _one(community: Community) -> tuple[Community, bool]:
        try:
            return await _one_inner(community)
        except Exception as exc:
            logger.warning("社群 %s 報告生成例外（%s）", community.id, exc)
            community.title   = f"Community {community.id}"
            community.summary = ", ".join(community.entity_names[:5])
            return community, False

    async def _one_inner(community: Community) -> tuple[Community, bool]:
        entity_lines: list[str] = []
        for name in community.entity_names[:20]:
            e = entity_map.get(name)
            if e:
                entity_lines.append(f"- {e.name} ({e.type}): {e.description[:120]}")

        if not entity_lines:
            community.title   = f"Community {community.id}"
            community.summary = "No entity descriptions available."
            done["n"] += 1
            return community, True

        prompt   = _REPORT_USER.format(entity_list="\n".join(entity_lines))
        messages = [Message("system", _REPORT_SYSTEM), Message("user", prompt)]

        for attempt in range(max_retries + 1):
            try:
                async with sem:
                    resp = await asyncio.to_thread(
                        llm.chat, messages, temperature=0.2, max_tokens=512
                    )
                raw_c = resp.content.strip()
                raw_c = re.sub(r"^```(?:json)?\s*", "", raw_c)
                raw_c = re.sub(r"\s*```$",          "", raw_c)
                data  = json.loads(raw_c)
                community.title   = data.get("title",   f"Community {community.id}")
                community.summary = data.get("summary", "")
                done["n"] += 1
                logger.debug("社群 %d/%d 完成：%s", done["n"], total, community.title)
                return community, True
            except json.JSONDecodeError as exc:
                # LLM 輸出非法 JSON，重試
                if attempt == max_retries:
                    logger.warning("社群 %s JSON 解析失敗（已重試 %d 次）：%s",
                                   community.id, max_retries, exc)
                    break
                logger.debug("社群 %s JSON 解析失敗，重試 %d/%d", community.id, attempt + 1, max_retries)
                await asyncio.sleep(1)
            # API 例外（RateLimitError、APITimeoutError 等）直接往上拋，
            # 由 groq_client 的 key 輪換機制處理後再往上，_one 的外層 try/except 捕捉

        community.title   = f"Community {community.id}"
        community.summary = ", ".join(community.entity_names[:5])
        return community, False

    results  = await asyncio.gather(*[_one(c) for c in communities])
    updated  = [c       for c, _  in results]
    failed   = [c.id    for c, ok in results if not ok]
    logger.info("社群報告完成：%d / %d 個成功", total - len(failed), total)
    return updated, failed


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: 寫入 Neo4j
# ══════════════════════════════════════════════════════════════════════════════

# def store_neo4j(
#     text_units: list[TextUnit],
#     entity_map: dict[str, Entity],
#     rel_map: dict[str, Relationship],
#     communities: list[Community],
#     uri: str = "bolt://localhost:7687",
#     user: str = "neo4j",
#     password: str = "password",
# ) -> bool:
#     """
#     將索引結果寫入 Neo4j。若驅動未安裝或連線失敗則回傳 False。

#     節點標籤：TextUnit, Entity, Community
#     關係類型：MENTIONS, RELATED_TO, BELONGS_TO
#     """
#     try:
#         from neo4j import GraphDatabase
#     except ImportError:
#         logger.warning("neo4j 驅動未安裝，跳過（pip install neo4j）")
#         return False

#     try:
#         driver = GraphDatabase.driver(uri, auth=(user, password))
#         driver.verify_connectivity()
#     except Exception as exc:
#         logger.warning("Neo4j 連線失敗（%s），跳過", exc)
#         return False

#     with driver.session() as session:
#         # 清除舊資料（重新索引時）
#         session.run("MATCH (n) DETACH DELETE n")

#         # TextUnit 節點
#         for tu in text_units:
#             session.run(
#                 """
#                 CREATE (t:TextUnit {
#                     id: $id, text: $text,
#                     source_video: $source_video,
#                     start_time: $start_time,
#                     end_time: $end_time,
#                     slide_image: $slide_image,
#                     source_file: $source_file
#                 })
#                 """,
#                 id=tu.id, text=tu.text[:500],
#                 source_video=tu.source_video,
#                 start_time=tu.start_time,
#                 end_time=tu.end_time,
#                 slide_image=tu.slide_image,
#                 source_file=tu.source_file,
#             )

#         # Entity 節點
#         for e in entity_map.values():
#             session.run(
#                 """
#                 CREATE (e:Entity {
#                     id: $id, name: $name,
#                     type: $type, description: $description
#                 })
#                 """,
#                 id=e.id, name=e.name, type=e.type, description=e.description,
#             )

#         # MENTIONS 關係（Entity → TextUnit）
#         for e in entity_map.values():
#             for tu_id in e.text_unit_ids:
#                 session.run(
#                     """
#                     MATCH (e:Entity {id: $eid}), (t:TextUnit {id: $tid})
#                     CREATE (e)-[:MENTIONS]->(t)
#                     """,
#                     eid=e.id, tid=tu_id,
#                 )

#         # RELATED_TO 關係（Entity → Entity）
#         for r in rel_map.values():
#             src_e = entity_map.get(r.source.lower())
#             tgt_e = entity_map.get(r.target.lower())
#             if src_e and tgt_e:
#                 session.run(
#                     """
#                     MATCH (a:Entity {id: $sid}), (b:Entity {id: $tid})
#                     CREATE (a)-[:RELATED_TO {description: $desc, weight: $weight}]->(b)
#                     """,
#                     sid=src_e.id, tid=tgt_e.id,
#                     desc=r.description, weight=r.weight,
#                 )

#         # Community 節點 + BELONGS_TO
#         for c in communities:
#             session.run(
#                 """
#                 CREATE (c:Community {
#                     id: $id, title: $title, summary: $summary, level: $level
#                 })
#                 """,
#                 id=c.id, title=c.title, summary=c.summary, level=c.level,
#             )
#             for name in c.entity_names:
#                 e = entity_map.get(name)
#                 if e:
#                     session.run(
#                         """
#                         MATCH (e:Entity {id: $eid}), (c:Community {id: $cid})
#                         CREATE (e)-[:BELONGS_TO]->(c)
#                         """,
#                         eid=e.id, cid=c.id,
#                     )

#     driver.close()
#     logger.info("Neo4j 寫入完成（%d 個實體，%d 個社群）",
#                 len(entity_map), len(communities))
#     return True


# ══════════════════════════════════════════════════════════════════════════════
# Step 8: 嵌入向量化 + 寫入 LanceDB
# ══════════════════════════════════════════════════════════════════════════════

def embed_texts(texts: list[str], llm_client) -> list[list[float]]:
    """
    呼叫 Ollama / OpenAI embedding API 產生向量。
    llm_client 需有 _client (openai.OpenAI) 屬性。
    """
    client = getattr(llm_client, "_client", None)
    if client is None:
        raise RuntimeError("llm_client 沒有 _client 屬性，無法取得 embedding")

    # 取 settings 中的 embedding model
    embed_model = getattr(llm_client, "embed_model", "nomic-embed-text")
    embeddings = []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        resp = client.embeddings.create(model=embed_model, input=chunk)
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings


def store_lancedb(
    text_units: list[TextUnit],
    embeddings: list[list[float]],
    lancedb_path: str = "./qa_Module/graphrag/lancedb",
) -> bool:
    """將 text_unit 向量寫入 LanceDB。若未安裝則回傳 False。"""
    try:
        import lancedb
        import pyarrow as pa
    except ImportError:
        logger.warning("lancedb 未安裝，跳過（pip install lancedb pyarrow）")
        return False

    db = lancedb.connect(lancedb_path)

    dim = len(embeddings[0]) if embeddings else 1
    schema = pa.schema([
        pa.field("id",           pa.string()),
        pa.field("text",         pa.string()),
        pa.field("source_video", pa.string()),
        pa.field("start_time",   pa.float32()),
        pa.field("end_time",     pa.float32()),
        pa.field("slide_image",  pa.string()),
        pa.field("source_file",  pa.string()),
        pa.field("vector",       pa.list_(pa.float32(), dim)),
    ])

    data = [
        {
            "id":           tu.id,
            "text":         tu.text,
            "source_video": tu.source_video,
            "start_time":   tu.start_time,
            "end_time":     tu.end_time,
            "slide_image":  tu.slide_image,
            "source_file":  tu.source_file,
            "vector":       emb,
        }
        for tu, emb in zip(text_units, embeddings)
    ]

    table_name = "text_units"
    if table_name in db.table_names():
        db.drop_table(table_name)
    db.create_table(table_name, data=data, schema=schema)

    logger.info("LanceDB 寫入完成：%d 個向量 → %s", len(data), lancedb_path)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Step 9: 儲存結果（Parquet + JSON ）
# ══════════════════════════════════════════════════════════════════════════════

def save_output(
    text_units: list[TextUnit],
    entity_map: dict[str, Entity],
    rel_map: dict[str, Relationship],
    communities: list[Community],
    output_dir: Path,
) -> None:
    """索引結果持久化，委派至 storage.save()。"""
    from qa_Module.graphrag.storage import save as _save
    _save(text_units, entity_map, rel_map, communities, output_dir)


def load_output(output_dir: Path) -> dict[str, Any]:
    """載入已持久化的索引結果，委派至 storage.load()。"""
    from qa_Module.graphrag.storage import load as _load
    return _load(output_dir)



 


# ══════════════════════════════════════════════════════════════════════════════
# 設定載入
# ══════════════════════════════════════════════════════════════════════════════

def load_settings(settings_path: Path) -> dict:
    """載入 settings.yaml；若不存在則回傳預設值。"""
    defaults = {
        "llm": {"provider": "ollama", "model": "mistral"},
        "embeddings": {"model": "nomic-embed-text"},
        "chunking": {"size": 600, "overlap": 100},
        "storage": {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "lancedb_path": str(_ROOT / "qa_Module" / "graphrag" / "lancedb"),
        },
        "community": {"resolution": 1.0},
    }
    if not settings_path.exists():
        logger.warning("settings.yaml 不存在，使用預設值：%s", settings_path)
        return defaults
    try:
        import yaml
        with open(settings_path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        # 深度合併（只合併第一層 key）
        for k, v in user.items():
            if isinstance(v, dict) and k in defaults:
                defaults[k].update(v)
            else:
                defaults[k] = v
        return defaults
    except ImportError:
        logger.warning("pyyaml 未安裝，使用預設值（pip install pyyaml）")
        return defaults


# ══════════════════════════════════════════════════════════════════════════════
# 主索引管線
# ══════════════════════════════════════════════════════════════════════════════

async def _run_indexing_async(
    input_dir: Path,
    output_dir: Path,
    settings_path: Path,
    concurrency: int,
    embed_concurrency: int,
) -> dict:
    """
    非同步索引管線主函式，由 run_indexing() 透過 asyncio.run() 呼叫。
    實體擷取、社群報告生成、向量嵌入均以非同步並發方式執行。
    """
    settings   = load_settings(settings_path)
    chunk_size = settings["chunking"]["size"]
    overlap    = settings["chunking"]["overlap"]

    sys.path.insert(0, str(_ROOT))
    from qa_Module.llm import create_llm_from_config
    llm_cfg = dict(settings["llm"])
    llm = create_llm_from_config(llm_cfg)
    llm.embed_model = settings["embeddings"]["model"]

    logger.info("=== GraphRAG 索引管線啟動 ===")
    logger.info("LLM: %s / %s  (concurrency=%d  embed_concurrency=%d)",
                llm_cfg.get("provider"), llm_cfg.get("model"),
                concurrency, embed_concurrency)

    # ── 載入 source_map（支援平坦目錄與多層子資料夾） ──────────────────────────
    source_map: dict = {}
    source_map_files = [input_dir / "source_map.json"] + sorted(
        input_dir.glob("*/source_map.json")
    )
    loaded_maps = 0
    for sm_path in source_map_files:
        if sm_path.exists():
            source_map.update(json.loads(sm_path.read_text(encoding="utf-8")))
            loaded_maps += 1
    if loaded_maps == 0:
        logger.warning("找不到任何 source_map.json，將從 TXT 標頭解析")
    else:
        logger.info("載入 %d 個 source_map.json，共 %d 筆記錄", loaded_maps, len(source_map))

    from qa_Module.graphrag.storage import (
        save_text_units, save_entities, save_relationships,
        save_communities, save_meta,
    )

    # ── Step 1 & 2: 解析 + 分塊（純 CPU，同步執行） ──────────────────────────
    # rglob 同時支援平坦目錄（*.txt）與子資料夾（*/*.txt）
    txt_files = sorted(input_dir.rglob("*.txt"))
    logger.info("找到 %d 個 TXT 輸入檔案", len(txt_files))

    # TEST_FILE1  = DEFAULT_INPUT_DIR / "IETF118_ADD_session_add_001.txt"
    # TEST_FILE2  = DEFAULT_INPUT_DIR / "IETF118_ADD_session_add_002.txt"
    # TEST_FILE3  = DEFAULT_INPUT_DIR / "IETF118_ADD_session_add_003.txt"
    # TEST_FILE4  = DEFAULT_INPUT_DIR / "IETF118_ADD_session_add_004.txt"
    # txt_files = [TEST_FILE1, TEST_FILE2, TEST_FILE3, TEST_FILE4]

    all_text_units: list[TextUnit] = []
    for txt_path in txt_files:
        source_meta = parse_source_header(txt_path, source_map)
        text = txt_path.read_text(encoding="utf-8")
        units = chunk_text(text, source_meta, txt_path.name, chunk_size, overlap)
        all_text_units.extend(units)

    total_units = len(all_text_units)
    logger.info("分塊完成：%d 個 TextUnit", total_units)
    save_text_units(all_text_units, output_dir)                    # [儲存 1] text_units

    # ── Step 3: 非同步實體 / 關係擷取（含失敗重試） ──────────────────────────
    logger.info("開始非同步實體擷取（concurrency=%d）…", concurrency)
    all_entities, all_relationships, failed_extract_ids = await _extract_batch_async(
        all_text_units, llm, concurrency=concurrency,
    )

    _EXTRACT_RETRY_WAIT = 60   # 失敗重試等待秒數
    _EXTRACT_MAX_ROUNDS = 3    # 最多重試幾輪
    _id_to_unit = {u.id: u for u in all_text_units}

    for round_ in range(_EXTRACT_MAX_ROUNDS):
        if not failed_extract_ids:
            break
        logger.warning(
            "第 %d/%d 輪重試：%d 個 TextUnit 擷取失敗，等待 %ds 後重試…",
            round_ + 1, _EXTRACT_MAX_ROUNDS, len(failed_extract_ids), _EXTRACT_RETRY_WAIT,
        )
        await asyncio.sleep(_EXTRACT_RETRY_WAIT)
        retry_units = [_id_to_unit[uid] for uid in failed_extract_ids if uid in _id_to_unit]
        new_ents, new_rels, failed_extract_ids = await _extract_batch_async(
            retry_units, llm, concurrency=concurrency,
        )
        all_entities.extend(new_ents)
        all_relationships.extend(new_rels)
        logger.info("重試完成：新增 %d 實體、%d 關係，仍失敗 %d 筆",
                    len(new_ents), len(new_rels), len(failed_extract_ids))

    if failed_extract_ids:
        logger.warning("最終仍有 %d 個 TextUnit 擷取失敗，將跳過：%s",
                       len(failed_extract_ids), failed_extract_ids[:10])

    # ── Step 4: 合併去重（純 CPU，同步執行） ─────────────────────────────────
    entity_map = merge_entities(all_entities)
    rel_map    = merge_relationships(all_relationships, entity_map)
    logger.info("合併後：%d 個實體，%d 個關係", len(entity_map), len(rel_map))

    # 將 entity_ids 寫回 TextUnit
    name_to_tu: dict[str, list[str]] = {}
    for e in entity_map.values():
        for tu_id in e.text_unit_ids:
            name_to_tu.setdefault(tu_id, []).append(e.id)
    for tu in all_text_units:
        tu.entity_ids = name_to_tu.get(tu.id, [])

    save_text_units(all_text_units, output_dir)                    # [儲存 2] text_units（含 entity_ids）
    save_entities(entity_map, output_dir)                          # [儲存 3] entities
    save_relationships(rel_map, output_dir)                        # [儲存 4] relationships + graph_edges

    # ── Step 5: 社群偵測（純 CPU，同步執行） ─────────────────────────────────
    resolution  = settings["community"].get("resolution", 1.0)
    communities = detect_communities(entity_map, rel_map, resolution)
    logger.info("社群偵測：%d 個社群", len(communities))
    save_communities(communities, entity_map, rel_map, output_dir) # [儲存 5] communities（圖結構）

    # ── Step 6: 非同步社群報告生成 ───────────────────────────────────────────
    failed_report_ids: list[str] = []
    if communities:
        logger.info("開始非同步社群報告生成（concurrency=%d）…", concurrency)
        communities, failed_report_ids = await _report_batch_async(
            communities, entity_map, llm, concurrency=concurrency,
        )
        save_communities(communities, entity_map, rel_map, output_dir)  # [儲存 6] 更新 title/summary

    # ── Step 7: Neo4j（已停用） ───────────────────────────────────────────────
    # neo4j_cfg = settings["storage"]
    # neo4j_ok = store_neo4j(...)
    # logger.info("Neo4j %s", "寫入成功" if neo4j_ok else "不可用，已略過")

    # ── Step 8: 非同步向量化 + LanceDB ───────────────────────────────────────
    try:
        from qa_Module.graphrag.vectorizer import build_index_async as _build_vec_async
        logger.info("開始非同步向量化並建立 LanceDB 索引（embed_concurrency=%d）…",
                    embed_concurrency)
        lancedb_path = await _build_vec_async(
            output_dir, settings_path, force=True, concurrency=embed_concurrency,
        )
        logger.info("LanceDB 索引完成：%s", lancedb_path)
    except Exception as exc:
        logger.warning("向量化失敗（%s），已略過 LanceDB", exc)

    # ── 最終 meta ─────────────────────────────────────────────────────────────
    save_meta(all_text_units, entity_map, rel_map, communities, output_dir)

    # ── 失敗統計 ──────────────────────────────────────────────────────────────
    if failed_extract_ids:
        logger.warning("實體擷取失敗：%d / %d 個 TextUnit（id: %s）",
                       len(failed_extract_ids), total_units,
                       ", ".join(failed_extract_ids))
    else:
        logger.info("實體擷取：全部 %d 個 TextUnit 成功", total_units)

    if failed_report_ids:
        logger.warning("社群報告生成失敗：%d / %d 個 Community（id: %s）",
                       len(failed_report_ids), len(communities),
                       ", ".join(failed_report_ids))
    else:
        logger.info("社群報告生成：全部 %d 個 Community 成功", len(communities))

    summary = {
        "text_units":                len(all_text_units),
        "entities":                  len(entity_map),
        "relationships":             len(rel_map),
        "communities":               len(communities),
        "failed_extract_units":      len(failed_extract_ids),
        "failed_report_communities": len(failed_report_ids),
        "output_dir":                str(output_dir),
    }
    logger.info("=== 索引完成 %s ===", summary)
    return summary


def run_indexing(
    input_dir:         Path = DEFAULT_INPUT_DIR,
    output_dir:        Path = DEFAULT_OUTPUT_DIR,
    settings_path:     Path = DEFAULT_SETTINGS,
    verbose:           bool = False,
    concurrency:       int  = 1,
    embed_concurrency: int  = 1,
) -> dict:
    """
    執行完整非同步索引管線。

    Parameters
    ----------
    concurrency       LLM 呼叫（實體擷取 + 社群報告）的最大同時執行數
    embed_concurrency 向量嵌入的最大同時 batch 數

    Returns
    -------
    dict  { text_units, entities, relationships, communities } 統計摘要。
    """
    logging.basicConfig(
        format  = "%(asctime)s  %(levelname)-7s  %(message)s",
        level   = logging.DEBUG if verbose else logging.INFO,
        datefmt = "%H:%M:%S",
    )
    return asyncio.run(_run_indexing_async(
        input_dir, output_dir, settings_path, concurrency, embed_concurrency,
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 設定區（修改這裡）
# ══════════════════════════════════════════════════════════════════════════════

# 要索引的會議名稱（input_dir 會自動指向 input/{MEETING_NAME}/）
MEETING_NAME = "IETF 125_ IAB Open"

INPUT_DIR    = DEFAULT_INPUT_DIR / MEETING_NAME
OUTPUT_DIR   = DEFAULT_OUTPUT_DIR
SETTINGS     = DEFAULT_SETTINGS
VERBOSE      = False

if __name__ == "__main__":
    summary = run_indexing(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        settings_path=SETTINGS,
        verbose=VERBOSE,
    )
    print("\n索引摘要：")
    for k, v in summary.items():
        print(f"  {k:<16} {v}")
