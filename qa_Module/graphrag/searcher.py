"""
searcher.py
===========
GraphRAG 搜尋介面，封裝 Global Search 與 Local Search。

Global Search
-------------
以向量語意搜尋社群摘要（Community Reports），並加入圖遍歷擴展（Graph Traversal）：
  1. 向量搜尋取得初始社群候選（語意路徑）
  2. 提取候選社群的 entity_names 作為種子，BFS 遍歷 relationships 圖取得鄰近實體
  3. 掃描所有社群，計算圖分數（Graph Score）= 社群中實體與鄰近實體的重疊程度
  4. 最終分數 = α×向量分數 + (1-α)×圖分數，發現向量沒直接命中的相關社群

Local Search
------------
以向量語意搜尋文字塊（TextUnit），並反向查找關聯實體，
回傳 TextUnitResult 列表，每筆含來源影片/時間戳與相關實體。

資料來源：本機 output/ parquet 檔案 + LanceDB 向量索引（無需 Neo4j）。

公開 API
--------
  GraphRAGSearcher(output_dir, lancedb_path, settings_path)
      .global_search(query, top_k) -> list[CommunityResult]
      .local_search(query, top_k)  -> list[TextUnitResult]
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


def _parse_list_field(value) -> list:
    """將 parquet 欄位統一轉為 list（處理 JSON 字串或已是 list 的情況）。"""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []


# ══════════════════════════════════════════════════════════════════════════════
# 資料類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceRef:
    """可溯源的來源片段，對應原始影片時間段。"""
    text_unit_id: str
    source_video: str
    start_time: float
    end_time: float
    slide_image: str
    text_snippet: str   # 完整來源文字，供 UI 展開顯示


@dataclass
class EntitySnippet:
    """單一實體的精簡摘要，附在搜尋結果中供 LLM 使用。"""
    name: str
    type: str
    description: str


@dataclass
class RelationshipSnippet:
    """單一關係的精簡摘要，供 entity graph context 使用。"""
    source: str
    target: str
    description: str
    weight: float


@dataclass
class EntityGraphContext:
    """
    以實體為中心的圖上下文，由 entity_graph_search() 回傳。

    Attributes
    ----------
    seed_entities       查詢中明確提及、已在圖中找到的實體
    neighbor_entities   1-hop 鄰居實體（按相關性分數降序）
    relationships       種子與鄰居之間的關係
    """
    seed_entities: list[EntitySnippet] = field(default_factory=list)
    neighbor_entities: list[EntitySnippet] = field(default_factory=list)
    relationships: list[RelationshipSnippet] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.seed_entities


@dataclass
class TextUnitResult:
    """Local Search 單筆結果：原始文字塊 + 時間戳 + 關聯實體。"""
    id: str
    text: str
    source_video: str
    start_time: float
    end_time: float
    slide_image: str
    score: float
    related_entities: list[EntitySnippet] = field(default_factory=list)


@dataclass
class CommunityResult:
    """Global Search 單筆結果：社群報告 + 展開後的實體詳情 + 可溯源片段。"""
    id: str
    title: str
    summary: str
    level: int
    score: float
    entity_names: list[str] = field(default_factory=list)
    text_unit_ids: list[str] = field(default_factory=list)
    entities: list[EntitySnippet] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)  # 展開的來源片段
    via_graph: bool = False   # True = 圖遍歷發現（非純向量命中）


# ══════════════════════════════════════════════════════════════════════════════
# 主類別
# ══════════════════════════════════════════════════════════════════════════════

class GraphRAGSearcher:
    """
    組合 VectorSearcher + 本機 parquet 資料，提供 global_search / local_search。

    使用方式
    --------
    searcher = GraphRAGSearcher(
        output_dir  = "qa_Module/graphrag/output",
        lancedb_path= "qa_Module/graphrag/lancedb",
    )
    results = searcher.local_search("Who chairs the ADD working group?", top_k=5)
    reports = searcher.global_search("What are the key IETF discussions?", top_k=3)
    """

    def __init__(
        self,
        output_dir: Path | str,
        lancedb_path: Path | str | None = None,
        settings_path: Path | str | None = None,
    ):
        self._output_dir   = Path(output_dir)
        self._lancedb_path = (
            Path(lancedb_path) if lancedb_path
            else self._output_dir.parent / "lancedb"
        )
        self._settings_path = settings_path

        # 載入：第一次搜尋才初始化
        self._vector_searcher = None
        self._entity_list: list[dict] | None = None   # 原始 entity records
        self._entity_by_name: dict[str, dict] = {}    # name → entity record
        self._tu_by_id: dict[str, dict] | None = None  # text_unit_id → text_unit record
        self._adj_graph: dict[str, list[tuple[str, float]]] | None = None  # entity graph

    # ── 載入輔助 ────────────────────────────────────────────────────────────

    def _get_vector_searcher(self):
        if self._vector_searcher is None:
            from qa_Module.graphrag.vectorizer import VectorSearcher
            self._vector_searcher = VectorSearcher(
                self._lancedb_path, self._settings_path
            )
        return self._vector_searcher

    def _get_entities(self) -> list[dict]:
        if self._entity_list is None:
            from qa_Module.graphrag.storage import load_table
            self._entity_list = load_table("entities", self._output_dir)
            # 建立 name → record 的快速查找（大小寫不敏感）
            self._entity_by_name = {
                e["name"].lower(): e for e in self._entity_list
            }
            logger.debug("載入 %d 個 entity", len(self._entity_list))
        return self._entity_list

    def _get_text_units(self) -> dict[str, dict]:
        """懶載入 text_units，建立 id → record 的快速查找表。"""
        if self._tu_by_id is None:
            from qa_Module.graphrag.storage import load_table
            rows = load_table("text_units", self._output_dir)
            self._tu_by_id = {r["id"]: r for r in rows}
            logger.debug("載入 %d 個 text_unit", len(self._tu_by_id))
        return self._tu_by_id

    def _get_adj_graph(self) -> dict[str, list[tuple[str, float]]]:
        """
        建立實體關係的無向加權鄰接表。

        結構：{ entity_name_lower: [(neighbor_name_lower, weight), ...] }
        weight 來自 relationships.parquet 的 weight 欄位（關係出現次數）。
        """
        if self._adj_graph is None:
            from qa_Module.graphrag.storage import load_table
            rows = load_table("relationships", self._output_dir)
            adj: dict[str, list[tuple[str, float]]] = {}
            for r in rows:
                src = r.get("source", "")
                tgt = r.get("target", "")
                if not src or not tgt:
                    continue
                src, tgt = src.lower(), tgt.lower()
                w = float(r.get("weight", 1.0))
                adj.setdefault(src, []).append((tgt, w))
                adj.setdefault(tgt, []).append((src, w))
            self._adj_graph = adj
            logger.debug("建立關係圖：%d 個節點", len(adj))
        return self._adj_graph

    # ── 內部工具 ──────────────────────────────────────────────────────────────

    def _graph_expand(
        self, seed_names: list[str], hops: int = 1
    ) -> dict[str, float]:
        """
        從種子實體出發做加權 BFS，回傳每個可達實體的相關性分數。

        Parameters
        ----------
        seed_names  種子實體名稱列表（來自初始向量命中社群的 entity_names）
        hops        BFS 跳數，預設 1（只擴展一層鄰居）

        Returns
        -------
        { entity_name_lower: relevance_score }
        種子實體本身分數為 1.0；每跳衰減 0.5，乘以關係 weight(正規化後)。
        hop0(種子)= 1.0
        hop1 鄰居 = 1.0 x (rel_weight/max_weight) x 1.0
        hop2 鄰居 = ... x 0.5(若 hops=2)
        """
        adj = self._get_adj_graph()
        if not adj:
            return {}

        # 正規化 weight：取所有邊的最大值
        max_w = max(w for neighbors in adj.values() for _, w in neighbors) or 1.0

        visited: dict[str, float] = {}
        frontier: dict[str, float] = {
            name.lower(): 1.0 for name in seed_names if name
        }

        for hop in range(hops):
            decay = 0.5 ** hop
            next_frontier: dict[str, float] = {}
            for node, node_score in frontier.items():
                for neighbor, raw_w in adj.get(node, []):
                    if neighbor in visited or neighbor in frontier:
                        continue
                    score = node_score * (raw_w / max_w) * decay
                    if score > next_frontier.get(neighbor, 0.0):
                        next_frontier[neighbor] = score
            visited.update(frontier)
            frontier = next_frontier

        visited.update(frontier)
        return visited

    def _resolve_source_refs(self, text_unit_ids: list[str]) -> list[SourceRef]:
        """將 text_unit_ids 展開為 SourceRef 列表（含影片/時間戳）。"""
        tu_map = self._get_text_units()
        refs: list[SourceRef] = []
        for tu_id in text_unit_ids:
            rec = tu_map.get(tu_id)
            if rec:
                refs.append(SourceRef(
                    text_unit_id = tu_id,
                    source_video = rec.get("source_video", ""),
                    start_time   = float(rec.get("start_time", 0.0)),
                    end_time     = float(rec.get("end_time", 0.0)),
                    slide_image  = rec.get("slide_image", ""),
                    text_snippet = rec.get("text", ""),
                ))
        return refs

    def _expand_entity_snippets(self, entity_names: list[str]) -> list[EntitySnippet]:
        """
        將 entity name 列表展開為 EntitySnippet（查找 parquet 資料）。

        community.entity_names 可能使用 'canonical_name|TYPE' 格式
        （如 'add wg|WORKING_GROUP'），需先取 '|' 前的名稱部分再查找。
        """
        self._get_entities()
        snippets: list[EntitySnippet] = []
        for name_key in entity_names:
            # 去除 |TYPE 後綴（如 "add wg|WORKING_GROUP" → "add wg"）
            lookup = name_key.split("|")[0].strip().lower()
            rec = self._entity_by_name.get(lookup)
            if rec:
                snippets.append(EntitySnippet(
                    name=rec["name"],
                    type=rec.get("type", ""),
                    description=rec.get("description", ""),
                ))
        return snippets

    def _get_relationships_rows(self) -> list[dict]:
        """懶載入 relationships 原始列表（每行含 source/target/description/weight）。"""
        if not hasattr(self, "_rel_rows") or self._rel_rows is None:
            from qa_Module.graphrag.storage import load_table
            self._rel_rows: list[dict] = load_table("relationships", self._output_dir)
            logger.debug("載入 %d 條 relationships", len(self._rel_rows))
        return self._rel_rows

    def _related_entities_for_tu(self, tu_id: str) -> list[EntitySnippet]:
        """找出與特定 TextUnit 關聯的實體（反向查找 text_unit_ids）。"""
        entities = self._get_entities()
        snippets: list[EntitySnippet] = []
        for e in entities:
            tu_ids = e.get("text_unit_ids", [])
            # text_unit_ids 可能被序列化為 JSON 字串
            if isinstance(tu_ids, str):
                try:
                    tu_ids = json.loads(tu_ids)
                except Exception:
                    tu_ids = []
            if tu_id in tu_ids:
                snippets.append(EntitySnippet(
                    name=e["name"],
                    type=e.get("type", ""),
                    description=e.get("description", ""),
                ))
        return snippets

    # ── 公開 API ──────────────────────────────────────────────────────────────

    def global_search(
        self,
        query: str,
        top_k: int = 5,
        graph_weight: float = 0.3,
        graph_hops: int = 1,
        graph_discover_threshold: float = 0.05,
    ) -> list[CommunityResult]:
        """
        Global Search：向量搜尋 + 圖遍歷擴展，適合概念性問題。

        流程
        ----
        1. search_communities(top_k*2) → 初始候選（向量語意路徑）
        2. 提取候選社群的 entity_names 作為種子
        3. BFS 遍歷 relationships 圖（graph_hops 跳），取得鄰近實體相關性分數
        4. 掃描所有社群，計算圖分數 = avg(entity_relevance for entity in community)
        5. final_score = (1-graph_weight)*vector_score + graph_weight*graph_score
        6. 納入圖遍歷新發現的社群（vector_score=0，graph_score≥threshold）
        7. 依 final_score 排序，回傳 top_k

        Parameters
        ----------
        graph_weight              圖分數的權重（0=純向量，1=純圖）
        graph_hops                BFS 跳數
        graph_discover_threshold  圖發現新社群的最低圖分數門檻
        """
        vs = self._get_vector_searcher()
        # 加寬初始候選池，確保種子實體夠多
        hits = vs.search_communities(query, top_k=top_k * 2)
        if not hits:
            logger.info("global_search：LanceDB 無結果")
            return []

        from qa_Module.graphrag.storage import load_table
        all_communities = load_table("communities", self._output_dir)
        comm_by_id = {c["id"]: c for c in all_communities}

        # ── Step 1: 向量分數映射 ──────────────────────────────────────────────
        score_map: dict[str, float] = {h["id"]: h.get("score", 0.0) for h in hits}
        hit_data_map: dict[str, dict] = {h["id"]: h for h in hits}

        # ── Step 2: 圖遍歷擴展 ───────────────────────────────────────────────
        # entity_names 是 'canonical_name|TYPE' 格式，需取名稱部分與鄰接表匹配
        seed_entities: list[str] = []
        for hit in hits:
            full = comm_by_id.get(hit["id"], {})
            for key in _parse_list_field(full.get("entity_names", [])):
                seed_entities.append(key.split("|")[0].strip())

        entity_relevance = self._graph_expand(seed_entities, hops=graph_hops)
        logger.debug(
            "graph_expand：種子 %d 個，擴展後鄰近實體 %d 個",
            len(seed_entities), len(entity_relevance),
        )

        # ── Step 3: 對所有社群計算圖分數 ─────────────────────────────────────
        def _graph_score(comm_data: dict) -> float:
            names = _parse_list_field(comm_data.get("entity_names", []))
            if not names:
                return 0.0
            # 同樣去除 |TYPE 後綴後查找相關性分數
            scores = [
                entity_relevance.get(n.split("|")[0].strip().lower(), 0.0)
                for n in names
            ]
            return sum(scores) / len(scores)

        # ── Step 4: 合併向量命中 + 圖遍歷新發現的社群 ────────────────────────
        # candidates: id → (vector_score, graph_score)
        candidates: dict[str, tuple[float, float]] = {}

        for hit in hits:
            cid = hit["id"]
            candidates[cid] = (score_map[cid], _graph_score(comm_by_id.get(cid, {})))

        for comm in all_communities:
            cid = comm["id"]
            if cid in candidates:
                continue
            g = _graph_score(comm)
            
            if g >= graph_discover_threshold:
                candidates[cid] = (0.0, g)
                logger.debug("圖遍歷發現新社群：%s  g_score=%.4f", cid, g)

        # ── Step 5: 最終排序 ──────────────────────────────────────────────────
        def _final_score(v: float, g: float) -> float:
            return (1.0 - graph_weight) * v + graph_weight * g

        ranked = sorted(
            candidates.items(),
            key=lambda x: _final_score(x[1][0], x[1][1]),
            reverse=True,
        )[:top_k]

        # ── Step 6: 組裝 CommunityResult ─────────────────────────────────────
        results: list[CommunityResult] = []
        for cid, (v_score, g_score) in ranked:
            full     = comm_by_id.get(cid, {})
            hit_data = hit_data_map.get(cid, {})
      
            entity_names  = _parse_list_field(full.get("entity_names", []))
            text_unit_ids = _parse_list_field(full.get("text_unit_ids", []))
        
            results.append(CommunityResult(
                id            = cid,
                title         = hit_data.get("title") or full.get("title", ""),
                summary       = hit_data.get("summary") or full.get("summary", ""),
                level         = int(hit_data.get("level") or full.get("level", 0)),
                score         = _final_score(v_score, g_score),
                entity_names  = entity_names,
                text_unit_ids = text_unit_ids,
                entities      = self._expand_entity_snippets(entity_names),
                source_refs   = self._resolve_source_refs(text_unit_ids),
                via_graph     = (v_score == 0.0),
            ))

        logger.info(
            "global_search('%s')：向量命中 %d，圖發現 %d，回傳 %d 筆",
            query[:50],
            len(score_map),
            sum(1 for cid, _ in ranked if cid not in score_map),
            len(results),
        )
        return results

    def local_search(self, query: str, top_k: int = 5) -> list[TextUnitResult]:
        """
        Local Search：語意搜尋文字塊，並展開關聯實體，適合具體事實查詢。

        流程
        ----
        1. VectorSearcher.search_text_units() → 最相關文字塊（含 score）
        2. 對每個 TextUnit，反向查找 entity.text_unit_ids 取得關聯實體
        3. 組裝 TextUnitResult
        """
        vs = self._get_vector_searcher()
        hits = vs.search_text_units(query, top_k=top_k)
        if not hits:
            logger.info("local_search：LanceDB 無結果")
            return []

        results: list[TextUnitResult] = []
        for hit in hits:
            tu_id = hit["id"]
            results.append(TextUnitResult(
                id               = tu_id,
                text             = hit.get("text", ""),
                source_video     = hit.get("source_video", ""),
                start_time       = float(hit.get("start_time", 0.0)),
                end_time         = float(hit.get("end_time", 0.0)),
                slide_image      = hit.get("slide_image", ""),
                score            = hit.get("score", 0.0),
                related_entities = self._related_entities_for_tu(tu_id),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("local_search('%s')：%d 筆文字塊結果", query[:50], len(results))
        return results

    def entity_graph_search(
        self,
        entity_names: list[str],
        hops: int = 1,
        max_neighbors: int = 10,
    ) -> EntityGraphContext:
        """
        以 entity_names 為種子做圖遍歷，回傳相關實體與關係。

        流程
        ----
        1. 在實體表中模糊比對 entity_names（大小寫不敏感）
        2. BFS 展開 hops 跳，取得鄰居實體與相關性分數
        3. 收集種子與鄰居之間的所有 Relationship 記錄
        4. 回傳 EntityGraphContext

        Parameters
        ----------
        entity_names    query_result.entities（LLM 從查詢中抽取的實體名稱列表）
        hops            BFS 跳數（預設 1）
        max_neighbors   最多回傳幾個鄰居實體
        """
        if not entity_names:
            return EntityGraphContext()

        self._get_entities()  # 確保 _entity_by_name 已載入

        # ── Step 1: 找出種子實體（支援部分名稱匹配） ──────────────────────────
        seed_keys: set[str] = set()
        for name in entity_names:
            name_lower = name.lower().strip()
            if name_lower in self._entity_by_name:
                seed_keys.add(name_lower)
            else:
                # 部分名稱匹配：entity 名稱包含查詢名稱，或查詢名稱包含 entity 名稱
                for key in self._entity_by_name:
                    if name_lower in key or key in name_lower:
                        seed_keys.add(key)
                        break

        if not seed_keys:
            logger.info("entity_graph_search：查無符合的種子實體 %s", entity_names)
            return EntityGraphContext()

        seed_entities = [
            EntitySnippet(
                name=self._entity_by_name[k]["name"],
                type=self._entity_by_name[k].get("type", ""),
                description=self._entity_by_name[k].get("description", ""),
            )
            for k in seed_keys
        ]

        # ── Step 2: BFS 展開取得鄰居及相關性分數 ─────────────────────────────
        entity_relevance = self._graph_expand(list(seed_keys), hops=hops)
        # 去除種子本身，只保留真正的鄰居
        neighbor_relevance = {
            k: v for k, v in entity_relevance.items() if k not in seed_keys
        }
        # 按相關性降序取 max_neighbors 個
        top_neighbor_keys = sorted(
            neighbor_relevance, key=lambda k: neighbor_relevance[k], reverse=True
        )[:max_neighbors]

        neighbor_entities = []
        for k in top_neighbor_keys:
            rec = self._entity_by_name.get(k)
            if rec:
                neighbor_entities.append(EntitySnippet(
                    name=rec["name"],
                    type=rec.get("type", ""),
                    description=rec.get("description", ""),
                ))

        # ── Step 3: 收集種子↔鄰居之間的關係 ─────────────────────────────────
        relevant_keys = seed_keys | set(top_neighbor_keys)
        rel_rows = self._get_relationships_rows()
        relationships: list[RelationshipSnippet] = []
        seen_pairs: set[frozenset] = set()

        for row in rel_rows:
            src = row.get("source", "").lower()
            tgt = row.get("target", "").lower()
            if not src or not tgt:
                continue
            # 關係兩端都需在 relevant_keys 中，且至少一端是種子
            if (src in relevant_keys and tgt in relevant_keys
                    and (src in seed_keys or tgt in seed_keys)):
                pair = frozenset([src, tgt])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                relationships.append(RelationshipSnippet(
                    source=row.get("source", src),
                    target=row.get("target", tgt),
                    description=row.get("description", ""),
                    weight=float(row.get("weight", 1.0)),
                ))

        # 按 weight 降序排列
        relationships.sort(key=lambda r: r.weight, reverse=True)

        logger.info(
            "entity_graph_search：種子 %d 個，鄰居 %d 個，關係 %d 條",
            len(seed_entities), len(neighbor_entities), len(relationships),
        )
        return EntityGraphContext(
            seed_entities=seed_entities,
            neighbor_entities=neighbor_entities,
            relationships=relationships,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 命令列快速測試
# ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     import sys

#     if str(_ROOT) not in sys.path:
#         sys.path.insert(0, str(_ROOT))

#     if sys.platform == "win32":
#         import io
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

#     logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

#     # ── 測試參數（直接修改這裡）────────────────────────────────────────────────
#     OUTPUT_DIR   = _ROOT / "qa_Module/graphrag/output"
#     LANCEDB_PATH = _ROOT / "qa_Module/graphrag/lancedb"
#     SETTINGS     = None

#     LOCAL_QUERY  = "Who is ADD working group chair"
#     # GLOBAL_QUERY = "What are the main working groups discussed"
#     GLOBAL_QUERY = "工作組的整體討論方向是什麼？"  # 測試中文查詢與圖遍歷發現

#     TOP_K        = 3
#     # ─────────────────────────────────────────────────────────────────────────

#     searcher = GraphRAGSearcher(OUTPUT_DIR, LANCEDB_PATH, SETTINGS)

#     # ── Local Search ──────────────────────────────────────────────────────────
    # print(f"\n{'='*60}")
    # print(f"[LOCAL SEARCH] query: {LOCAL_QUERY!r}")
    # print('='*60)
    # local_results = searcher.local_search(LOCAL_QUERY, top_k=TOP_K)
    # if not local_results:
    #     print("  (無結果)")
    # for rank, r in enumerate(local_results, 1):
    #     print(f"\n  [{rank}] score={r.score:.4f}")
    #     print(f"       來源: {r.source_video}  {r.start_time:.1f}s - {r.end_time:.1f}s")
    #     if r.slide_image:
    #         print(f"       投影片: {r.slide_image}")
    #     print(f"       文字: {r.text[:200]}")
    #     if r.related_entities:
    #         print(f"       相關實體({len(r.related_entities)}):", ", ".join(
    #             f"{e.name}({e.type})" for e in r.related_entities[:5]
    #         ))

    # # ── Global Search ─────────────────────────────────────────────────────────
    # print(f"\n{'='*60}")
    # print(f"[GLOBAL SEARCH] query: {GLOBAL_QUERY!r}")
    # print('='*60)
    # global_results = searcher.global_search(GLOBAL_QUERY, top_k=TOP_K)
    # if not global_results:
    #     print("  (無結果)")
    # for rank, r in enumerate(global_results, 1):
    #     tag = "[GRAPH]" if r.via_graph else "[VEC]  "
    #     print(f"\n  [{rank}] {tag} score={r.score:.4f}  level={r.level}")
    #     print(f"       標題: {r.title}")
    #     print(f"       摘要: {r.summary[:250]}")
    #     if r.entities:
    #         print(f"       相關實體({len(r.entities)}):", ", ".join(
    #             f"{e.name}({e.type})" for e in r.entities[:5]
    #         ))
    #     if r.source_refs:
    #         print(f"       來源片段({len(r.source_refs)}):")
    #         for ref in r.source_refs[:3]:
    #             print(f"         - {ref.source_video}  {ref.start_time:.1f}s - {ref.end_time:.1f}s")
    #             if ref.slide_image:
    #                 print(f"           投影片: {ref.slide_image}")
    #             print(f"           預覽: {ref.text_snippet[:100]}")
