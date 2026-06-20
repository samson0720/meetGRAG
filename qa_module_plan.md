# QA Module 實作計畫

## 前提條件

- `database/` 資料夾中已有資料處理模組產出的 JSON 檔案
- 每筆 JSON 記錄格式假設如下：
  ```json
  {
    "id": "rec_001",
    "video_name": "W3C_CSS_2023.mp4",
    "start_time": 605.0,
    "end_time": 635.0,
    "slide_image": "slides/W3C_CSS_2023_42.png",
    "ocr_text": "CSS Subgrid — Working Group Discussion",
    "chart_description": "架構圖說明 Grid 與 Subgrid 的巢狀關係",
    "transcript": "So the subgrid proposal has been formally accepted..."
  }
  ```
- LLM：Mistral 7B（透過 Ollama 本地部署，`http://localhost:11434`）

---

## 目錄結構

```
qa_Module/
├── graphrag/
│   ├── document_loader.py     # JSON → GraphRAG 輸入 TXT（含 [SOURCE] 標注）
│   ├── indexer.py             # 主索引管線（客製化 GraphRAG + 可溯源注入）
│   ├── searcher.py            # global_search() / local_search() 介面
│   └── settings.yaml          # GraphRAG 設定檔
├── query_processor.py         # 查詢語意解析 + 問題類型分類
├── retriever.py               # 根據問題類型路由至對應搜尋策略
├── organizer.py               # 過濾冗餘，保留核心事實
└── generator.py               # 生成最終回覆（含可溯源 sources[]）
```

---

## Step 1 — `graphrag/document_loader.py`

**職責**：讀取 `database/` 中的 JSON 檔案，轉換為 GraphRAG 索引管線所需的 TXT 輸入格式，並在每份文件開頭嵌入可溯源元資料標注。

**輸入**：`database/*.json`
**輸出**：`graphrag/input/*.txt`

### 核心邏輯

```python
def load_records(database_dir: str) -> list[dict]:
    """讀取 database/ 下所有 JSON 檔案，合併為記錄列表"""

def record_to_txt(record: dict) -> str:
    """
    將單筆記錄轉換為帶 [SOURCE] 標注的 TXT 內容
    格式：
        [SOURCE: {video_name}, START: {start_time}, END: {end_time}]
        {ocr_text}
        {chart_description}
        {transcript}
    """

def export_to_input_dir(records: list[dict], output_dir: str):
    """將每筆記錄寫出為獨立的 .txt 檔案至 graphrag/input/"""
```

### 注意事項
- `chart_description` 為空時跳過，避免在文字中產生空行雜訊
- 每筆記錄對應一個 `.txt` 檔案，檔名格式：`{video_name}_{id}.txt`
- 同時輸出一份 `source_map.json`，記錄 `txt 檔名 → {source_video, start_time, end_time}` 的映射，供索引階段注入使用

---

## Step 2 — `graphrag/indexer.py`

**職責**：執行客製化的 Microsoft GraphRAG 索引管線，在 `text_units` DataFrame 注入可溯源欄位，最終將知識圖譜寫入 Neo4j、向量資料寫入 LanceDB。

**輸入**：`graphrag/input/*.txt` + `source_map.json`
**輸出**：Neo4j 知識圖譜 + LanceDB 向量索引

### 索引流程

```
載入 TXT 文件
    → 文字分塊 (TextSplitter, chunk_size=600 tokens)
    → 【客製化】解析每個 text_unit 所屬文件的 [SOURCE] 標注
             注入 source_video / start_time / end_time 欄位
    → LLM 實體擷取 (Entity Extraction, Mistral 7B)
    → LLM 關係擷取 (Relationship Extraction)
    → Leiden 社群偵測 (resolution=1.0)
    → LLM 社群報告生成 (Community Summarization)
    → 寫入 Neo4j（Entity, Relationship, Community, TextUnit 節點）
    → Embedding 向量化 text_units
    → 寫入 LanceDB
```

### 關鍵客製化：可溯源欄位注入

```python
def inject_traceability(text_units_df: pd.DataFrame,
                        source_map: dict) -> pd.DataFrame:
    """
    在 text_units DataFrame 中新增三個欄位：
      - source_video: 來源影片名稱
      - start_time:   片段開始秒數 (float)
      - end_time:     片段結束秒數 (float)
    從 source_map 以 document_id 為 key 查找對應值
    """
```

### settings.yaml 關鍵設定項

```yaml
llm:
  api_base: "http://localhost:11434/v1"   # Ollama
  model: "mistral"
  max_tokens: 4096

embeddings:
  model: "nomic-embed-text"               # Ollama 本地 embedding 模型

chunking:
  size: 600
  overlap: 100

storage:
  neo4j_uri: "bolt://localhost:7687"
  lancedb_path: "./graphrag/lancedb"
```

---

## Step 3 — `graphrag/searcher.py`

**職責**：封裝 Global Search 與 Local Search，提供統一的函式介面供 `retriever.py` 呼叫。

```python
def global_search(query: str, top_communities: int = 5) -> list[CommunityResult]:
    """
    Global Search：從 Neo4j 取得最相關的社群報告
    適用於：概念性、趨勢性、跨文件推論問題

    回傳：
      [{ community_id, title, summary, text_unit_ids[] }]
    """

def local_search(query: str, top_k: int = 5) -> list[TextUnitResult]:
    """
    Local Search：
      1. 以 Embedding 在 LanceDB 向量檢索 top_k text_units
      2. 以 text_unit_ids 在 Neo4j 查詢關聯的 Entity 節點與鄰域
      3. 合併後回傳含可溯源資訊的結果

    回傳：
      [{
        text_unit_id, text, source_video,
        start_time, end_time,
        related_entities: [{ name, type, description }]
      }]
    """
```

---

## Step 4 — `query_processor.py`

**職責**：接收使用者原始查詢字串，透過 LLM 進行語意解析，輸出問題類型（`local` 或 `global`）與核心概念列表。

**判斷邏輯（LLM Prompt 設計要點）**：

| 問題特徵 | 判定為 |
|---|---|
| 詢問特定技術、組織、人物、事件 | `local` |
| 詢問某個具體時間點發生了什麼 | `local` |
| 詢問整體趨勢、演進脈絡、比較 | `global` |
| 詢問「哪些」、「概述」、「歷史」 | `global` |

```python
class QueryResult:
    query: str
    query_type: Literal["local", "global"]
    core_concepts: list[str]   # LLM 抽取的關鍵詞，用於輔助搜尋
    reasoning: str             # LLM 判斷依據（debug 用）

def process_query(query: str) -> QueryResult:
    """
    呼叫 Ollama/Mistral，以 structured output 格式回傳 QueryResult
    System Prompt 指示：
      1. 抽取查詢中的核心技術概念
      2. 判斷問題屬於 local 還是 global
      3. 以 JSON 格式回覆（避免自由文字）
    """
```

---

## Step 5 — `retriever.py`

**職責**：根據 `QueryResult.query_type` 路由至對應的搜尋函式，回傳原始檢索結果。

```python
def retrieve(query_result: QueryResult, top_k: int = 5) -> RetrievalContext:
    """
    - query_type == "global"  → searcher.global_search()
    - query_type == "local"   → searcher.local_search()
    - query_type == "hybrid"  → 兩者均呼叫，合併結果（未來擴充）

    回傳 RetrievalContext：
      {
        query_type: str,
        community_results: list[CommunityResult],   # global 時有值
        text_unit_results: list[TextUnitResult],    # local 時有值
        raw_context_text: str   # 供 organizer 使用的純文字合併結果
      }
    """
```

---

## Step 6 — `organizer.py`

**職責**：將 `RetrievalContext.raw_context_text` 送給 LLM，過濾冗餘資訊，僅保留與查詢最相關的核心事實，同時維護 source 來源映射不丟失。

```python
def organize(query: str, context: RetrievalContext) -> OrganizedContext:
    """
    System Prompt 設計要點：
      1. 給定「查詢問題」與「原始檢索段落列表」
      2. 要求 LLM 以 JSON 格式回覆：
         - relevant_facts: 每條保留的核心事實（附帶原始 source_id）
         - dropped_reason: 哪些段落被丟棄及原因（debug 用）
      3. 嚴禁 LLM 自行補充未在原始段落中出現的資訊

    回傳 OrganizedContext：
      {
        relevant_facts: [{ fact: str, source_ids: [str] }],
        sources: [TextUnitResult]   # 保留事實對應的完整 source 資訊
      }
    """
```

---

## Step 7 — `generator.py`

**職責**：以整理後的 `OrganizedContext` 與使用者原始查詢為輸入，生成最終回覆，並附帶完整的可溯源 `sources[]` 陣列。

```python
def generate(query: str, context: OrganizedContext) -> FinalResponse:
    """
    System Prompt 設計要點：
      1. 依據 relevant_facts 生成回覆，在文中以 [SOURCE_ID: xxx] 標注引用來源
      2. 語言與查詢問題保持一致（中文問 → 中文答）
      3. 若 relevant_facts 為空，回覆「依據現有資料無法確認」，不得推測
      4. 禁止在答案中出現未在 relevant_facts 中的資訊

    回傳 FinalResponse（最終 JSON 格式）：
      {
        "answer": "...[SOURCE_ID: ts_001]...",
        "sources": [
          {
            "source_id": "ts_001",
            "video_name": "W3C_CSS_2023.mp4",
            "start_time": 605.0,
            "end_time": 635.0,
            "transcript_snippet": "...",
            "slide_image_url": "slides/..."
          }
        ],
        "query_type_used": "local",
        "processing_time_ms": 1234
      }
    """
```

---

## 完整資料流

```
使用者輸入查詢
    │
    ▼
query_processor.process_query(query)
    → query_type: "local" | "global"
    → core_concepts: ["CSS Subgrid", "W3C"]
    │
    ▼
retriever.retrieve(query_result)
    → local  : searcher.local_search()  → LanceDB 向量 + Neo4j 圖擴展
    → global : searcher.global_search() → Neo4j 社群報告
    → RetrievalContext（含 raw_context_text）
    │
    ▼
organizer.organize(query, context)
    → LLM 過濾冗餘段落
    → OrganizedContext（relevant_facts + sources）
    │
    ▼
generator.generate(query, organized_context)
    → LLM 生成回覆（強制引用 source ID）
    → FinalResponse（answer + sources[]）
```

---

## 實作順序建議

| 優先級 | 步驟 | 原因 |
|---|---|---|
| 1 | `graphrag/document_loader.py` | 其他所有步驟的資料來源 |
| 2 | `graphrag/settings.yaml` | 索引前需確認 LLM/DB 連線設定 |
| 3 | `graphrag/indexer.py` | 建立圖譜與向量庫，後續才能搜尋 |
| 4 | `graphrag/searcher.py` | 封裝搜尋介面，驗證圖譜建立正確 |
| 5 | `query_processor.py` | 相對獨立，可用 unit test 驗證 |
| 6 | `retriever.py` | 依賴 searcher + query_processor |
| 7 | `organizer.py` | 依賴 retriever 輸出 |
| 8 | `generator.py` | 最後一層，依賴 organizer 輸出 |

---

## 驗收測試案例

| 測試類型 | 輸入 | 期望輸出 |
|---|---|---|
| Local Query | 「2023 年 IETF 117 哪個 RFC 草案被採納？」 | 回覆含具體 RFC 編號，sources[] 有對應時間戳 |
| Global Query | 「W3C 近年對 CSS Grid 的整體立場為何？」 | 回覆含演進趨勢，query_type_used = "global" |
| 無資料問題 | 「量子計算在 IETF 的討論進度？」 | 回覆「依據現有資料無法確認」，sources[] 為空 |
| 可溯源驗證 | 任意查詢 | sources[] 中所有 start_time/end_time 可對應至 database/ 中的原始記錄 |
