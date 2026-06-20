# vectorizer.py 流程說明

## 概覽

`vectorizer.py` 負責 GraphRAG 管線中的向量化階段（Step 8）。
它從本機的 `output/` 目錄讀取已索引的資料，將文字轉換為嵌入向量，並寫入 LanceDB 向量資料庫，供後續查詢模組（`searcher.py`）進行語意搜尋。

```
output/*.parquet  →  EmbeddingClient  →  LanceDB
   （索引資料）        （文字轉向量）       （向量儲存）
```

---

## 模組結構

```
vectorizer.py
│
├── 工具函式
│   ├── _load_settings()          讀取 settings.yaml
│   └── _resolve_lancedb_path()   解析 LanceDB 儲存路徑
│
├── 嵌入模型客戶端
│   ├── EmbeddingClient           基類（定義 embed / embed_batch 介面）
│   ├── OpenAIEmbeddingClient     呼叫 OpenAI Embedding API
│   ├── OllamaEmbeddingClient     呼叫本機 Ollama HTTP API
│   └── create_embedding_client() 根據 settings 建立對應客戶端
│
├── LanceDB 索引建立
│   └── _build_lancedb()          核心嵌入 + 寫入邏輯
│
└── 公開 API
    ├── build_index()             建立向量索引（外部呼叫入口）
    └── VectorSearcher            向量搜尋封裝類別
        ├── search_text_units()
        ├── search_entities()
        └── search_communities()
```

---

## 一、設定讀取

### `_load_settings(settings_path, ref_dir)`

讀取 `settings.yaml`，提供嵌入模型的設定來源。搜尋優先順序：

1. 呼叫端明確傳入的 `settings_path`
2. `ref_dir/../settings.yaml`（通常是 `output/../settings.yaml`）
3. `qa_Module/graphrag/settings.yaml`（專案預設位置）

與向量化相關的設定欄位：

```yaml
embeddings:
  provider: openai                  # openai | ollama
  model: text-embedding-3-small     # 嵌入模型名稱
  base_url: http://localhost:11434  # ollama 專用

storage:
  lancedb_path: ./qa_Module/graphrag/lancedb  # 向量索引儲存位置
```

### `_resolve_lancedb_path(settings, output_dir)`

從 `settings["storage"]["lancedb_path"]` 取得路徑。
若為相對路徑，以專案根目錄（`_ROOT`）為基準解析；若未設定則預設為 `output_dir/../lancedb`。

---

## 二、嵌入模型客戶端

### 類別繼承關係

```
EmbeddingClient（基類）
├── OpenAIEmbeddingClient
└── OllamaEmbeddingClient
```

### `EmbeddingClient`（基類）

定義兩個對外介面：

| 方法 | 說明 |
|------|------|
| `embed(text)` | 嵌入單筆文字，回傳 `list[float]` |
| `embed_batch(texts, batch_size=96)` | 分批嵌入多筆文字，每批間隔 0.1 秒以緩衝 rate limit |

`embed_batch` 的分批邏輯：

```
texts = [t1, t2, ..., tN]
         ├── batch 1: t1~t96   → _embed_batch_raw()
         ├── sleep 0.1s
         ├── batch 2: t97~t192 → _embed_batch_raw()
         └── ...
```

子類別只需覆寫 `_embed_batch_raw()`，不需處理分批邏輯。

### `OpenAIEmbeddingClient`

- 初始化時從環境變數讀取 API key（`OPENAPI_API_KEY` 或 `OPENAI_API_KEY`）
- 呼叫 `openai.OpenAI.embeddings.create(model, input=[...])`
- 一次 API 呼叫可送出整批 96 筆，效率最高

### `OllamaEmbeddingClient`

- 呼叫本機 Ollama HTTP API：`POST /api/embeddings`
- 因 Ollama API 不支援 batch，每筆文字各自發送一次 HTTP 請求
- 不需 API key，適合完全本機化部署

### `create_embedding_client(settings)`

根據 `settings["embeddings"]["provider"]` 的值決定建立哪種客戶端：

```
provider == "ollama"  →  OllamaEmbeddingClient
其他（預設）          →  OpenAIEmbeddingClient（並呼叫 load_dotenv()）
```

若 `embeddings.provider` 未設定，則繼承 `llm.provider` 的值。

---

## 三、LanceDB 索引建立

### `_build_lancedb(lancedb_path, text_units, entities, communities, client, force)`

連線 LanceDB 後，依序處理三張資料表。每張表的流程相同：

```
1. 檢查資料表是否已存在 + force 旗標
   ├── 已存在且 force=False → 跳過
   └── 不存在 或 force=True → 執行嵌入

2. 建構嵌入用文字（embed_texts）

3. client.embed_batch(embed_texts) → vectors

4. 將原始欄位 + vector 組合成 rows

5. 若舊資料表存在則先 drop，再 create_table
```

#### 三張資料表的嵌入內容

| 資料表 | 嵌入的文字 | 儲存的欄位 |
|--------|-----------|-----------|
| `text_units` | `r["text"]`（原始文字塊） | id, text, source_video, start_time, end_time, slide_image, entity_ids, vector |
| `entities` | `"{name}: {description}"` | id, name, type, description, embed_text, vector |
| `communities` | `"{title}\n{summary}"` | id, title, summary, level, vector |

#### `entity_ids` 的特殊處理

LanceDB 不支援 `list[str]` 欄位，因此 `entity_ids` 在寫入前以 `json.dumps()` 序列化為 JSON 字串：

```python
"entity_ids": json.dumps(r.get("entity_ids", []), ensure_ascii=False)
# 例：["ent-001", "ent-002"] → '["ent-001", "ent-002"]'
```

#### `communities` 的條件判斷

```python
c_with_summary = [r for r in communities if r.get("summary")]
```

只嵌入有 `summary` 的社群。`summary` 由 `indexer.py` Step 6（LLM 社群報告）填入，若尚未執行 Step 6，此欄位為空字串，直接跳過不報錯。

---

## 四、公開 API：`build_index()`

外部呼叫的主入口，整合上述所有步驟：

```
build_index(output_dir, settings_path, *, force)
│
├── 1. _load_settings()                     讀取 settings.yaml
│
├── 2. storage.load(output_dir)             從 Parquet 讀取索引資料
│       → text_units / entities / communities
│
├── 3. create_embedding_client(settings)    建立嵌入客戶端
│
├── 4. _resolve_lancedb_path(settings)      決定 LanceDB 路徑
│
└── 5. _build_lancedb(...)                  嵌入 + 寫入 LanceDB
         └── 回傳 lancedb_path
```

**參數說明：**

| 參數 | 類型 | 說明 |
|------|------|------|
| `output_dir` | `Path \| str` | `storage.save()` 的輸出目錄，含 `text_units.parquet` 等 |
| `settings_path` | `Path \| str \| None` | settings.yaml 路徑，`None` 時自動搜尋 |
| `force` | `bool` | `True` 時強制重新嵌入，即使索引已存在 |

**回傳值：** LanceDB 資料夾的 `Path`

---

## 五、公開 API：`VectorSearcher`

封裝搜尋邏輯的類別，供 `retriever.py` 等查詢模組呼叫。

### 初始化流程

```python
searcher = VectorSearcher(lancedb_path, settings_path)
```

1. 讀取 `settings.yaml`
2. 建立 `EmbeddingClient`（用於嵌入查詢字串）
3. LanceDB 連線採用 lazy 初始化（首次搜尋時才 `lancedb.connect()`）

### 搜尋流程（`_search()`）

所有三個搜尋方法共用同一個內部實作：

```
query（字串）
│
├── client.embed(query)           將查詢字串嵌入為向量
│
├── db.open_table(table_name)     開啟對應 LanceDB 資料表
│
├── table.search(query_vec)
│        .limit(top_k)
│        .to_list()               ANN 向量搜尋，回傳 top-k 結果
│
└── 將 _distance 轉換為 score
    score = round(1.0 - _distance, 6)
    （LanceDB 預設使用 L2 距離；數值越小代表越相似，轉換後 score 越高越相關）
```

### 三個搜尋方法

| 方法 | 查詢的資料表 | 適用情境 |
|------|------------|---------|
| `search_text_units(query, top_k=10)` | `text_units` | 找語意相關的原文片段，含時間戳溯源 |
| `search_entities(query, top_k=10)` | `entities` | 找與查詢相關的知識圖譜實體 |
| `search_communities(query, top_k=5)` | `communities` | Global Search：找最相關的社群摘要 |

### 回傳格式

每筆結果為一個 `dict`，包含資料表原始欄位加上 `score`：

```python
# search_text_units 範例
{
    "id": "tu-001",
    "text": "HTTP/3 uses QUIC as the transport layer...",
    "source_video": "IETF118_ADD_session_add_001",
    "start_time": 120.5,
    "end_time": 185.0,
    "slide_image": "",
    "entity_ids": '["ent-001", "ent-002"]',  # JSON 字串
    "score": 0.923456
}

# search_entities 範例
{
    "id": "ent-001",
    "name": "HTTP/3",
    "type": "PROTOCOL",
    "description": "The third major version of HTTP...",
    "embed_text": "HTTP/3: The third major version of HTTP...",
    "score": 0.891234
}

# search_communities 範例
{
    "id": "comm-0",
    "title": "HTTP/3 and QUIC Transport Layer",
    "summary": "This community focuses on...",
    "level": 0,
    "score": 0.876543
}
```

---

## 六、命令列使用

```bash
# 建立向量索引（從 output/ 讀取，寫入 lancedb/）
python -m qa_Module.graphrag.vectorizer build

# 強制重建（忽略已存在的索引）
python -m qa_Module.graphrag.vectorizer build --force

# 指定自訂路徑
python -m qa_Module.graphrag.vectorizer build \
    --output-dir qa_Module/graphrag/output \
    --settings qa_Module/graphrag/settings.yaml

# 向量搜尋測試
python -m qa_Module.graphrag.vectorizer search "HTTP/3 QUIC handshake" --table text_units --top-k 5
python -m qa_Module.graphrag.vectorizer search "transport protocol" --table entities --top-k 10
python -m qa_Module.graphrag.vectorizer search "key improvements" --table communities --top-k 3
```

---

## 七、與其他模組的關係

```
indexer.py          執行 Step 1~6，產生 output/*.parquet
    ↓
vectorizer.py       執行 Step 8（build_index），產生 lancedb/
    ↓
retriever.py        呼叫 VectorSearcher 進行語意搜尋
    ↓
generator.py        根據搜尋結果生成答案
```

`build_index()` 依賴 `storage.load()` 讀取已索引資料，因此必須在 `indexer.py` 完成 Step 2（文字分塊）至 Step 4（實體合併）後才能執行 text_units 和 entities 的向量化。communities 的向量化則需 Step 6（LLM 社群報告）完成後才有意義。

---

## 八、相依套件

| 套件 | 用途 | 安裝指令 |
|------|------|---------|
| `lancedb` | 向量資料庫 | `pip install lancedb` |
| `openai` | OpenAI Embedding API | `pip install openai` |
| `pyyaml` | 讀取 settings.yaml | `pip install pyyaml` |
| `python-dotenv` | 讀取 .env 中的 API key | `pip install python-dotenv` |
| `urllib.request` | Ollama HTTP 呼叫 | 標準函式庫，無需安裝 |
