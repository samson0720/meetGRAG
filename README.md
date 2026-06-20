# meetGRAG — 基於多模態圖譜推論的可溯源會議知識檢索系統

> **Graph-based Retrieval And Source-traceable Platform for Technical Meeting Knowledge**

meetGRAG 結合多模態資料對齊、GraphRAG 知識圖譜與 Chrome 瀏覽器擴充功能，讓使用者能以自然語言查詢大型國際技術會議（IETF、W3C、IEEE）的知識，並取得可直接跳轉至影片指定時間點的可溯源引用。

---

## 目錄

- [專案動機](#專案動機)
- [核心技術貢獻](#核心技術貢獻)
- [系統架構](#系統架構)
- [目錄結構](#目錄結構)
- [快速開始](#快速開始)
- [Docker 部署](#docker-部署)
- [詳細安裝說明](#詳細安裝說明)
- [設定說明](#設定說明)
- [API 參考](#api-參考)
- [Chrome Extension（GRASP）](#chrome-extensiongrasp)
- [評估框架](#評估框架)
- [技術棧](#技術棧)
- [常見問題](#常見問題)

---

## 專案動機

現有技術會議知識管理面臨三個核心痛點：

1. **資料高度分散**：知識散落於影片錄影、投影片、逐字稿三種異質媒體，缺乏統一整合機制。
2. **向量 RAG 的推論局限**：餘弦相似度檢索無法建立實體間的邏輯關聯，無法支援跨年份演進脈絡或跨文件引用關係等複雜推論。
3. **問答結果缺乏可溯源性**：現有系統的回覆無從驗證，使用者無法確認答案來自哪場演講或哪個時間點。

---

## 核心技術貢獻

| # | 貢獻 | 說明 |
|---|------|------|
| 1 | **多模態時間軸對齊管線** | 融合 SSIM 投影片偵測、Whisper 語音時間戳，將影片、投影片、逐字稿統一整合於同一時間軸 |
| 2 | **注入可溯源性元資料的客製化 GraphRAG** | 修改標準 GraphRAG 索引管線，在 `text_units` 中注入 `start_time`/`end_time`，確保可溯源元資料從索引起即完整保存 |
| 3 | **混合式檢索架構（Hybrid Retrieval）** | 整合 Global Search（社群報告摘要推論）與 Local Search（向量實體精確檢索），自動依問題類型選擇最佳策略 |
| 4 | **Click-to-seek 可溯源介面** | Chrome Extension 側邊欄問答後可直接點擊時間戳，立即跳轉至瀏覽器中播放的原始影片對應片段 |

---

## 系統架構

### 四階段資料流

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1 — 多模態資料擷取（離線批次）                                     │
│                                                                         │
│  會議影片 (.mp4) ──► SSIM 投影片偵測 ──► Phi-3.5-Vision OCR/圖表分析 ──┐ │
│                  └─► Whisper ASR（word-level 時間戳）──────────────────┤ │
│                                                       ↓                 │ │
│                                              時間軸對齊邏輯              │ │
│                                                       ↓                 │ │
│                                        PostgreSQL temporal_alignments   │ │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — 知識圖譜索引建構（離線批次）                                   │
│                                                                         │
│  PostgreSQL 資料                                                         │
│       │                                                                 │
│       ├─► document_loader.py（JSON → TXT，注入 [SOURCE] 標注）           │
│       │                                                                 │
│       └─► indexer.py（客製化 GraphRAG 管線）                             │
│              ├─ 文字分塊（TextUnit，~600 tokens）                        │
│              ├─ LLM 實體/關係擷取（Groq / OpenAI / Ollama）              │
│              ├─ Leiden 社群偵測 + LLM 社群報告生成                       │
│              └─► 輸出至 Parquet + LanceDB 向量資料庫                     │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 3 — 即時查詢管線（線上服務）                                       │
│                                                                         │
│  POST /api/v1/query                                                      │
│       │                                                                 │
│       ├─► query_processor（判斷 local / global，抽取關鍵概念）            │
│       ├─► retriever（路由至 Global / Local Search，支援會議過濾）         │
│       ├─► organizer（分數過濾、Jaccard 去重、整合實體與來源）              │
│       └─► generator（LLM 生成含 [REF:chunk_id] 標注的回覆）              │
│              └─► JSON 回應（answer + citations + cited_entity_names）    │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 4 — Chrome Extension GRASP（前端）                                │
│                                                                         │
│  Side Panel 問答介面 ──► POST /api/v1/query ──► 渲染回覆 + 來源卡片      │
│  content.js 偵測 <video> 播放時間 ──► 自動同步當前投影片內容              │
│  使用者點擊時間戳 ──► SEEK_VIDEO 訊息 ──► video.currentTime 跳轉         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 查詢時序

```
使用者輸入問題
      │
      ▼
Chrome Extension ──POST /api/v1/query──► FastAPI
                                              │
                                    process_query()（local/global 分類）
                                              │
                           ┌──────────────────┴──────────────────┐
                     local search                           global search
                     （向量相似度                          （社群報告
                      + 圖實體擴展）                        摘要推論）
                           └──────────────────┬──────────────────┘
                                              │
                                        organize()（過濾去重）
                                              │
                                        generate()（LLM 生成）
                                              │
                                    回傳 JSON（answer + citations）
                                              │
                                        渲染回覆 + 來源卡片
                                              │
                                    使用者點擊時間戳 → 影片跳轉
```

---

## 目錄結構

```
meetGRAG/
├── app.py                          # FastAPI 後端入口
├── .env.example                    # 環境變數範本
├── qa_Module/                      # 核心問答模組
│   ├── query_processor.py          # 查詢語意解析（local/global 分類）
│   ├── retriever.py                # 混合式檢索路由
│   ├── organizer.py                # 結果整理與去重
│   ├── generator.py                # LLM 答案生成
│   ├── run_query.py                # CLI 查詢測試工具
│   ├── llm/                        # LLM 客戶端抽象層
│   │   ├── base.py                 # BaseLLMClient 介面
│   │   ├── factory.py              # create_llm() / create_llm_from_config()
│   │   ├── groq_client.py          # Groq Cloud（多 Key 輪替）
│   │   ├── openai_client.py        # OpenAI API
│   │   └── ollama_client.py        # Ollama 本地推論
│   └── graphrag/                   # GraphRAG 知識圖譜模組
│       ├── indexer.py              # 索引管線（實體/關係/社群）
│       ├── searcher.py             # Global / Local Search 介面
│       ├── document_loader.py      # JSON → 帶 [SOURCE] 標注的 TXT
│       ├── vectorizer.py           # LanceDB 向量嵌入與檢索
│       ├── storage.py              # Parquet 讀寫工具
│       ├── settings.yaml           # 索引管線設定（LLM、Embedding、分塊）
│       ├── input/                  # 索引輸入（TXT + source_map.json）
│       ├── output/                 # 索引輸出（*.parquet + index_meta.json）
│       └── lancedb/                # LanceDB 向量資料庫檔案
├── extension/                      # Chrome Extension GRASP
│   ├── manifest.json               # Manifest V3 設定
│   ├── background.js               # Service Worker
│   ├── content.js                  # Content Script（影片時間同步）
│   ├── sidepanel.html              # 側邊欄 UI
│   └── sidepanel.js                # 側邊欄邏輯（API 呼叫、跳轉）
├── database/                       # 原始會議資料
│   └── meet_origin_data/
│       ├── *.json                  # 會議逐字稿與投影片資料
│       └── link.json               # 會議名稱 → YouTube URL 映射
├── eval/                           # 自訂評估框架
│   ├── run_eval.py                 # CLI 入口
│   ├── metrics/                    # 評估指標（Faithfulness、Relevance 等）
│   └── testset/                    # 測試集管理與範例
├── ragas_eval/                     # Ragas 評估集成
│   ├── run_ragas_eval.py
│   └── requirements.txt
└── sys_arch.md                     # 系統架構設計文件
```

---

## 快速開始

### 前置條件

- Python 3.10+
- Google Chrome（安裝 Extension 用）
- Groq API Key（免費申請：[console.groq.com](https://console.groq.com)）
- OpenAI API Key（Embedding 用）
- 已建立索引資料（`qa_Module/graphrag/output/*.parquet` 存在）

### 三步驟啟動

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 設定 API 金鑰
cp .env.example .env
# 編輯 .env，填入 OPENAPI_API_KEY 與 GROQ_API_KEY

# 3. 啟動後端服務
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

服務啟動後訪問 `http://localhost:8000/docs` 查看互動式 API 文件。

---

## Docker 部署

專案提供兩個 Docker 服務：

| 服務 | 用途 |
|------|------|
| `api` | 常駐 FastAPI 查詢服務，對外提供 `http://localhost:9000/api/v1` |
| `jobs` | 手動執行 GraphRAG 索引、多模態 YouTube 分析與評估任務 |

### 1. 準備環境變數

```bash
cp .env.example .env
# 編輯 .env，填入 OPENAPI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY
```

### 2. 建置映像檔

`jobs` 映像檔會安裝 PyTorch、Whisper、Transformers、Ragas 等重型套件。建置前請確認 Docker Desktop 使用的磁碟至少有 15-20 GB 可用空間；Windows 預設通常會用 C 槽的 `AppData\Local\Docker\wsl\disk\docker_data.vhdx`。

```bash
docker compose build api
docker compose build jobs
```

### 3. 啟動 API

```bash
docker compose up api
```

啟動後可開啟：

- API health check：`http://localhost:9000/api/v1/health`
- OpenAPI 文件：`http://localhost:9000/docs`

### 4. 執行批次任務

```bash
# 建立或更新 GraphRAG 索引
docker compose run --rm jobs python -m qa_Module.graphrag.run_index

# 分析 YouTube 影片並輸出 meetGRAG JSON
docker compose run --rm jobs python -m qa_Module.multimodal.run_youtube_analysis \
  "https://www.youtube.com/watch?v=xxxxx" \
  --meeting-name "IETF 125_ IAB Open" \
  --auto-index

# 執行自訂評估
docker compose run --rm jobs python -m eval.run_eval

# 執行 Ragas 評估
docker compose run --rm jobs python ragas_eval/run_ragas_eval.py
```

Docker 會掛載並共用以下資料目錄：

- `database/`
- `qa_Module/graphrag/input/`
- `qa_Module/graphrag/output/`
- `qa_Module/graphrag/lancedb/`

Chrome Extension 目前預設呼叫 `http://localhost:9000/api/v1`，因此 Docker 模式不需要修改 `extension/sidepanel.js`。

> 注意：`api` 映像檔刻意不安裝 `torch`、`transformers`、Whisper 等重型多模態套件；YouTube 分析請使用 `jobs` container 執行。`jobs` 預設使用 CPU-only PyTorch wheel，若要使用 GPU，需另外改成 CUDA base image 與 NVIDIA Container Toolkit。

---

## 詳細安裝說明

### Step 1：建立 Python 環境

```bash
# 建立虛擬環境
python -m venv .venv

# 啟動虛擬環境
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 安裝主要依賴
pip install -r requirements.txt
```

### Step 2：設定環境變數

```bash
cp .env.example .env
```

編輯 `.env`（詳見[設定說明](#設定說明)）。

### Step 3：準備資料庫與索引

**3a. 準備原始會議資料**

將會議 JSON 檔案放入 `database/meet_origin_data/`，並在 `link.json` 登錄 YouTube URL：

```json
[
  { "meet": "IETF 125_ IAB Open", "link": "https://www.youtube.com/watch?v=xxxxx" }
]
```

**3b. 執行索引管線**

啟動後端後，透過 API 觸發索引（約需數分鐘至數十分鐘，依資料量而定）：

```bash
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "run_document_loader": true,
    "run_indexer": true,
    "run_vectorizer": true
  }'
```

查詢進度：

```bash
curl http://localhost:8000/api/v1/index/status
```

### Step 4：啟動後端

```bash
# 預設 port 8000（對外服務）
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 指定 port 9000（僅本機）
uvicorn app:app --host 127.0.0.1 --port 9000 --reload
```

### Step 5：安裝 Chrome Extension

1. 開啟 Chrome，前往 `chrome://extensions/`
2. 啟用右上角**「開發者模式」**
3. 點擊**「載入未封裝的擴充功能」**，選取本專案的 `extension/` 資料夾
4. 在 `extension/sidepanel.js` 第一行確認 `API_BASE` 設定正確：
   ```javascript
   const API_BASE = "http://localhost:9000/api/v1";  // Docker 預設位址
   ```

### Step 6：驗證安裝

```bash
# CLI 快速測試
python qa_Module/run_query.py "What is QUIC?"

# 或透過 curl 測試 API
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is QUIC?", "top_k": 3, "max_chunks": 3}'
```

---

## 設定說明

### `.env`（API 金鑰）

| 變數 | 用途 | 取得方式 |
|------|------|----------|
| `OPENAPI_API_KEY` | OpenAI Embedding API（`text-embedding-3-small`） | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `OPENAI_API_KEY` | 標準名稱，ragas 等套件讀取 | 同上 |
| `GROQ_API_KEY` | Groq 主要 API Key（LLM 推論） | [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_API_KEY_2` ~ `GROQ_API_KEY_10` | Groq 備用 Key（自動輪替，避免超限） | 同上，建立多個帳號 |

> 系統自動蒐集所有非空的 `GROQ_API_KEY*` 並輪替使用，可依需求填入 2~10 組。

### `qa_Module/graphrag/settings.yaml`（索引管線）

```yaml
llm:
  provider: groq              # openai | groq | ollama
  model: groq/compound-mini  # 實體擷取與社群報告使用的模型

embeddings:
  provider: openai            # openai | ollama
  model: text-embedding-3-small

chunking:
  size: 600                   # 每個 TextUnit 的最大 token 數
  overlap: 100                # 相鄰 chunk 的重疊 token 數

storage:
  lancedb_path: ./qa_Module/graphrag/lancedb

community:
  resolution: 1.0             # Leiden resolution（越高 → 社群越小）
```

#### 切換 LLM Provider

**使用 OpenAI**：
```yaml
llm:
  provider: openai
  model: gpt-4o-mini
```

**使用本地 Ollama**（完全離線）：
```bash
# 先安裝並啟動 Ollama
ollama serve
ollama pull mistral
```
```yaml
llm:
  provider: ollama
  model: mistral
  base_url: http://localhost:11434/v1
```

---

## API 參考

### `POST /api/v1/query`

主要問答端點。

**Request Body**：

```json
{
  "query": "Who chairs the IETF ADD working group?",
  "top_k": 5,
  "score_cutoff": -0.5,
  "max_chunks": 5,
  "dedup_threshold": 0.6,
  "temperature": 0.2,
  "max_tokens": 8192,
  "current_meeting": "IETF 125_ IAB Open"
}
```

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `query` | `string` | 必填 | 使用者查詢問題 |
| `top_k` | `int` | `5` | 向量檢索結果數量 |
| `score_cutoff` | `float` | `-0.5` | 相關性分數門檻，低於此值的結果將被過濾 |
| `max_chunks` | `int` | `5` | 傳入 LLM 的最大片段數 |
| `dedup_threshold` | `float` | `0.6` | Jaccard 去重門檻（0~1，越小越嚴格） |
| `temperature` | `float` | `0.2` | LLM 生成溫度 |
| `max_tokens` | `int` | `8192` | LLM 最大生成 token 數 |
| `current_meeting` | `string\|null` | `null` | 限定查詢的會議名稱，`null` 表示不限定 |

**Response（200 OK）**：

```json
{
  "answer": "The IETF ADD working group is chaired by...[REF:chunk_001]...",
  "query_type": "local",
  "is_fallback": false,
  "citations": [
    {
      "chunk_id": "chunk_001",
      "chunk_type": "text_unit",
      "title": "IETF 125_ IAB Open.mp4  605.0s–635.0s",
      "source_refs": [
        {
          "source_video": "IETF 125_ IAB Open.mp4",
          "start_time": 605.0,
          "end_time": 635.0,
          "video_url": "https://www.youtube.com/watch?v=xxxxx",
          "text_snippet": "The ADD working group..."
        }
      ],
      "entity_names": ["IETF ADD", "Working Group Chair"]
    }
  ],
  "usage": { "prompt_tokens": 1234, "completion_tokens": 256 },
  "cited_entity_names": ["IETF ADD"]
}
```

---

### `GET /api/v1/health`

健康檢查。

```json
{ "status": "ok", "model": "llama-3.3-70b-versatile" }
```

---

### `GET /api/v1/meetings`

列出所有已登錄的會議及 YouTube URL。

```json
[
  { "name": "IETF 125_ IAB Open", "url": "https://www.youtube.com/watch?v=xxxxx" }
]
```

---

### `GET /api/v1/graph`

回傳完整知識圖譜資料（供 D3.js 可視化）。

```json
{
  "nodes": [
    {
      "id": "QUIC",
      "type": "PROTOCOL",
      "description": "Quick UDP Internet Connections protocol",
      "community_id": "comm_001",
      "meetings": ["IETF 125_ IAB Open"]
    }
  ],
  "links": [
    {
      "source": "QUIC",
      "target": "HTTP/3",
      "weight": 0.9,
      "description": "QUIC is the transport layer for HTTP/3"
    }
  ],
  "communities": [
    { "id": "comm_001", "title": "Network Protocols", "level": 1 }
  ]
}
```

---

### `GET /api/v1/slides`

回傳會議投影片段（含 OCR 文字、逐字稿、圖表分析）。

**Query Parameters**：`video_name`（可選，過濾指定影片）

```json
{
  "slides": [
    {
      "slide_index": 1,
      "video_name": "IETF 125_ IAB Open.mp4",
      "slide_image": "",
      "time_range": {
        "start_sec": 120.0,
        "end_sec": 180.0,
        "display_timestamp": "00:02:00"
      },
      "multimodal_content": {
        "visual_info": {
          "title": "QUIC Protocol Overview",
          "content": ["Key features", "Implementation details"]
        },
        "audio_transcript": "So QUIC is a new transport protocol..."
      }
    }
  ],
  "total": 100
}
```

---

### `POST /api/v1/index`

觸發離線索引管線（背景執行，立即回傳）。

**Request Body**：

```json
{
  "run_document_loader": true,
  "run_indexer": true,
  "run_vectorizer": true,
  "force_vector": false
}
```

| 參數 | 說明 |
|------|------|
| `run_document_loader` | 重新將 `database/` 的 JSON 轉為 TXT 輸入 |
| `run_indexer` | 重新執行 GraphRAG 實體/關係/社群索引 |
| `run_vectorizer` | 重新建立 LanceDB 向量索引 |
| `force_vector` | 強制重新嵌入（即使向量索引已存在） |

---

### `GET /api/v1/index/status`

查詢索引管線進度。

```json
{
  "status": "done",
  "stage": null,
  "started_at": "2026-06-16T10:00:00+00:00",
  "finished_at": "2026-06-16T10:45:00+00:00",
  "stats": {
    "text_units": 1234,
    "entities": 567,
    "relationships": 890,
    "communities": 45
  },
  "error": null
}
```

`status` 可能值：`idle` | `running` | `done` | `failed`

---

## Chrome Extension（GRASP）

**GRASP** — Graph-based Retrieval And Source-traceable Platform

### 主要功能

| 功能 | 說明 |
|------|------|
| **側邊欄問答** | 以自然語言向知識圖譜提問，獲得可溯源回覆 |
| **投影片同步** | 自動偵測 YouTube 影片播放時間，同步顯示當前投影片內容 |
| **Click-to-seek** | 點擊回覆中的時間戳引用，影片立即跳轉至對應片段 |
| **會議過濾** | 自動偵測當前觀看的會議，限定查詢範圍 |

### 安裝步驟

1. 前往 `chrome://extensions/`，啟用**開發者模式**
2. 點擊**「載入未封裝的擴充功能」**，選取 `extension/` 資料夾
3. 設定後端位址（`extension/sidepanel.js` 第一行）：
   ```javascript
   const API_BASE = "http://localhost:9000/api/v1";
   ```

### 使用方式

1. 在 YouTube 開啟 IETF 會議錄影
2. 點擊 Chrome 工具列中的 GRASP 圖示，開啟側邊欄
3. 在輸入框輸入問題，點擊**「Analyze」**
4. 查看回覆與來源卡片，點擊時間戳即可跳轉至影片對應位置

### 架構（Manifest V3）

```
extension/
├── manifest.json      # 權限宣告（sidePanel, activeTab, scripting）
├── background.js      # Service Worker — 開啟側邊欄
├── content.js         # Content Script — 注入影片頁面，處理時間同步與跳轉
├── sidepanel.html     # 側邊欄 UI（深板岩 + 靛青色系設計）
└── sidepanel.js       # 側邊欄邏輯（API 呼叫、投影片同步、Click-to-seek）
```

---

## 評估框架

### 自訂評估框架（`eval/`）

```bash
# 規則型評估（快速，不消耗額外 LLM tokens）
python eval/run_eval.py \
  --testset eval/testset/samples/sample_testset.json \
  --judge-mode rule

# LLM judge 完整評估
python eval/run_eval.py \
  --testset eval/testset/samples/sample_testset.json \
  --judge-mode llm

# 建立測試集
python eval/run_eval.py \
  --build-testset --n-samples 20 \
  --output eval/testset/my_testset.json
```

**評估指標**（分數範圍 0.0–1.0）：

| 指標 | 說明 | 目標值 |
|------|------|--------|
| Faithfulness | 答案忠實度（無幻覺，原子主張逐一驗證） | > 0.85 |
| Answer Relevance | 答案與查詢相關性 | > 0.80 |
| Context Precision | 檢索上下文中相關片段的精確率 | > 0.75 |
| Context Recall | 關鍵資訊被成功召回的比例 | > 0.80 |
| Citation Accuracy | 引用正確性（0.4×語法 + 0.4×語意 + 0.2×來源匹配） | > 0.85 |

### Ragas 評估集成（`ragas_eval/`）

```bash
# 安裝 Ragas 依賴
pip install -r ragas_eval/requirements.txt

# 執行評估
python ragas_eval/run_ragas_eval.py \
  --testset ragas_eval/testset.json \
  --provider groq
```

**測試集格式**：

```json
{
  "metadata": { "version": "1.0", "source_videos": ["IETF125_IABOpen.mp4"] },
  "test_cases": [
    {
      "id": "case_001",
      "query": "Who chairs the IETF ADD working group?",
      "query_type": "local",
      "expected_answer": "David Lawrence and Glenn Deen.",
      "relevant_source_videos": ["IETF118_ADD.mp4"],
      "ground_truth_entities": ["David Lawrence", "Glenn Deen"]
    }
  ]
}
```

---

## 技術棧

| 類別 | 技術 | 用途 |
|------|------|------|
| **後端框架** | FastAPI + Uvicorn | REST API 服務 |
| **知識圖譜** | 客製化 GraphRAG + Leiden Algorithm | 實體/關係/社群索引 |
| **向量資料庫** | LanceDB | 語意向量索引與 Top-K 檢索 |
| **儲存格式** | Apache Parquet | 知識圖譜節點與邊的列式儲存 |
| **LLM（推論）** | Groq Cloud / OpenAI / Ollama | 實體擷取、社群報告、問答生成 |
| **Embedding** | OpenAI `text-embedding-3-small` | 文字向量化 |
| **多模態前處理** | Phi-3.5-Vision + Whisper | OCR/圖表分析 + 語音辨識 |
| **圖演算法** | NetworkX + python-louvain | 圖建構與社群偵測 |
| **Chrome Extension** | Manifest V3 Side Panel API | 瀏覽器前端問答介面 |
| **評估** | Ragas | GraphRAG 管線品質量化評估 |
| **部署** | Docker Compose | 分離常駐 API 與批次 jobs container |

---

## 常見問題

**Q：索引建置需要多久？**

視資料量而定。10 場會議（約 100 個文字片段）通常需要 15–60 分鐘（Groq 免費層）。可透過 `GET /api/v1/index/status` 監控進度。

---

**Q：支援哪些 LLM？OpenAI 是必須的嗎？**

- 索引與問答的 LLM 部分支援 Groq、OpenAI、Ollama，可在 `settings.yaml` 自由切換。
- Embedding 部分目前需要 OpenAI（`text-embedding-3-small`），若要完全離線可改用 Ollama 的 `nomic-embed-text`，需修改 `settings.yaml` 中的 `embeddings` 段落。

---

**Q：多組 Groq API Key 如何運作？**

系統自動蒐集 `.env` 中所有非空的 `GROQ_API_KEY`、`GROQ_API_KEY_2` ~ `GROQ_API_KEY_10`，並在一個 Key 超限時自動輪替至下一組，避免索引過程中斷。

---

**Q：Click-to-seek 為何沒有反應？**

1. 確認已安裝 Extension 且 `API_BASE` 設定正確
2. 確認目前頁面的影片為 `<video>` 元素（不支援嵌入式 iframe 播放器）
3. 檢查 `chrome://extensions/` 中 GRASP 的 Content Script 是否已注入（頁面重新整理後生效）

---

**Q：如何新增一場新的會議？**

```bash
# 1. 將會議資料 JSON 放入 database/meet_origin_data/
cp new_conference.json database/meet_origin_data/

# 2. 在 link.json 登錄 YouTube URL
# [{"meet": "New Conference 2026", "link": "https://youtube.com/..."}]

# 3. 重新觸發索引（舊資料會合併更新）
curl -X POST http://localhost:8000/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{"run_document_loader": true, "run_indexer": true, "run_vectorizer": true}'
```

---

**Q：如何排查索引管線失敗？**

```bash
# 查看錯誤訊息
curl http://localhost:8000/api/v1/index/status
# 若 "status": "failed"，"error" 欄位會說明原因

# 常見原因：
# - API Key 無效或超限 → 更換 .env 中的 API Key
# - 記憶體不足 → 減少 chunking.size 或分批索引
# - 輸入 JSON 格式不符 → 檢查 database/meet_origin_data/ 中的 JSON 格式
```

---

## 授權

本專案為 NCU 114 學年度資工系專題研究，學術使用。
