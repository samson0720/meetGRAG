# meetGRAG 評估框架（Evaluation Framework）

本框架用於量化評估 meetGRAG GraphRAG 系統的問答品質，支援 LLM-as-judge 與規則型兩種評估模式，可獨立執行（不依賴 FastAPI server）。

---

## 目錄結構

```
eval/
├── config.py                        # EvalConfig：所有執行參數
├── pipeline.py                      # EvalPipeline：直接呼叫 qa_Module 各階段
├── runner.py                        # EvalRunner：批次執行 + 聚合報告
├── reporter.py                      # 輸出 JSON / CSV + terminal 摘要
├── run_eval.py                      # CLI 入口點
│
├── metrics/
│   ├── base.py                      # BaseMetric 抽象基類 + MetricResult
│   ├── faithfulness.py              # 答案忠實度
│   ├── answer_relevance.py          # 答案相關性
│   ├── context_precision.py         # 檢索精確率
│   ├── context_recall.py            # 檢索召回率
│   └── citation_accuracy.py         # 引用正確性（[REF:xxx]）
│
├── testset/
│   ├── builder.py                   # 測試集建立工具（手動 + 半自動）
│   ├── validator.py                 # 測試集格式驗證
│   └── samples/
│       └── sample_testset.json      # 範例測試集（6 題）
│
└── results/                         # 評估結果輸出（每次執行獨立資料夾）
```

---

## 評估指標說明

所有指標分數範圍均為 **0.0 ~ 1.0**，分數越高代表品質越好。

| 指標 | 說明 | LLM judge | 規則備援 |
|------|------|:---------:|:-------:|
| **Faithfulness** | 答案中的事實是否都有 context 支撐，不捏造 | 原子主張逐一驗證 | TF-IDF token 重疊率 |
| **Answer Relevance** | 答案是否切題地回應了問題 | 0-10 直接評分 | core_concepts 命中率 |
| **Context Precision** | 被檢索到的 chunks 中相關比例 | per-chunk YES/NO 判斷 | cited/retrieved 比例 |
| **Context Recall** | 正確答案所需的資訊是否都被找到 | 主張覆蓋率驗證 | entities 命中率 |
| **Citation Accuracy** | `[REF:xxx]` 引用的正確性（語法 + 語意 + 來源） | 語意層 YES/NO 判斷 | 語法解析 + 影片比對 |

### Citation Accuracy 詳細計算

```
最終分數 = 0.4 × 語法準確率 + 0.4 × 語意準確率 + 0.2 × 來源匹配率

語法準確率：[REF:xxx] 中的 xxx 能否對應到真實 chunk_id
語意準確率：引用的 chunk 是否真的支撐對應的句子（LLM judge）
來源匹配率：被引用影片是否與 relevant_source_videos 相符
```

---

## 測試集格式

測試集為 JSON 檔案，結構如下：

```json
{
  "metadata": {
    "version": "1.0",
    "created_at": "2026-04-06T00:00:00+00:00",
    "source_videos": ["IETF118_ADD_session.mp4"],
    "description": "測試集描述"
  },
  "test_cases": [
    {
      "id": "case_001",
      "query": "Who chairs the IETF ADD working group?",
      "query_type": "local",
      "expected_answer": "David Lawrence and Glenn Deen.",
      "expected_chunks": [],
      "relevant_source_videos": ["IETF118_ADD_session.mp4"],
      "ground_truth_entities": ["David Lawrence", "Glenn Deen", "ADD"],
      "tags": ["person", "easy"],
      "difficulty": "easy",
      "notes": "選填備註"
    }
  ]
}
```

### 欄位說明

| 欄位 | 必填 | 說明 |
|------|:----:|------|
| `id` | ✓ | 唯一識別碼 |
| `query` | ✓ | 問題字串 |
| `query_type` | ✓ | `"local"` / `"global"` / `"auto"` |
| `expected_answer` | ✓ | 參考答案（用於 Faithfulness / Recall 計算） |
| `expected_chunks` | | 期望被檢索到的 chunk_id 列表（填寫後 Precision/Recall 更準確） |
| `relevant_source_videos` | | 正確答案所在影片（用於 Citation Accuracy 來源比對） |
| `ground_truth_entities` | | 答案中的關鍵實體（用於規則型 Recall） |
| `tags` | | 分類標籤（可用於 `--filter-tags` 過濾） |
| `difficulty` | | `"easy"` / `"medium"` / `"hard"` |

---

## 快速開始

> 請在虛擬環境中，從專案根目錄（`meetGRAG/`）執行以下指令。

### 1. 驗證測試集格式

```bash
python eval/run_eval.py --validate-testset eval/testset/samples/sample_testset.json
```

### 2. 規則型評估（快速，不消耗額外 LLM tokens）

```bash
python eval/run_eval.py \
    --testset eval/testset/samples/sample_testset.json \
    --judge-mode rule
```

### 3. LLM judge 完整評估

```bash
python eval/run_eval.py \
    --testset eval/testset/samples/sample_testset.json \
    --judge-mode llm
```

### 4. 只評估特定指標

```bash
python eval/run_eval.py \
    --testset eval/testset/samples/sample_testset.json \
    --judge-mode llm \
    --metrics faithfulness citation_accuracy
```

### 5. 過濾特定難度或標籤

```bash
# 只跑 easy 與 medium 難度
python eval/run_eval.py \
    --testset eval/testset/samples/sample_testset.json \
    --judge-mode rule \
    --filter-difficulty easy medium

# 只跑含 "person" tag 的案例
python eval/run_eval.py \
    --testset eval/testset/samples/sample_testset.json \
    --judge-mode rule \
    --filter-tags person
```

### 6. 指定輸出目錄與格式

```bash
python eval/run_eval.py \
    --testset eval/testset/samples/sample_testset.json \
    --judge-mode llm \
    --output-dir eval/results/ \
    --report-format both
```

---

## 測試集建立

### 方式一：半自動生成（從 text_units 抽樣）

```bash
python eval/run_eval.py \
    --build-testset \
    --n-samples 20 \
    --output eval/testset/my_testset.json
```

執行後會逐題互動式確認（輸入 `Y` 加入、`n` 跳過、`e` 編輯）。

若不需互動審核（直接全部加入）：

```bash
python eval/run_eval.py \
    --build-testset \
    --n-samples 20 \
    --no-interactive \
    --output eval/testset/my_testset.json
```

### 方式二：Python API 手動新增

```python
from eval.testset.builder import TestSetBuilder
from pathlib import Path

builder = TestSetBuilder(
    output_graphrag_dir=Path("qa_Module/graphrag/output"),
    description="我的測試集",
)

builder.add_case(
    query="Who chairs the IETF ADD working group?",
    expected_answer="David Lawrence and Glenn Deen.",
    query_type="local",
    ground_truth_entities=["David Lawrence", "Glenn Deen"],
    relevant_source_videos=["IETF118_ADD_session.mp4"],
    tags=["person", "easy"],
    difficulty="easy",
)

builder.save(Path("eval/testset/my_testset.json"))
```

---

## 輸出結果

每次評估會在 `eval/results/{run_id}/` 產生以下檔案：

```
eval/results/
└── eval_20260406_143022/
    ├── eval_20260406_143022_report.json    # 完整報告
    ├── eval_20260406_143022_summary.csv    # 摘要（每行一題）
    └── intermediate/                       # 每題的中間結果（JSON）
        ├── case_001.json
        └── case_002.json
```

### report.json 結構

```json
{
  "run_id": "eval_20260406_143022",
  "summary": {
    "total_cases": 6,
    "successful_cases": 6,
    "overall_average": 0.742,
    "metric_averages": {
      "faithfulness": 0.85,
      "answer_relevance": 0.78,
      "context_precision": 0.71,
      "context_recall": 0.63,
      "citation_accuracy": 0.89
    },
    "metric_pass_rates": {
      "faithfulness": 0.833
    },
    "by_query_type": {
      "local":  { "count": 3, "avg_score": 0.79 },
      "global": { "count": 3, "avg_score": 0.68 }
    }
  },
  "cases": [
    {
      "case_id": "sample_001",
      "query": "...",
      "answer": "...",
      "overall_score": 0.91,
      "metric_scores": { "faithfulness": 1.0, "citation_accuracy": 0.9 },
      "metric_reasons": { "faithfulness": "All 2 claims verified." },
      "cited_chunk_ids": ["chunk_abc123"]
    }
  ]
}
```

### summary.csv 欄位

| 欄位 | 說明 |
|------|------|
| `case_id` | 測試案例 ID |
| `query` | 問題（前 100 字） |
| `query_type_actual` | 系統實際判斷的查詢類型 |
| `overall_score` | 各指標未加權平均 |
| `faithfulness` | 各指標分數 |
| `latency_ms` | Pipeline 執行時間（毫秒） |
| `is_fallback` | 是否觸發 fallback 回應 |

---

## 完整 CLI 參數說明

```
usage: run_eval.py [-h] [--build-testset | --validate-testset PATH]
                   [--testset PATH]
                   [--metrics {faithfulness,answer_relevance,...} [...]]
                   [--judge-mode {llm,rule,hybrid}]
                   [--judge-provider PROVIDER] [--judge-model MODEL]
                   [--top-k INT] [--score-cutoff FLOAT] [--max-chunks INT]
                   [--filter-tags TAG [...]] [--filter-difficulty {easy,medium,hard} [...]]
                   [--filter-ids ID [...]]
                   [--output-dir PATH] [--report-format {json,csv,both}]
                   [--no-save-intermediate]
                   [--n-samples INT] [--output PATH] [--no-interactive]
                   [--verbose]

評估執行：
  --testset PATH              測試集 JSON 路徑（必填）
  --metrics [...]             要計算的指標，預設全部
  --judge-mode                llm（預設）| rule | hybrid
  --judge-provider            judge LLM provider（預設 groq）
  --judge-model               judge LLM 模型（預設 llama-3.3-70b-versatile）
  --top-k INT                 檢索結果數（預設 5）
  --score-cutoff FLOAT        相關性門檻（預設 -0.5）
  --max-chunks INT            最大 chunk 數（預設 5）

過濾：
  --filter-tags [...]         只評估含這些 tag 的案例
  --filter-difficulty [...]   只評估特定難度
  --filter-ids [...]          只評估指定 case_id

輸出：
  --output-dir PATH           結果儲存目錄（預設 eval/results/）
  --report-format             json | csv | both（預設 both）
  --no-save-intermediate      不儲存每題中間狀態

測試集建立：
  --build-testset             啟用半自動生成模式
  --n-samples INT             抽取樣本數（預設 10）
  --output PATH               輸出路徑
  --no-interactive            不進行互動式審核

其他：
  --validate-testset PATH     只驗證格式，不執行評估
  --verbose                   啟用詳細日誌
```

---

## 架構說明

### 資料流

```
測試集 JSON
    │
    ▼
EvalRunner.run()
    │
    ├─ 載入測試集 → 套用 filter
    ├─ EvalPipeline.from_config()  ← qa_Module LLM + GraphRAGSearcher
    ├─ 初始化 judge LLM（若 judge_mode != "rule"）
    │
    └─ 對每個 test_case：
         │
         ▼
         EvalPipeline.run(query)
           │  process_query → retrieve → organize → generate
           ▼
         PipelineResult（保留各階段中間物件）
           │
           ▼
         各 BaseMetric.compute(pipeline_result, test_case, judge_llm)
           ▼
         MetricResult(score, passed, reason, details)
           │
           ▼
         CaseResult（彙整所有指標結果）
    │
    ▼
EvalReport（聚合統計）
    │
    ├─ EvalReporter.to_json()
    ├─ EvalReporter.to_csv()
    └─ EvalReporter.print_summary()
```

### 關鍵設計決策

**Judge LLM 與系統 LLM 分離**
評估器使用獨立的 judge LLM（預設 groq/llama-3.3），避免被評估系統與評估者使用同一模型導致的自我強化偏差。

**PipelineResult 保留所有中間物件**
各指標可直接存取 `OrganizedContext.chunks[i].entities`、`Citation.source_refs` 等原始物件，不需重新計算。

**expected_chunks 為選填**
Context Recall 的 ground truth（chunk_id 列表）填寫成本高，可改用 LLM 分解 `expected_answer` 為原子主張來替代。

**judge_mode 的差異**

| 模式 | 說明 | 適用情境 |
|------|------|----------|
| `rule` | 純規則型，不呼叫額外 LLM | 快速驗證、CI 流程 |
| `llm` | LLM-as-judge，準確度高 | 正式評估、論文實驗 |
| `hybrid` | 優先用 LLM，失敗時降級為規則 | 平衡成本與準確度 |

---

## 擴充自定義指標

1. 在 `eval/metrics/` 建立新的 Python 檔案，繼承 `BaseMetric`：

```python
from eval.metrics.base import BaseMetric, MetricResult

class MyMetric(BaseMetric):
    name = "my_metric"
    threshold = 0.6

    def compute(self, pipeline_result, test_case, judge_llm=None) -> MetricResult:
        score = ...  # 計算邏輯
        return self._make_result(score, reason="...")
```

2. 在 `eval/metrics/__init__.py` 的 `METRIC_REGISTRY` 中登錄：

```python
from eval.metrics.my_metric import MyMetric

METRIC_REGISTRY["my_metric"] = MyMetric
```

3. 使用時加入 `--metrics my_metric` 參數即可。
