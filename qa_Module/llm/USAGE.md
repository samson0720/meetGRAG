# LLM Client 使用教學

提供統一介面，支援 **OpenAI**、**Groq**、**Ollama** 三種後端，可隨時切換而不需修改業務邏輯。

---

## 安裝依賴

```bash
pip install openai groq
```

> Ollama 使用 OpenAI SDK 的相容 endpoint，不需額外套件，但需在本機啟動 Ollama server：
> ```bash
> ollama serve
> ollama pull mistral   # 下載你要使用的模型
> ```

---

## 快速開始

### 方式一：`create_llm()` 直接建立

```python
from qa_Module.llm import create_llm

# Ollama（本地，不需 API key）
llm = create_llm("ollama", model="mistral")

# Groq Cloud
llm = create_llm("groq", model="llama-3.3-70b-versatile", api_key="gsk_...")

# OpenAI
llm = create_llm("openai", model="gpt-4o", api_key="sk-...")
```

### 方式二：`create_llm_from_config()` 從設定字典建立

適合搭配 `settings.yaml` 使用：

```python
from qa_Module.llm import create_llm_from_config

config = {
    "provider": "ollama",
    "model": "mistral",
}
llm = create_llm_from_config(config)
```

---

## 呼叫方式

### `complete()` — 單次問答（最常用）

```python
response = llm.complete("What is the QUIC protocol?")
print(response.content)
```

加入 system prompt：

```python
response = llm.complete(
    prompt="Summarize the key changes in HTTP/3.",
    system="You are an IETF networking expert. Be concise.",
    temperature=0.2,
    max_tokens=512,
)
print(response.content)
```

### `chat()` — 多輪對話

```python
from qa_Module.llm import create_llm, Message

llm = create_llm("ollama", model="mistral")

messages = [
    Message(role="system", content="You are a networking expert."),
    Message(role="user",   content="What is head-of-line blocking?"),
]
response = llm.chat(messages, temperature=0.1, max_tokens=256)
print(response.content)

# 繼續對話：將 assistant 回覆加入 messages
messages.append(Message(role="assistant", content=response.content))
messages.append(Message(role="user", content="How does QUIC solve it?"))
response2 = llm.chat(messages)
print(response2.content)
```

---

## 回傳物件 `LLMResponse`

| 欄位 | 型別 | 說明 |
|------|------|------|
| `content` | `str` | 模型回覆文字 |
| `model` | `str` | 實際使用的模型名稱 |
| `prompt_tokens` | `int` | 輸入 token 數（部分 provider 才有） |
| `completion_tokens` | `int` | 輸出 token 數 |
| `raw` | `dict` | 原始 API 回應（除錯用） |

```python
response = llm.complete("Hello")
print(response.content)          # 回覆文字
print(response.model)            # 模型名稱
print(response.prompt_tokens)    # token 用量
```

---

## 各 Provider 參數說明

### Ollama

```python
llm = create_llm(
    "ollama",
    model="mistral",                          # ollama pull <model>
    base_url="http://localhost:11434/v1",     # 預設值，可省略
)
```

常用模型：`mistral`、`llama3.2`、`phi3.5`、`qwen2.5`

### Groq

```python
llm = create_llm(
    "groq",
    model="llama-3.3-70b-versatile",
    api_key="gsk_...",   # 從 console.groq.com 取得
)
```

常用模型：`llama-3.3-70b-versatile`、`llama-3.1-8b-instant`、`mixtral-8x7b-32768`

### OpenAI

```python
llm = create_llm(
    "openai",
    model="gpt-4o",
    api_key="sk-...",
)
```

#### 使用 OpenAI 相容的本地伺服器（如 LM Studio、vLLM）

```python
llm = create_llm(
    "openai",
    model="local-model-name",
    api_key="any",                          # 本地伺服器通常不驗證 key
    api_base="http://localhost:1234/v1",    # LM Studio 預設 port
)
```

---

## 在專案中的建議用法

在 `settings.yaml` 中集中管理 provider 設定：

```yaml
llm:
  provider: ollama
  model: mistral
  # api_key: ""        # Ollama 不需要
  # base_url: http://localhost:11434/v1
```

在程式中讀取並建立 client：

```python
import yaml
from qa_Module.llm import create_llm_from_config

with open("settings.yaml") as f:
    settings = yaml.safe_load(f)

llm = create_llm_from_config(settings["llm"])
```

---

## 在其他主機跑模型（遠端 API 呼叫）

不論模型跑在哪台機器，只要該機器對外開放 HTTP port，就能透過現有 client 呼叫。

### 情境一：遠端 Ollama 主機

在**遠端主機**上啟動 Ollama 並開放網路存取：

```bash
# 遠端主機（預設只監聽 127.0.0.1，需改為 0.0.0.0 才能被外部存取）
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

在**本機**呼叫：

```python
llm = create_llm(
    "ollama",
    model="mistral",
    base_url="http://192.168.1.100:11434/v1",   # 換成遠端主機 IP
)
```

### 情境二：遠端 OpenAI 相容伺服器（vLLM、LM Studio、Xinference…）

這類框架都提供 OpenAI-compatible API，直接用 `openai` provider 並指向遠端：

```python
# vLLM（常見於 GPU 伺服器）
llm = create_llm(
    "openai",
    model="mistral-7b-instruct",
    api_key="any",                                    # vLLM 預設不驗證
    api_base="http://gpu-server.local:8000/v1",
)

# LM Studio（遠端 Mac/Windows）
llm = create_llm(
    "openai",
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    api_key="any",
    api_base="http://192.168.1.50:1234/v1",
)
```

### 情境三：搭配 settings.yaml 管理多個主機

```yaml
# settings.yaml
llm:
  provider: openai
  model: mistral-7b-instruct
  api_key: "any"
  api_base: "http://192.168.1.100:8000/v1"   # 遠端 vLLM
```

```python
llm = create_llm_from_config(settings["llm"])
```

### 遠端主機安全注意事項

| 問題 | 建議做法 |
|------|----------|
| 遠端 port 不應直接對公網開放 | 使用 SSH Tunnel 或 VPN |
| 需要認證 | 在 vLLM/Ollama 前加一層 Nginx + API Key 驗證 |
| 加密傳輸 | 使用 HTTPS（Nginx reverse proxy + TLS 憑證） |

**SSH Tunnel 範例**（最簡單的安全做法）：

```bash
# 在本機執行，將遠端 11434 port 轉發到本地 11434
ssh -L 11434:localhost:11434 user@remote-host

# 之後程式直接連本地
llm = create_llm("ollama", model="mistral", base_url="http://localhost:11434/v1")
```

---

## 新增自訂 Provider

繼承 `BaseLLMClient` 並實作 `chat()` 方法，再透過 `register_provider()` 註冊：

```python
from qa_Module.llm import BaseLLMClient, LLMResponse, Message, register_provider

class MyCustomClient(BaseLLMClient):
    def _init(self, api_key: str, **_):
        # 初始化你的 SDK
        self._api_key = api_key

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        # 呼叫你的 API
        result = my_sdk.call(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return LLMResponse(
            content=result.text,
            model=self.model,
        )

# 註冊並使用
register_provider("my_service", MyCustomClient)
llm = create_llm("my_service", model="my-model", api_key="...")
response = llm.complete("Hello!")
```

---

## 檔案結構

```
qa_Module/llm/
├── __init__.py          # 公開 API（import 入口）
├── base.py              # BaseLLMClient、Message、LLMResponse
├── openai_client.py     # OpenAI / Azure / 相容 endpoint
├── groq_client.py       # Groq Cloud
├── ollama_client.py     # Ollama 本地推論
├── factory.py           # create_llm()、create_llm_from_config()
└── USAGE.md             # 本文件
```
