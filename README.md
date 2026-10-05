# AI Agent 工程说明

基于大模型的对话 Agent，支持多模型切换、对话记忆、角色设定、工具调用（时间 / 天气 / 城市）、txt 文档问答和相似问题答案复用，通过 Gradio 网页使用。

## 使用的框架与库

| 框架 / 库 | 用途 |
|------|------|
| Gradio（6.x，要求 >=5.0） | 网页界面：聊天框、模型选择、参数滑块、文件上传 |
| LangChain（`langchain-classic`、`langchain-core`） | ReAct Agent（`create_react_agent` + `AgentExecutor`）、`@tool` 工具定义、`ConversationBufferMemory` 对话记忆、`InMemoryVectorStore` 向量库 |
| `langchain-openai` | `ChatOpenAI` 供 ReAct 调用模型，`OpenAIEmbeddings` 生成文本向量 |
| `langchain-community` | `TextLoader` 读取 txt 文档 |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` 文档分块 |
| `openai` SDK | 直接调用 OpenAI 兼容接口（角色识别等），统一错误处理 |
| `requests` | 调用 Open-Meteo 地理编码与天气接口 |
| `python-dotenv` | 从 `.env` 读取 API Key |

模型服务均走 OpenAI 兼容协议：

| 提供方 | 地址 | 默认模型 |
|------|------|------|
| Ollama（本地，界面默认） | `http://127.0.0.1:11434/v1` | `qwen2.5:3b` |
| 阿里云百炼 | `dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` / `qwen-max`，嵌入 `text-embedding-v3` |
| OpenAI | `api.openai.com/v1` | `gpt-4o-mini` / `gpt-4o`，嵌入 `text-embedding-3-small` |

外部数据接口：Open-Meteo 地理编码 `geocoding-api.open-meteo.com`、天气预报 `api.open-meteo.com`（免 Key）。

## 目录结构

```text
ai-learning/
├── app.py              # Gradio 网页入口
├── main.py             # 命令行示例入口
├── agent/              # Agent 核心
│   ├── agent.py        # Agent 主流程
│   ├── tool_agent.py   # ReAct 工具 Agent
│   ├── llm.py          # 模型配置、调用与错误映射
│   ├── lc_llm.py       # LangChain ChatOpenAI 封装
│   ├── persona.py      # 角色设定识别
│   ├── embeddings.py   # 文本嵌入
│   ├── embedding_guard.py  # 嵌入失败降级
│   ├── documents.py    # 文档向量库与检索
│   └── answer_cache.py # 相似问题答案复用
├── tools/              # LangChain 工具
│   ├── time.py         # 当前时间
│   ├── weather.py      # 天气查询
│   ├── city.py         # 城市信息
│   └── geocode.py      # 地理编码与参数清洗
└── data/               # 本地持久化向量（document_store.json、answer_cache.json）
```

## 模块功能

### 入口

- **`app.py`**：Gradio 网页。左侧设置栏有 temperature、max_tokens 和「新对话」；右侧是对话区（固定高度、内部滚动）、推荐问题和输入区。输入区底部一行是「+」（弹出「文件选择」上传 txt）、模型下拉框和发送按钮（输入框有内容时才显示）。切换模型或参数会重建 Agent 并清空对话。
- **`main.py`**：命令行示例，用默认配置向 Agent 提两个问题并打印回答。

### `agent/` Agent 核心

- **`agent.py`**：`Agent` 类，每轮对话按以下顺序处理：
  1. 识别本轮是否在设定角色，有则更新角色并写入提示词；
  2. 已上传文档时检索相关片段作为参考；
  3. 未使用文档时，先查相似问题，命中则直接复用旧答案；
  4. 否则交给 ReAct Agent 回答，写入对话记忆，并把问答存入答案复用库。

  对话记忆用 `ConversationBufferMemory`；输入、输出和记忆内容都会写日志。
- **`tool_agent.py`**：用 `create_react_agent` + `AgentExecutor` 构建 ReAct Agent，可多轮调用工具（最多 5 轮），能容忍模型输出格式错误。
- **`llm.py`**：维护模型提供方（`PROVIDERS`）和界面下拉选项（`MODEL_CHOICES`，Ollama 排第一）；封装流式和非流式调用；把 HTTP 状态码和错误码（额度不足、Key 无效、限流等）转换成中文提示。
- **`lc_llm.py`**：按提供方创建 LangChain `ChatOpenAI` 实例，供 ReAct Agent 使用。
- **`persona.py`**：用 JSON Schema 让模型判断用户是否在设定助手身份（身份、风格、年龄），解析结果并生成角色提示词。
- **`embeddings.py`**：按提供方创建 `OpenAIEmbeddings`，另有余弦相似度工具函数。
- **`embedding_guard.py`**：嵌入接口失败（例如百炼欠费）时关闭文档检索和答案复用，只警告一次，避免反复报错。
- **`documents.py`**：txt 文档问答。按 UTF-8/GBK 读取，分块（500 字，重叠 80），每 10 块一批写入向量库，持久化到 `data/document_store.json`；检索取前 4 条，相似度不足时改用文首片段，总长不超过 6000 字。只有上传成功后才启用检索。
- **`answer_cache.py`**：相似问题答案复用。问题向量与历史问题相似度不低于 0.9 时直接返回旧答案，数据保存在 `data/answer_cache.json`。

### `tools/` 工具

工具均用 LangChain `@tool` 定义，汇总在 `tools/__init__.py` 的 `TOOLS` 中，每次调用都会记录请求与返回日志。

| 工具 | 文件 | 功能 |
|------|------|------|
| `get_current_time` | `time.py` | 返回本地当前时间 |
| `get_weather` | `weather.py` | 查询城市此刻天气（天况、气温、体感、湿度、降水、云量、气压、风）和当天预报（最高 / 最低温、降水概率、紫外线、日出日落） |
| `get_city_info` | `city.py` | 查询城市所属国家、省 / 州、时区、海拔、人口 |

`geocode.py` 是辅助模块：调用 Open-Meteo 地理编码把城市名转成经纬度；`normalize_city` 清洗模型传入的脏参数（如 `city="武汉"`、JSON 字符串），只保留城市名。

### 未被引用的旧文件

`agent/chain.py`、`agent/prompt.py` 是早期实现，当前代码没有引用。

## 配置与运行

1. 复制 `.env.example` 为 `.env`，填写 `DASHSCOPE_API_KEY`、`OPENAI_API_KEY`；只用本地 Ollama 时 Key 留默认值即可。
2. 安装依赖：`pip install -r requirements.txt`（Python >= 3.10）。
3. 使用 Ollama 时先拉取模型：`ollama pull qwen2.5:3b`。
4. 启动网页：`python app.py`；命令行示例：`python main.py`。

注意：文档问答和答案复用使用百炼嵌入模型 `text-embedding-v3`，需要百炼 Key 可用；不可用时这两项会自动关闭，对话本身不受影响。
