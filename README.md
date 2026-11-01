# IFRS17 智能财务分析与运营 Agent 平台 (Python版)

## 📖 项目背景
本项目是一个基于 Python 3.10 + FastAPI + LangGraph 构建的企业级多智能体（Multi-Agent）后端原型。旨在验证大模型能力（意图识别、工具调用、状态机编排、RAG 检索、语义层防幻觉）接入传统财务核算（IFRS17）场景的可行性，具备高并发、数据隔离与敏捷交付特性。并配备了完整的前端可视化工作台。

## 🛠️ 技术栈
- **后端框架**: Python 3.10, FastAPI, Uvicorn
- **AI 框架**: LangChain, LangGraph (官方原生版)
- **大模型服务**: Ollama 本地私有化部署 (Qwen2.5-7B / nomic-embed-text)
- **数据与中间件**: SQLite, ChromaDB, ContextVar
- **前端交互**: 原生 HTML + Tailwind CSS + marked.js
- **部署交付**: Docker, Docker Compose

## 🏗️ 核心架构设计
1. **复合技能路由 (Multi-Skill Router)**: 通过强约束的 Prompt，将用户意图精准分发至 `DIFF`（差异分析）、`KNOWLEDGE`（知识检索）、`METRIC`（指标查询）、`REPORT`（报告生成）四大独立分支。
2. **多源证据并发与 Reducer**: 在差异分析节点中，使用 `asyncio.gather` 并发查库与查规则；利用 LangGraph 原生的 `operator.add` 实现全局 State 的安全合并，杜绝并发覆盖。
3. **语义层与指标防幻觉**: 大模型不直接生成物理表 SQL。通过轻量级 SQLite 语义层字典表与 `@tool` 工具调用，将业务公式、来源表固化，确保口径唯一。
4. **企业级数据隔离**: 利用 Python 的 `ContextVar` 结合 FastAPI 异步特性，在网关层实现轻量级的行级数据权限隔离（`tenant_id`），杜绝越权查询。
5. **检索增强生成 (RAG)**: 使用 Chroma 向量库配合本地 Embedding 模型，将企业规则和 UAT 文档向量化召回，配合引用来源抑制幻觉。
6. **SSE 流式输出**: 基于 FastAPI `StreamingResponse` 配合 LangGraph 原生的流式机制，实现打字机效果。

## 🚀 快速启动
1. 本地安装 Ollama 并拉取模型: `ollama run qwen2.5:7b` 和 `ollama pull nomic-embed-text`
2. 安装依赖: `pip install -r requirements.txt`
3. 启动 FastAPI 应用: `uvicorn main:app --reload` (或使用指定端口 `--port 8001`)
4. 打开浏览器，访问可视化工作台：
   👉 **`http://127.0.0.1:8000/static/index.html`**

## 🎬 界面操作指南 (Demo)

进入工作台后，您可以直接在底部输入框提问，或点击左侧的**“技能快捷指令”**体验四大核心场景：

1. **🔴 差异排查 (DIFF)**
   - **表现**：系统并发查询账务明细与核算规则，收集完整证据链后，交由大模型流式输出差异归因报告。
2. **🟢 指标查询 (METRIC)**
   - **表现**：系统路由至 SQLite 语义层，调用 `@tool` 获取口径公式与默认过滤条件，避免模型胡编乱造。
3. **🟡 知识问答 (KNOWLEDGE)**
   - **表现**：通过 Chroma 向量检索从本地知识库召回 UAT 报错原因与解决方案，并结合出处生成回答。
4. **🟣 报告生成 (REPORT)**
   - **表现**：强约束 Prompt 生成五大财务模块的长文本报告，前端界面将 Markdown 渲染为精美的表格与报表。

## 🐳 一键部署 (Docker)
本项目已实现全链路容器化，支持在容器内访问宿主机的 Ollama 大模型：

```bash
docker-compose up -d --build
```
*(注意：请确保宿主机已开启 Ollama 服务)*