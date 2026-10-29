# IFRS17 智能财务分析与运营 Agent 平台 (Python版)

## 📖 项目背景
本项目是一个基于 Python 3.10 + FastAPI + LangGraph 构建的企业级多场景 Agent 原型。旨在验证大模型能力（意图识别、状态机编排、多源证据并发收集、语义层路由）在传统财务核算（IFRS17）场景下的可行性。与 Java 版形成跨语言技术互补，用于个人架构验证与快速敏捷交付演示。

## 🛠️ 技术栈
- **后端框架**: Python 3.10, FastAPI, Uvicorn
- **AI 框架**: LangChain, LangGraph (官方原生版)
- **大模型服务**: Ollama 本地私有化部署 (Qwen2.5-7B / nomic-embed-text)
- **核心特性**: 异步并发 (Asyncio), 状态合并策略 (Reducer)

## 🏗️ 核心架构设计
1. **状态机编排 (LangGraph)**: 使用 `TypedDict` 定义全局状态，将大模型调用封装为 Node 节点，实现强流程控制的图编排。
2. **多源证据并发收集**: 利用 `asyncio.gather` 并发查询底层明细与核算规则，大幅提升复杂财务排查场景的响应速度。
3. **Reducer 状态合并**: 在 State 中针对 `evidence_list` 使用 `operator.add` 作为 Reducer。确保并发节点写入数据时自动追加合并，防止覆盖，保障证据链完整。
4. **语义层意图路由 (Router)**: 通过强约束 Prompt 识别用户意图，将“查明细”与“查指标口径”精准路由。结合语义层工具，彻底隔离大模型与底层物理表。

## 🚀 快速启动

1. **创建并激活虚拟环境**：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Mac/Linux
   ```

2. **安装依赖**：
   ```bash
   pip install fastapi uvicorn langchain-ollama langgraph -i https://mirrors.aliyun.com/pypi/simple/
   ```

3. **本地启动 Ollama 并拉取模型**：
   ```bash
   ollama run qwen2.5:7b
   ```

4. **启动 FastAPI 服务**：
   ```bash
   uvicorn main:app --reload
   ```

5. **访问测试接口**：
   在浏览器或 Postman 中访问：
   `GET http://127.0.0.1:8000/api/agent/chat?question=帮我查一下上海机构准备金为什么对不上`

## 📝 演进规划 (TODO)
- [ ] 接入 Redis Checkpointer，实现分布式工作流状态持久化。
- [ ] 引入 SSE 流式输出（Server-Sent Events），优化前端用户体验。
- [ ] 编写 Dockerfile，实现 Python 服务的容器化部署。
