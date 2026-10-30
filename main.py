from fastapi import FastAPI
from fastapi.responses import StreamingResponse # 👈 引入流式响应
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import TypedDict, Annotated, List
import operator
import uvicorn
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. 初始化本地 RAG 知识库 (保持不变)
# ==========================================
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(embedding_function=embeddings, collection_name="ifrs17_rules")
if vectorstore._collection.count() == 0:
    docs = [
        Document(page_content="准备金余额的计算公式是：SUM(closing_balance)，默认过滤条件 reserve_type = 'NET'。", metadata={"source": "准则第4章"}),
        Document(page_content="费用分摊采用三级漏斗模型：先按机构分，再按产品分，最后按险种分。", metadata={"source": "企业核算规则"})
    ]
    vectorstore.add_documents(docs)
    print(">>> [RAG 初始化] 知识库文档已向量化并写入 Chroma！")

# ==========================================
# 2. 定义 State 与 Tool (保持不变)
# ==========================================
class AgentState(TypedDict):
    question: str
    intent: str
    evidence_list: Annotated[List[str], operator.add]
    answer: str

@tool
def query_gl_database(org_name: str) -> str:
    """查询底层总账数据库，获取指定机构的准备金金额明细。"""
    print(f">>> [真实工具调用] 正在连接数据库查询: {org_name}...")
    if "上海" in org_name:
        return "数据库结果：上海机构准备金旧账套 1000 万，新账套 1250 万。"
    return "数据库结果：未找到该机构数据。"

# ==========================================
# 3. 定义节点 (保持不变)
# ==========================================
def router_node(state: AgentState):
    print(">>> [Router 节点] 正在识别意图...")
    llm = ChatOllama(model="qwen2.5:7b")
    prompt = "你是一个财务意图识别助手。请分析用户输入，严格返回以下之一：\n1. DIFF：排查差异。\n2. METRIC：查询口径。\n只输出单词。"
    response = llm.invoke([("system", prompt), ("user", state["question"])])
    intent = response.content.strip()
    if "DIFF" not in intent and "METRIC" not in intent:
        intent = "METRIC"
    return {"intent": intent}

async def diff_analysis_node(state: AgentState):
    print(">>> [差异分析节点] 准备调用工具查库...")
    llm = ChatOllama(model="qwen2.5:7b")
    llm_with_tools = llm.bind_tools([query_gl_database])
    response = llm_with_tools.invoke(state["question"])
    
    evidence = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_result = query_gl_database.invoke(tool_call["args"])
            evidence.append(tool_result)
    else:
        evidence.append("模型未触发工具调用。")
    return {"evidence_list": evidence}

def query_metric_node(state: AgentState):
    print(">>> [语义层节点] 正在通过 RAG 检索...")
    results = vectorstore.similarity_search(state["question"], k=1)
    metric_info = f"【RAG检索】: {results[0].page_content}" if results else "未找到相关规则。"
    return {"evidence_list": [metric_info]}

async def analyze_node(state: AgentState):
    print(f">>> [汇总分析] 收集到 {len(state['evidence_list'])} 条证据...")
    llm = ChatOllama(model="qwen2.5:7b")
    context = "\n".join(state["evidence_list"])
    prompt = f"你是一个财务专家。根据证据回答问题：\n【证据】\n{context}\n\n【问题】\n{state['question']}"
    
    print(">>> 正在等待大模型吐字...")
    # 注意这里是 await llm.ainvoke
    response = await llm.ainvoke(prompt)
    print(">>> 大模型吐字完成！")
    return {"answer": response.content}

# ==========================================
# 4. 构建图 (保持不变)
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("diff_analysis", diff_analysis_node)
workflow.add_node("query_metric", query_metric_node)
workflow.add_node("analyze", analyze_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", lambda x: x["intent"], {"DIFF": "diff_analysis", "METRIC": "query_metric"})
workflow.add_edge("diff_analysis", "analyze")
workflow.add_edge("query_metric", "analyze")
workflow.add_edge("analyze", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# ==========================================
# 5. FastAPI 接口 (改造为流式输出)
# ==========================================
app = FastAPI(title="IFRS17 Agent 企业级 Python 版")

# 👇 新增：解决前端本地文件跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 核心改造：定义一个异步生成器
async def stream_generator(question: str, session_id: str):
    print(f"\n========== [流式] 收到请求 [{session_id}]: {question} ==========")
    config = {"configurable": {"thread_id": session_id}}
    init_state = {"question": question, "intent": "", "evidence_list": [], "answer": ""}
    
    yield f"data: {json.dumps({'status': 'Agent 正在思考与查库...'}, ensure_ascii=False)}\n\n"
    
    try:
        # 👇 核心修改：使用 stream_mode="messages"，直接捕获图中的 LLM Token 流
        async for msg, metadata in app_graph.astream(init_state, config=config, stream_mode="messages"):
            # 只捕获 "analyze" 节点（最终生成节点）的大模型输出
            if metadata.get("langgraph_node") == "analyze":
                if hasattr(msg, "content") and msg.content:
                    yield f"data: {json.dumps({'text': msg.content}, ensure_ascii=False)}\n\n"
                    
    except Exception as e:
        print(f"❌ 报错: {e}")
        yield f"data: {json.dumps({'text': f'发生错误: {str(e)}'}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"

@app.get("/api/agent/chat/stream")
async def chat_stream(question: str, session_id: str = "user_001"):
    # 返回 StreamingResponse，设置媒体类型为 text/event-stream
    return StreamingResponse(stream_generator(question, session_id), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)