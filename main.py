from fastapi import FastAPI
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

# ==========================================
# 1. 初始化本地 RAG 知识库 (Chroma)
# ==========================================
# 使用 Ollama 的 embedding 模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(embedding_function=embeddings, collection_name="ifrs17_rules")

# 注入 IFRS17 准则文档
docs = [
    Document(page_content="准备金余额的计算公式是：SUM(closing_balance)，默认过滤条件 reserve_type = 'NET'。", metadata={"source": "准则第4章"}),
    Document(page_content="费用分摊采用三级漏斗模型：先按机构分，再按产品分，最后按险种分。", metadata={"source": "企业核算规则"})
]
# 如果集合为空，则写入文档（防止重复写入）
if vectorstore._collection.count() == 0:
    vectorstore.add_documents(docs)
    print(">>> [RAG 初始化] 知识库文档已向量化并写入 Chroma！")


# ==========================================
# 2. 定义全局 State 与 真实工具 (Tool)
# ==========================================
class AgentState(TypedDict):
    question: str
    intent: str
    evidence_list: Annotated[List[str], operator.add]
    answer: str

# 定义一个真正去查数据库的工具（这里用 Python 函数模拟去连 MySQL）
@tool
def query_gl_database(org_name: str) -> str:
    """查询底层总账数据库，获取指定机构的准备金金额明细。"""
    print(f">>> [真实工具调用] 正在连接数据库查询: {org_name}...")
    # 模拟查库逻辑
    if "上海" in org_name:
        return "数据库结果：上海机构准备金旧账套 1000 万，新账套 1250 万。"
    return "数据库结果：未找到该机构数据。"


# ==========================================
# 3. 定义节点 (Node)
# ==========================================
def router_node(state: AgentState):
    print(">>> [Router 节点] 正在识别意图...")
    llm = ChatOllama(model="qwen2.5:7b")
    prompt = """
    你是一个财务意图识别助手。请分析用户输入，严格返回以下之一：
    1. DIFF：如果用户问的是账套差异、数据对不上、排查原因。
    2. METRIC：如果用户问的是某个指标的计算口径、公式、含义。
    只输出单词，不要标点。
    """
    response = llm.invoke([("system", prompt), ("user", state["question"])])
    intent = response.content.strip()
    if "DIFF" not in intent and "METRIC" not in intent:
        intent = "METRIC"
    print(f">>> [Router 节点] 识别结果: {intent}")
    return {"intent": intent}

async def diff_analysis_node(state: AgentState):
    print(">>> [差异分析节点] 准备调用工具查库...")
    llm = ChatOllama(model="qwen2.5:7b")
    
    # 1. 将工具绑定给大模型
    llm_with_tools = llm.bind_tools([query_gl_database])
    
    # 2. 大模型判断是否需要调用工具
    response = llm_with_tools.invoke(state["question"])
    
    evidence = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            # 3. 执行真实的 Python 函数
            tool_result = query_gl_database.invoke(tool_call["args"])
            evidence.append(tool_result)
    else:
        evidence.append("模型未触发工具调用，直接回答。")
        
    return {"evidence_list": evidence}

def query_metric_node(state: AgentState):
    print(">>> [语义层节点] 正在通过 RAG 检索指标定义...")
    # 1. 去 Chroma 向量库检索相关文档 (Top 1)
    results = vectorstore.similarity_search(state["question"], k=1)
    if results:
        metric_info = f"【RAG检索到规则】: {results[0].page_content} (来源: {results[0].metadata.get('source')})"
    else:
        metric_info = "未在知识库中找到相关指标定义。"
        
    return {"evidence_list": [metric_info]}

def analyze_node(state: AgentState):
    print(f">>> [汇总分析] 收集到 {len(state['evidence_list'])} 条证据，交给大模型总结...")
    llm = ChatOllama(model="qwen2.5:7b")
    context = "\n".join(state["evidence_list"])
    prompt = f"你是一个财务专家。请根据以下收集到的真实证据，回答用户问题：\n\n【证据列表】\n{context}\n\n【用户问题】\n{state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}


# ==========================================
# 4. 构建图并挂载 Checkpointer
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("diff_analysis", diff_analysis_node)
workflow.add_node("query_metric", query_metric_node)
workflow.add_node("analyze", analyze_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    lambda x: x["intent"],
    {"DIFF": "diff_analysis", "METRIC": "query_metric"}
)
workflow.add_edge("diff_analysis", "analyze")
workflow.add_edge("query_metric", "analyze")
workflow.add_edge("analyze", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)


# ==========================================
# 5. FastAPI 接口
# ==========================================
app = FastAPI(title="IFRS17 Agent 企业级 Python 版")

@app.get("/api/agent/chat")
async def chat(question: str, session_id: str = "user_001"):
    print(f"\n========== 收到请求 [{session_id}]: {question} ==========")
    config = {"configurable": {"thread_id": session_id}}
    init_state = {"question": question, "intent": "", "evidence_list": [], "answer": ""}
    result = await app_graph.ainvoke(init_state, config=config)
    return {"answer": result["answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)