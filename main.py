from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles # 👈 新增：用于挂载前端静态页面
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import TypedDict, Annotated, List
from contextvars import ContextVar
import operator
import uvicorn
import asyncio
import json
import sqlite3
import os
import time

# ==========================================
# 0. 环境变量与静态资源挂载
# ==========================================
# 👇 核心：如果 Docker 注入了环境变量，就用宿主机的地址，否则默认本地
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI(title="IFRS17 Agent Python 终局版")

# 👇 挂载前端页面（确保项目根目录下有一个叫 static 的文件夹，里面放着 index.html）
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================================
# 1. 初始化本地 RAG 知识库 (Chroma)
# ==========================================
# 👇 加上 base_url=OLLAMA_BASE_URL
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(embedding_function=embeddings, collection_name="ifrs17_rules")
if vectorstore._collection.count() == 0:
    docs = [
        Document(page_content="UAT报错E1001：准备金计算溢出。原因：底层折现率配置为空。解决建议：请前往系统参数配置页，补充该机构当月的折现率。", metadata={"source": "UAT手册"}),
        Document(page_content="费用分摊采用三级漏斗模型：先按机构分，再按产品分，最后按险种分。", metadata={"source": "企业核算规则"})
    ]
    vectorstore.add_documents(docs)
    print(">>> [RAG 初始化] 知识库文档已向量化并写入 Chroma！")

# ==========================================
# 2. 初始化 SQLite 语义层 (指标字典)
# ==========================================
db_path = "semantic_layer.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS metric_dict 
                  (metric_name TEXT, formula TEXT, source_table TEXT, default_filter TEXT)''')
cursor.execute("SELECT COUNT(*) FROM metric_dict")
if cursor.fetchone()[0] == 0:
    cursor.execute("INSERT INTO metric_dict VALUES ('准备金余额', 'SUM(closing_balance)', 'dws_ifrs17_reserve', 'reserve_type = NET')")
    conn.commit()
    print(">>> [语义层初始化] 指标字典表已建好并灌入数据！")

# ==========================================
# 3. 上下文与数据隔离 (ContextVar)
# ==========================================
current_org_code = ContextVar("current_org_code", default="3101")

# ==========================================
# 4. 定义 State 与 Tools
# ==========================================
class AgentState(TypedDict):
    question: str
    intent: str
    evidence_list: Annotated[List[str], operator.add]
    answer: str

@tool
def query_gl_database(org_name: str) -> str:
    """查询底层总账数据库，获取指定机构的准备金金额明细。"""
    tenant_id = current_org_code.get()
    print(f">>> [真实工具调用] 正在连接数据库查询: {org_name} (当前租户: {tenant_id})...")
    if "上海" in org_name:
        return "数据库结果：上海机构准备金旧账套 1000 万，新账套 1250 万。"
    return "数据库结果：未找到该机构数据。"

@tool
def query_metric_definition(metric_name: str) -> str:
    """查询财务指标的语义定义。当用户询问某个指标的计算口径、公式时调用此工具。"""
    print(f">>> [语义层工具调用] 正在查询指标字典: {metric_name}")
    cursor.execute("SELECT metric_name, formula, source_table, default_filter FROM metric_dict WHERE metric_name LIKE ?", (f"%{metric_name}%",))
    result = cursor.fetchone()
    if result:
        return f"【语义层定义】指标：{result[0]}, 公式：{result[1]}, 来源表：{result[2]}, 默认过滤：{result[3]}"
    return "未在语义层中找到该指标。"

# ==========================================
# 5. 定义节点 (Node)
# ==========================================
def router_node(state: AgentState):
    print(">>> [Router 节点] 正在识别意图...")
    # 👇 加上 base_url=OLLAMA_BASE_URL
    llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)
    prompt = """
    你是一个财务意图识别助手。请分析用户输入，严格返回以下之一：
    1. DIFF：如果用户问的是账套差异、数据对不上、排查原因。
    2. METRIC：如果用户问的是某个指标的计算口径、公式、含义。
    3. KNOWLEDGE：如果用户问的是 UAT 报错原因、系统操作步骤、业务规则解释。
    4. REPORT：如果用户明确要求“生成报告”、“导出分析”、“写一份总结”。
    只输出单词，不要标点。
    """
    response = llm.invoke([("system", prompt), ("user", state["question"])])
    intent = response.content.strip()
    if intent not in ["DIFF", "METRIC", "KNOWLEDGE", "REPORT"]:
        intent = "METRIC"
    print(f">>> [Router 节点] 识别结果: {intent}")
    return {"intent": intent}

async def diff_analysis_node(state: AgentState):
    print(">>> [差异分析节点] 启动多线程并发收集证据...")
    async def fetch_detail():
        await asyncio.sleep(0.5)
        return query_gl_database.invoke("上海")
    async def fetch_rule():
        await asyncio.sleep(0.5)
        return "规则库结果：IFRS17规定，准备金折现率必须使用即期利率曲线。"
    results = await asyncio.gather(fetch_detail(), fetch_rule())
    print(f">>> [差异分析节点] 证据收集完毕，共 {len(results)} 条")
    return {"evidence_list": results}

def metric_query_node(state: AgentState):
    print(">>> [指标查询节点] 准备检索语义层...")
    # 👇 加上 base_url=OLLAMA_BASE_URL
    llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)
    llm_with_tools = llm.bind_tools([query_metric_definition])
    response = llm_with_tools.invoke(state["question"])
    evidence = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            evidence.append(query_metric_definition.invoke(tool_call["args"]))
    else:
        evidence.append("模型未触发指标查询。")
    return {"evidence_list": evidence}

def rag_query_node(state: AgentState):
    print(">>> [RAG 节点] 正在检索知识库...")
    results = vectorstore.similarity_search(state["question"], k=1)
    info = f"【RAG检索】: {results[0].page_content}" if results else "未找到相关规则。"
    return {"evidence_list": [info]}

def report_generation_node(state: AgentState):
    print(">>> [报告生成节点] 启动，正在撰写专业财务分析报告...")
    # 👇 加上 base_url=OLLAMA_BASE_URL
    llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)
    prompt = """
    你是一个顶级的 IFRS17 财务分析师。请根据用户的提问，生成一份严谨、结构化的财务分析报告。
    报告必须包含以下五大固定模块：
    ## 一、概况
    ## 二、明细（以 Markdown 表格输出）
    ## 三、原因分析
    ## 四、影响评估
    ## 五、建议
    请使用专业、客观、严谨的财务话术，禁止口语化。
    """
    response = llm.invoke([("system", prompt), ("user", state["question"])])
    answer = response.content
    
    filename = f"IFRS17_Python分析报告_{int(time.time())}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(answer)
    print(f">>> [报告生成节点] 报告已成功保存为文件: {filename}")
    return {"answer": answer}

async def analyze_node(state: AgentState):
    print(f">>> [汇总分析节点] 收集到 {len(state['evidence_list'])} 条证据，交由大模型总结...")
    # 👇 核心：加上 streaming=True
    llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, streaming=True)
    context = "\n".join(state["evidence_list"])
    prompt = f"你是一个财务专家。请根据以下收集到的证据，回答用户问题：\n\n【证据列表】\n{context}\n\n【用户问题】\n{state['question']}"
    
    print(">>> 正在等待大模型生成（流式）...")
    response = await llm.ainvoke(prompt)
    print(">>> 大模型生成完毕！") # 👈 加上这句，看看是不是真的卡在大模型
    return {"answer": response.content}

# ==========================================
# 6. 构建状态图
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("diff_analysis", diff_analysis_node)
workflow.add_node("metric_query", metric_query_node)
workflow.add_node("rag_query", rag_query_node)
workflow.add_node("report_generation", report_generation_node)
workflow.add_node("analyze", analyze_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    lambda x: x["intent"],
    {
        "DIFF": "diff_analysis",
        "METRIC": "metric_query",
        "KNOWLEDGE": "rag_query",
        "REPORT": "report_generation"
    }
)
workflow.add_edge("diff_analysis", END)
workflow.add_edge("metric_query", END)
workflow.add_edge("rag_query", END)
workflow.add_edge("report_generation", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# ==========================================
# 7. FastAPI 流式接口
# ==========================================
async def stream_generator(question: str, session_id: str):
    print(f"\n========== [流式] 收到请求 [{session_id}]: {question} ==========")
    config = {"configurable": {"thread_id": session_id}}
    init_state = {"question": question, "intent": "", "evidence_list": [], "answer": ""}
    
    token = current_org_code.set("3101")
    
    try:
        yield f"data: {json.dumps({'status': 'Agent 正在分析意图并调度工具...'}, ensure_ascii=False)}\n\n"
        
        # 👇 1. 先跑图，拿到完整的 State（此时图里面只查库，不调大模型）
        result = await app_graph.ainvoke(init_state, config=config)
        
        # 如果识别为 REPORT（报告生成），图里面已经调过大模型了，直接返回结果
        if result.get("intent") == "REPORT":
            yield f"data: {json.dumps({'text': result.get('answer', '')}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 👇 2. 如果是常规问答，拿到证据后，我们在这里手动开启流式大模型
        evidence_list = result.get("evidence_list", [])
        context = "\n".join(evidence_list)
        prompt = f"你是一个财务专家。请根据以下收集到的证据，回答用户问题：\n\n【证据列表】\n{context}\n\n【用户问题】\n{question}"
        
        # 强制开启 streaming=True
        llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, streaming=True)
        
        yield f"data: {json.dumps({'status': '证据收集完毕，大模型正在生成回答...'}, ensure_ascii=False)}\n\n"
        
        # 👇 3. 直接流式吐给前端
        async for chunk in llm.astream(prompt):
            if chunk.content:
                yield f"data: {json.dumps({'text': chunk.content}, ensure_ascii=False)}\n\n"
                
    except Exception as e:
        print(f"❌ 报错: {e}")
        yield f"data: {json.dumps({'text': f'系统异常: {e}'}, ensure_ascii=False)}\n\n"
    finally:
        current_org_code.reset(token)
        
    yield "data: [DONE]\n\n"

@app.get("/api/agent/chat/stream")
async def chat_stream(question: str, session_id: str = "user_001"):
    return StreamingResponse(stream_generator(question, session_id), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)