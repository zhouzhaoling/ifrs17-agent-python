from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader # 👈 新增 PDF 解析
from langchain_text_splitters import RecursiveCharacterTextSplitter # 👈 新增 文本切分
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
import shutil
import pandas as pd # 👈 新增 Pandas 生成 Excel

# ==========================================
# 0. 环境变量与静态资源挂载
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
app = FastAPI(title="IFRS17 Agent Python 终局版")

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 确保文件存储目录存在
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================
# 1. 初始化本地 RAG 知识库 (Chroma)
# ==========================================
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(embedding_function=embeddings, collection_name="ifrs17_rules")
if vectorstore._collection.count() == 0:
    docs = [Document(page_content="UAT报错E1001：准备金计算溢出...", metadata={"source": "UAT手册"})]
    vectorstore.add_documents(docs)

# ==========================================
# 2. 初始化 SQLite 语义层
# ==========================================
conn = sqlite3.connect("semantic_layer.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS metric_dict (metric_name TEXT, formula TEXT, source_table TEXT, default_filter TEXT)''')
cursor.execute("SELECT COUNT(*) FROM metric_dict")
if cursor.fetchone()[0] == 0:
    cursor.execute("INSERT INTO metric_dict VALUES ('准备金余额', 'SUM(closing_balance)', 'dws_ifrs17_reserve', 'reserve_type = NET')")
    conn.commit()

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
    if "上海" in org_name:
        return "数据库结果：上海机构准备金旧账套 1000 万，新账套 1250 万。"
    return "数据库结果：未找到该机构数据。"

@tool
def query_metric_definition(metric_name: str) -> str:
    """查询财务指标的语义定义。"""
    cursor.execute("SELECT metric_name, formula, source_table, default_filter FROM metric_dict WHERE metric_name LIKE ?", (f"%{metric_name}%",))
    result = cursor.fetchone()
    if result:
        return f"【语义层定义】指标：{result[0]}, 公式：{result[1]}, 来源表：{result[2]}, 默认过滤：{result[3]}"
    return "未在语义层中找到该指标。"

# ==========================================
# 5. 定义节点 (Node)
# ==========================================
def router_node(state: AgentState):
    llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)
    prompt = "你是一个意图识别助手。返回以下之一：DIFF（差异排查）, METRIC（指标查询）, KNOWLEDGE（知识问答）, REPORT（生成报告/导出Excel）。只输出单词。"
    response = llm.invoke([("system", prompt), ("user", state["question"])])
    intent = response.content.strip()
    if intent not in ["DIFF", "METRIC", "KNOWLEDGE", "REPORT"]:
        intent = "METRIC"
    print(f">>> [Router] 意图: {intent}")
    return {"intent": intent}

async def diff_analysis_node(state: AgentState):
    async def fetch_detail():
        return query_gl_database.invoke("上海")
    async def fetch_rule():
        return "规则：折现率必须使用即期利率曲线。"
    results = await asyncio.gather(fetch_detail(), fetch_rule())
    return {"evidence_list": results}

def metric_query_node(state: AgentState):
    llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)
    llm_with_tools = llm.bind_tools([query_metric_definition])
    response = llm_with_tools.invoke(state["question"])
    evidence = []
    if response.tool_calls:
        for tc in response.tool_calls:
            evidence.append(query_metric_definition.invoke(tc["args"]))
    else:
        evidence.append("模型未触发指标查询。")
    return {"evidence_list": evidence}

def rag_query_node(state: AgentState):
    results = vectorstore.similarity_search(state["question"], k=1)
    info = f"【RAG检索】: {results[0].page_content}" if results else "未找到。"
    return {"evidence_list": [info]}

def report_generation_node(state: AgentState):
    print(">>> [报告生成节点] 启动，正在生成专业财务分析报告与 Excel 文件...")
    
    # 1. 生成模拟的 Excel 报表数据
    data = {
        "科目": ["未到期责任准备金", "未决赔款准备金", "费用分摊"],
        "旧账套 (万元)": [1000, 500, 200],
        "新账套 (万元)": [1250, 480, 220],
        "差异金额 (万元)": [250, -20, 20]
    }
    df = pd.DataFrame(data)
    
    filename = f"IFRS17_差异分析报表_{int(time.time())}.xlsx"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    # 写入 Excel
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='差异明细', index=False)
    print(f">>> [报告生成节点] Excel 文件已落盘: {filename}")

    # 2. 用大模型生成 Markdown 分析文本
    llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)
    prompt = """
    你是一个顶级的 IFRS17 财务分析师。请根据用户的提问，生成一份严谨的分析报告。
    必须包含：## 一、概况、## 二、原因分析、## 三、影响评估、## 四、建议。
    """
    response = llm.invoke([("system", prompt), ("user", state["question"])])
    
    # 3. 巧妙拼接：在 Markdown 文本末尾加上带文件名的下载链接
    download_url = f"http://127.0.0.1:8000/download/{filename}"
    markdown_answer = f"{response.content}\n\n---\n### 📎 报表附件\n[👉 点击此处下载完整的 Excel 数据分析报表]({download_url})"
    
    return {"answer": markdown_answer}

async def analyze_node(state: AgentState):
    llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, streaming=True)
    context = "\n".join(state["evidence_list"])
    prompt = f"根据以下证据回答问题：\n【证据】\n{context}\n\n【问题】\n{state['question']}"
    response = await llm.ainvoke(prompt)
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
    {"DIFF": "diff_analysis", "METRIC": "metric_query", "KNOWLEDGE": "rag_query", "REPORT": "report_generation"}
)
workflow.add_edge("diff_analysis", END)
workflow.add_edge("metric_query", END)
workflow.add_edge("rag_query", END)
workflow.add_edge("report_generation", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# ==========================================
# 7. FastAPI 接口 (流式对话 & 文件上传 & 文件下载)
# ==========================================
async def stream_generator(question: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    init_state = {"question": question, "intent": "", "evidence_list": [], "answer": ""}
    token = current_org_code.set("3101")
    
    try:
        yield f"data: {json.dumps({'status': 'Agent 正在分析并调度工具...'}, ensure_ascii=False)}\n\n"
        result = await app_graph.ainvoke(init_state, config=config)
        
        # 如果走的是报告生成（REPORT），直接把这整段（包含下载链接）吐给前端
        if result.get("intent") == "REPORT":
            yield f"data: {json.dumps({'text': result.get('answer', '')}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        evidence_list = result.get("evidence_list", [])
        context = "\n".join(evidence_list)
        prompt = f"你是一个财务专家。根据以下证据回答：\n{context}\n\n问题：\n{question}"
        llm = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, streaming=True)
        
        yield f"data: {json.dumps({'status': '正在生成回答...'}, ensure_ascii=False)}\n\n"
        async for chunk in llm.astream(prompt):
            if chunk.content:
                yield f"data: {json.dumps({'text': chunk.content}, ensure_ascii=False)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'text': f'异常: {e}'}, ensure_ascii=False)}\n\n"
    finally:
        current_org_code.reset(token)
    yield "data: [DONE]\n\n"

@app.get("/api/agent/chat/stream")
async def chat_stream(question: str, session_id: str = "user_001"):
    return StreamingResponse(stream_generator(question, session_id), media_type="text/event-stream")

# 👇 新增：PDF 上传接口
@app.post("/api/upload_doc")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 解析、切分、入库
    loader = PyPDFLoader(file_path)
    splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
    vectorstore.add_documents(splits)
    return {"message": f"成功解析 {file.filename}，切分为 {len(splits)} 个片段并入库！"}

# 👇 新增：Excel 文件下载接口
@app.get("/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return {"error": "文件不存在"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)