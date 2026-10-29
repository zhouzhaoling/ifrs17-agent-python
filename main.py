from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
import operator
import uvicorn
import asyncio

# ==========================================
# 1. 定义全局 State（重点体现文档里的 Reducer）
# ==========================================
class AgentState(TypedDict):
    question: str
    intent: str
    evidence_list: Annotated[List[str], operator.add]
    answer: str

# ==========================================
# 2. 定义各个节点 (Node)
# ==========================================

# 节点 A：Router 路由分发
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

# 节点 B：并发收集证据
async def diff_analysis_node(state: AgentState):
    print(">>> [并发节点] 正在并行收集明细和规则证据...")
    
    async def fetch_detail():
        await asyncio.sleep(1)
        return "明细数据：上海机构准备金旧账套 1000 万，新账套 1250 万。"
        
    async def fetch_rule():
        await asyncio.sleep(1)
        return "规则约束：新准则规定，准备金折现率必须使用即期利率曲线。"

    results = await asyncio.gather(fetch_detail(), fetch_rule())
    return {"evidence_list": results}

# 节点 C：查指标口径（模拟语义层工具）
def query_metric_node(state: AgentState):
    print(">>> [语义层工具] 正在查询指标元数据...")
    metric_info = "指标定义：准备金余额 = SUM(closing_balance)，默认过滤条件 reserve_type = 'NET'。"
    return {"evidence_list": [metric_info]}

# 节点 D：汇总分析并生成最终回答
def analyze_node(state: AgentState):
    print(f">>> [汇总分析] 当前已收集到 {len(state['evidence_list'])} 条证据，交由大模型总结...")
    llm = ChatOllama(model="qwen2.5:7b")
    
    context = "\n".join(state["evidence_list"])
    prompt = f"你是一个财务专家。请根据以下收集到的证据，回答用户问题：\n\n【证据列表】\n{context}\n\n【用户问题】\n{state['question']}"
    
    response = llm.invoke(prompt)
    return {"answer": response.content}


# ==========================================
# 3. 构建图 (Graph)
# ==========================================
workflow = StateGraph(AgentState)

# 👇 这里是你之前漏掉的关键一行：必须把 router 节点加进去！
workflow.add_node("router", router_node)

# 注册其他节点
workflow.add_node("diff_analysis", diff_analysis_node)
workflow.add_node("query_metric", query_metric_node)
workflow.add_node("analyze", analyze_node)

# 设置起点
workflow.add_edge(START, "router")

# 条件路由
workflow.add_conditional_edges(
    "router",
    lambda x: x["intent"],
    {
        "DIFF": "diff_analysis",
        "METRIC": "query_metric"
    }
)

# 汇总节点
workflow.add_edge("diff_analysis", "analyze")
workflow.add_edge("query_metric", "analyze")
workflow.add_edge("analyze", END)

# 编译图
app_graph = workflow.compile()


# ==========================================
# 4. FastAPI 接口
# ==========================================
app = FastAPI(title="IFRS17 Agent 企业级 Python 版")

@app.get("/api/agent/chat")
async def chat(question: str):
    print(f"\n========== 收到用户请求: {question} ==========")
    init_state = {
        "question": question,
        "intent": "",
        "evidence_list": [],
        "answer": ""
    }
    
    result = await app_graph.ainvoke(init_state)
    return {"answer": result["answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)