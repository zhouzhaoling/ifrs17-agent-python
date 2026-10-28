from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import uvicorn

# ==========================================
# 1. 定义全局 State (相当于 Java 的 Ifrs17AgentState)
# ==========================================
class AgentState(TypedDict):
    question: str
    answer: str
    history: str

# ==========================================
# 2. 定义节点 (Node)
# ==========================================
def llm_chat_node(state: AgentState):
    print(f">>> [Agent节点执行] 收到问题: {state['question']}")
    
    # 初始化本地模型
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)
    
    # 获取历史记录（后续阶段使用，现在为空）
    history = state.get("history", "")
    
    # 拼装 Prompt
    messages = [
        ("system", "你是一个 IFRS17 财务核算专家。请参考历史对话作答。\n历史记录：\n" + history),
        ("user", state["question"])
    ]
    
    # 调用大模型
    response = llm.invoke(messages)
    
    # 返回需要更新的 State 字段
    return {"answer": response.content}

# ==========================================
# 3. 构建状态图 (Graph)
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("chat_node", llm_chat_node)

# 添加边（路由）
workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", END)

# 编译图（相当于 Java 的 compile()）
app_graph = workflow.compile()


# ==========================================
# 4. FastAPI 接口
# ==========================================
app = FastAPI(title="IFRS17 Agent Python版")

@app.get("/api/agent/chat")
async def chat(question: str):
    # 初始化 State
    init_state = {
        "question": question,
        "answer": "",
        "history": ""
    }
    
    # 运行图 (注意：这里用 ainvoke 支持异步)
    result = await app_graph.ainvoke(init_state)
    
    return {"answer": result["answer"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)