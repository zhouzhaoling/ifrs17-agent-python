from fastapi import FastAPI
from langchain_ollama import ChatOllama
import uvicorn

app = FastAPI(title="IFRS17 Agent Python版")
llm = ChatOllama(model="qwen2.5:7b")

@app.get("/api/agent/chat")
async def chat(question: str):
    print(f">>> 收到用户请求: {question}")
    response = await llm.ainvoke(question)
    return {"answer": response.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)