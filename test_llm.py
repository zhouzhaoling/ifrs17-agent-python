from langchain_ollama import ChatOllama

# 初始化本地模型
llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

print("========== 正在测试连接本地 Ollama ==========")
response = llm.invoke("你好，请用一句话介绍你自己，并告诉我IFRS17是什么。")
print("【模型回复】:", response.content)
print("=============================================")