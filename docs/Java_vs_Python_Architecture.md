# Java 版 vs Python 版：Agent 架构代码级差异对比

在我的 IFRS17 智能财务 Agent 平台落地过程中，为了验证架构的跨语言兼容性，我分别用 Java (Spring AI) 和 Python (LangChain/FastAPI) 实现了两套原型。以下是核心代码层面的深度对比与设计思考：

## 1. 状态合并机制（Reducer）：极简 vs 严谨
在多源并发收集证据时，为防止数据覆盖，需使用 Reducer 进行状态合并。

**Python 版**：利用字典和 `operator.add`，一行代码声明。
```python
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    # 直接声明：新列表会自动追加到旧列表
    evidence_list: Annotated[List[str], operator.add] 
```

**Java 版**：由于强类型约束，必须在 State 类中显式重写 `SCHEMA`，利用 `Channel` 通道机制。
```java
public class Ifrs17AgentState extends AgentState {
    // 必须显式声明 Map 和 Channel，指定使用 appender
    public static final Map<String, Channel<?>> SCHEMA = Map.of(
            "evidence_list", Channels.appender(ArrayList::new)
    );
}
```

## 2. 并发查库（并发编排）：协程 vs 线程池
为了实现“同时查明细和查规则”：

**Python 版**：使用原生 `asyncio.gather`，由事件循环驱动，内存占用极低，代码形似同步。
```python
async def diff_analysis_node(state):
    # 直接并发执行两个异步函数
    results = await asyncio.gather(fetch_detail(), fetch_rule())
    return {"evidence_list": results}
```

**Java 版**：需引入线程池（`ExecutorService`），手动组合 `CompletableFuture`，处理 `allOf` 和 `get()`，对底层线程控制精准。
```java
CompletableFuture<String> f1 = CompletableFuture.supplyAsync(() -> fetchDetail(), executor);
CompletableFuture<String> f2 = CompletableFuture.supplyAsync(() -> fetchRule(), executor);
CompletableFuture.allOf(f1, f2).join(); // 等待合并
```

## 3. 数据权限隔离（Context）：ContextVar vs ThreadLocal
为了防止大模型越权查其他机构的数据（行级隔离）：

**Python 版**：FastAPI 基于异步协程，传统线程变量会失效。故使用 `ContextVar`，在协程切换时保持上下文隔离。
```python
from contextvars import ContextVar
current_org_code = ContextVar("current_org_code", default="3101")
tenant_id = current_org_code.get()
```

**Java 版**：Spring Boot 默认 Servlet 线程模型，使用 `ThreadLocal`，配合 MyBatis 拦截器实现底层 SQL 追加。
```java
public class UserContext {
    private static final ThreadLocal<String> ORG_CODE = new ThreadLocal<>();
    public static void setOrgCode(String code) { ORG_CODE.set(code); }
}
```

## 4. 工具定义（Tool Calling）：鸭子类型 vs 注解反射
让大模型调用后端函数：

**Python 版**：加个 `@tool` 装饰器，靠解析函数的 Docstring 和类型提示生成 JSON Schema，轻量直接。
```python
@tool
def query_gl_database(org_name: str) -> str:
    """查询底层总账数据库，获取指定机构的准备金金额明细。"""
    ...
```

**Java 版**：通过 `@Tool` 和 `@ToolParam` 双重注解，框架启动时通过反射扫描注解，生成大模型所需的入参 Schema。
```java
@Tool(description = "查询底层总账数据库...")
public String queryGlDatabase(
    @ToolParam(description = "机构名称") String orgName) {
    ...
}
```

## 5. 流式输出（SSE）：生成器 vs 响应式流
把大模型的结果实时推给前端：

**Python 版**：定义一个异步生成器（`async def` 搭配 `yield`），用 `StreamingResponse` 直接推流，极其轻量。
```python
async def stream_generator(...):
    async for chunk in llm.astream(prompt):
        yield f"data: {json.dumps({'text': chunk})}\n\n"
```

**Java 版**：需要借助 Spring WebFlux 的响应式流。底层返回的是 `Flux<String>`，配合 `SseEmitter` 或者 `Flux` 直接作为 Controller 返回值，工程结构更重，但天生支持高并发背压。
```java
@GetMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<String> streamChat(...) {
    return chatClient.prompt().user(...).stream().content();
}
```

## 💡 架构师总结：语言只是工具，架构才是灵魂
**Python 是 AI 的原生母语**，代码极其精简，完美契合了快速验证（POC）和复杂 AI 算法的敏捷迭代。
**Java 则是企业级工程的防线**，稍显“啰嗦”的代码背后（强类型约束、显式声明、线程池控制），保证了在金融级高并发场景下的绝对稳定与可控。
这两套源码足以证明：我具备跨语言的架构驾驭能力，能根据业务场景（敏捷验证 vs 生产落地）选择最合适的技术栈。