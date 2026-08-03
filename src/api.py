"""
[src/api.py]
agent.py의 Agent를 FastAPI로 감싸서 REST API + 웹 채팅 UI로 배포한다.
실행: python3 src/api.py (저장소 루트에서)
채팅 UI: http://localhost:8003
Swagger: http://localhost:8003/docs
"""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage
from agent import graph

app = FastAPI(title="rain-research-bot")

# 정적 파일 (채팅 UI) 서빙
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


class AskRequest(BaseModel):
    question: str
    thread_id: str = "default"


class AskResponse(BaseModel):
    answer: str
    tool_used: Optional[str] = None
    tool_query: Optional[str] = None
    trace: list[str] = []
    contexts: list[str] = []

@app.get("/")
def serve_ui():
    """채팅 UI 페이지 반환"""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="질문이 비어있습니다.")

    config = {"configurable": {"thread_id": request.thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=request.question)]}, config=config)

    tool_used = None
    tool_query = None
    trace = []
    contexts = []
    for msg in result["messages"]:
        # 사용자 질문을 만나면 초기화 (이번 질문 이후의 기록만 남긴다)
        if getattr(msg, "type", None) == "human":
            tool_used, tool_query, trace, contexts = None, None, [], []
            continue
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                trace.append(call["name"])
            tool_used = msg.tool_calls[0]["name"]
            tool_query = msg.tool_calls[0].get("args", {}).get("query")
        if isinstance(msg, ToolMessage):
            contexts.append(msg.content)

    answer = result["messages"][-1].content
    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if not answer:
        raise HTTPException(status_code=500, detail="답변 생성에 실패했습니다.")

    return AskResponse(
        answer=answer,
        tool_used=tool_used,
        tool_query=tool_query,
        trace=trace,
        contexts=contexts,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)