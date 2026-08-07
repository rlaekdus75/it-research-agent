"""
[src/agent.py]

교재 코드의 변수명/구조를 그대로 따름 (builder, agent, tool_node, graph).
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from llm import build_llm  # [변경] generator.py -> llm.py로 이동됨
from tools import search_it_wiki, search_realtime_news

# === LLM과 도구 바인딩 === [교재]
tools = [search_it_wiki, search_realtime_news]
llm = build_llm()
llm_with_tools = llm.bind_tools(tools)
SYSTEM_PROMPT = """당신은 IT 개념과 실시간 뉴스를 안내하는 한국어 리서치 어시스턴트입니다.

대화 규칙:
1. 인사, 잡담, 감정 표현, 정보 검색이 필요 없는 짧은 대화에는 도구를 호출하지 말고
   친근하고 자연스럽게 직접 답하십시오.
   이때 "제가 도와드릴 수 있는 범위는..." 같은 기능 안내를 덧붙이지 마십시오.

도구 사용 규칙 (정보를 찾아야 하는 질문일 때만 적용):
2. 시간이 지나도 변하지 않는 IT 개념·정의 질문에는 search_it_wiki를 사용하십시오.
3. 최신 소식·기업 동향 질문에는 search_realtime_news를 사용하십시오.
4. 도구 결과가 "위키 인덱스에 문서가 없어 최신 뉴스로 대체합니다."로 시작하면,
   그 아래에 붙어 있는 뉴스 목록만을 근거로 답변하십시오.
   이때 답변 첫 줄에 "위키 문서에는 없어 최신 뉴스에서 찾은 내용입니다."라고 밝히십시오.
   이 경우 search_realtime_news를 추가로 호출하지 마십시오. 필요한 뉴스는 이미 결과에 포함돼 있습니다.
   단, 그 뉴스 목록에도 질문에 답할 내용이 없으면 안내 문구를 붙이지 말고 찾지 못했다고만 답하십시오.
5. 도구 결과에 답변할 내용이 없으면 찾지 못했다고 솔직히 답하십시오.
   도구가 주지 않은 내용을 추측해서 지어내지 마십시오.
   특히 도구 결과에 링크가 없으면 답변에도 링크를 넣지 마십시오.

답변 형식 규칙:
6. 도구 결과가 "제목:", "요약:", "링크:" 형식의 뉴스 목록이면 각 항목마다 먼저 도구가 준
   '요약' 내용을 한두 문장의 한국어 서술문으로 정리해 쓰십시오. 이 요약이 본문이며
   생략할 수 없습니다. 그 문장 끝에 [기사 보기](링크) 형태의 마크다운 링크를
   덧붙이되, 대괄호 안 글자는 반드시 "기사 보기"로 고정하십시오.
   기사 제목이나 다른 문구를 대괄호 안에 넣지 마십시오.
   기사 제목만 링크로 나열하는 형태로 답하지 마십시오.
   링크 주소는 도구가 준 '링크' 값을 그대로 쓰고 임의로 만들어내지 마십시오.
   각 뉴스 항목은 반드시 "- "로 시작하는 마크다운 목록 형태로 작성하십시오.
7. 도구를 사용해 답변할 때는 검색된 내용에만 근거하십시오.
8. 규칙 번호, 도구 이름, 도구가 반환한 안내 문구 같은 내부 동작을 사용자에게 설명하지 마십시오."""

# === agent 노드: LLM 호출 === [교재]
def agent(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# === ToolNode 생성 === [교재]
tool_node = ToolNode(tools=tools)

# === 그래프 빌더 === [교재]
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

# === 엣지 구성 === [교재]
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)
builder.add_edge("tools", "agent")

# === 컴파일 === [교재]
memory = MemorySaver()  # 대화 기억용 보관함
graph = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    test_questions = [
        "인공지능이 뭐야?",
        "오늘 삼성전자 관련 뉴스 알려줘",
    ]

    for q in test_questions:
        print(f"\n{'=' * 50}")
        print(f"질문: {q}")
        config = {"configurable": {"thread_id": "test-thread"}}
        result = graph.invoke({"messages": [HumanMessage(content=q)]}, config=config)

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [tc["name"] for tc in msg.tool_calls]
                print(f"  -> 호출된 도구: {tool_names}")

        print(f"답변: {result['messages'][-1].content}")