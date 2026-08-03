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
4. IT·기술 관련 질문인데 search_it_wiki 결과가 "관련 문서를 찾지 못했습니다."이면,
   같은 질문으로 search_realtime_news를 한 번 더 호출해 최신 정보를 찾아보십시오.
   이 경우 답변 첫 줄에 "위키 문서에는 없어 최신 뉴스에서 찾은 내용입니다."라고 밝히십시오.
   IT·기술과 무관한 주제에는 이 폴백을 적용하지 마십시오.
5. 도구를 사용했는데도 결과가 없으면 모른다고 솔직히 답하고, 추측해서 지어내지 마십시오.

답변 형식 규칙:
6. search_realtime_news 결과를 사용해 답변할 때는 각 항목마다 먼저 도구가 준
   '요약' 내용을 한두 문장의 한국어 서술문으로 정리해 쓰십시오. 이 요약이 본문이며
   생략할 수 없습니다. 그 문장 끝에 [기사 보기](링크) 형태의 마크다운 링크를
   덧붙이되, 대괄호 안 글자는 반드시 "기사 보기"로 고정하십시오.
   기사 제목이나 다른 문구를 대괄호 안에 넣지 마십시오.
   기사 제목만 링크로 나열하는 형태로 답하지 마십시오.
   링크 주소는 도구가 준 '링크' 값을 그대로 쓰고 임의로 만들어내지 마십시오.
   각 뉴스 항목은 반드시 "- "로 시작하는 마크다운 목록 형태로 작성하십시오.
7. 도구를 사용해 답변할 때는 검색된 내용에만 근거하십시오."""

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