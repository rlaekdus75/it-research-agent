# [evaluation/run_evaluation.py]
#
# 1. 라우팅 정확도: Agent가 질문 종류에 맞는 도구를 골랐는지 (10문항 -> 6문항으로 축소)
# 2. 답변 품질: IT 질문 3개에 대해 5개 지표로 채점 (호출 아끼려고 통합 채점 재사용)
#
# 실행: 저장소 루트에서 python3 evaluation/run_evaluation.py

import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, ToolMessage
from agent import graph

from qa_dataset import ROUTING_QUESTIONS, QUALITY_QUESTIONS
from metrics import keyword_recall, judge_all


def run_agent(question: str, thread_id: str):
    """Agent 호출해서 (답변, 사용된 도구, 도구 결과 텍스트) 반환"""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)

    tool_used = None
    tool_output = ""
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_used = msg.tool_calls[0]["name"]
        if isinstance(msg, ToolMessage):
            tool_output = msg.content

    answer = result["messages"][-1].content
    return answer, tool_used, tool_output


def evaluate_routing():
    print("=" * 50)
    print("1. 라우팅 정확도 평가")
    print("=" * 50)

    correct = 0
    routing_results = []
    for i, qa in enumerate(ROUTING_QUESTIONS):
        question = qa["question"]
        answer, tool_used, _ = run_agent(question, thread_id=f"routing-{i}")
        is_correct = tool_used == qa["expected_tool"]
        correct += is_correct

        mark = "O" if is_correct else "X"
        print(f"[{mark}] {question}")
        print(f"    기대: {qa['expected_tool']} / 실제: {tool_used}")

        routing_results.append({
            "question": question,
            "expected_tool": qa["expected_tool"],
            "actual_tool": tool_used,
            "correct": is_correct,
        })

    accuracy = correct / len(ROUTING_QUESTIONS)
    print(f"\n라우팅 정확도: {accuracy:.1%} ({correct}/{len(ROUTING_QUESTIONS)})")
    return routing_results, accuracy


def evaluate_quality():
    print("\n" + "=" * 50)
    print("2. 답변 품질 평가 (IT 질문만)")
    print("=" * 50)

    quality_results = []
    for qa in QUALITY_QUESTIONS:
        question = qa["question"]
        answer, tool_used, context = run_agent(question, thread_id=f"quality-{question}")

        scores = {
            "keyword_recall": keyword_recall(answer, qa["expected_keywords"]),
            **judge_all(question, answer, context, qa["expected_answer"]),
        }

        print(f"\n[질문] {question}")
        print(f"  {scores}")

        quality_results.append({"question": question, "context": context, "answer": answer, "scores": scores})

    return quality_results


def main():
    routing_results, routing_accuracy = evaluate_routing()
    quality_results = evaluate_quality()

    metric_names = ["keyword_recall", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 50)
    print("3. 답변 품질 평균")
    print("=" * 50)
    for m in metric_names:
        avg = sum(r["scores"][m] for r in quality_results) / len(quality_results)
        print(f"  {m}: {avg:.2f}")

    output = {
        "routing_accuracy": routing_accuracy,
        "routing_results": routing_results,
        "quality_results": quality_results,
    }
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n결과 저장 완료: evaluation_results.json")


if __name__ == "__main__":
    main()
