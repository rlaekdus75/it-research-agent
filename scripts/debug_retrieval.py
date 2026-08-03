"""
[scripts/debug_retrieval.py]

리랭킹 ON/OFF일 때 실제로 어떤 문서가 검색되는지 눈으로 비교하는 진단용 스크립트.
LLM(Gemini) 호출은 하지 않고 검색 단계만 돌린다.

실행: 저장소 루트에서 python3 scripts/debug_retrieval.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_ROOT, "src"))

from retriever import load_hybrid_retriever

FAISS_PATH = os.path.join(_ROOT, "data", "faiss_index_it")

# 확인하고 싶은 질문 (여기만 바꿔가며 재실행하면 된다)
QUESTION = "인공지능이 뭐야?"


def show(use_reranker: bool):
    label = "ON" if use_reranker else "OFF"
    print("\n" + "=" * 70)
    print(f"리랭킹 {label} / 질문: {QUESTION}")
    print("=" * 70)

    retriever = load_hybrid_retriever(FAISS_PATH, use_reranker=use_reranker)
    docs = retriever.invoke(QUESTION)

    print(f"\n검색된 문서 {len(docs)}개")
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title") or doc.metadata.get("source") or "(제목 메타데이터 없음)"
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"\n  [{i}] 출처: {title}")
        print(f"      {preview}...")


if __name__ == "__main__":
    show(use_reranker=False)
    show(use_reranker=True)