"""
[src/tools.py]

[교재: Tool Calling] 패턴 그대로 적용.
tool 데코레이터로 일반 파이썬 함수를 LLM이 호출 가능한 도구로 변환한다.
"""
import os
from langchain_core.tools import tool

from retriever import load_hybrid_retriever
from naver_news_tool import search_news

# data/ 폴더 경로 계산 (이 파일 위치 기준: src/ -> 저장소 루트 -> data/faiss_index_it)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_FAISS_PATH = os.path.join(_DATA_DIR, "faiss_index_it")

# 평가용: USE_RERANKER=false 로 실행하면 리랭킹 없이(v1 baseline) 동작
_USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() != "false"
_it_retriever = load_hybrid_retriever(_FAISS_PATH, use_reranker=_USE_RERANKER)


def _format_news(results: list) -> str:
    return "\n\n".join(
        f"제목: {r['title']}\n요약: {r['description']}\n링크: {r['link']}"
        for r in results
    )


@tool
def search_it_wiki(query: str) -> str:
    """IT 개념이나 배경지식을 설명하는 위키 문서를 검색합니다.
    '인공지능이 뭐야?', 'HTTP란?', '블록체인 원리' 처럼
    시간이 지나도 안 바뀌는 정의/개념 질문에 사용합니다."""
    docs = _it_retriever.invoke(query)
    if docs:
        return "\n\n".join(d.page_content for d in docs)

    results = search_news(query)
    if not results:
        return "위키 인덱스와 뉴스 검색 모두에서 결과를 찾지 못했습니다."
    return "위키 인덱스에 문서가 없어 최신 뉴스로 대체합니다.\n\n" + _format_news(results)


@tool
def search_realtime_news(query: str) -> str:
    """최신 시사 뉴스나 기업 소식을 실시간으로 검색합니다.
    '오늘 삼성전자 뉴스', '카카오 최근 소식' 처럼
    매번 바뀌는 최신 정보 질문에 사용합니다."""
    results = search_news(query)
    if not results:
        return "관련 뉴스를 찾지 못했습니다."
    return _format_news(results)