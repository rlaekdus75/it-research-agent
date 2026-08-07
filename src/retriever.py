"""
retriever.py - 하이브리드 검색기 (FAISS + BM25/Mecab) + Reranking + Query Expansion

검색 흐름 (v3 - 전체 적용 시):
  0차: 쿼리 익스팬션 - Gemini가 질문을 여러 버전으로 바꿔서 검색 범위 확대
  1차: FAISS + BM25 하이브리드로 각 버전별 후보를 넉넉하게 뽑고 합침
  2차: Cross-encoder(bge-reranker-v2-m3)로 "질문-문서" 쌍을 재채점
  3차: 재채점 상위 3개만 최종 반환

베이스라인 비교를 위해 플래그로 각 기능을 on/off 할 수 있음:
  use_reranker=False, use_query_expansion=False -> v1 (baseline)
  use_reranker=True,  use_query_expansion=False -> v2 (reranking only)
  use_reranker=True,  use_query_expansion=True  -> v3 (reranking + query expansion)
  
"""
import logging
import warnings

logging.getLogger("langchain").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab
from sentence_transformers import CrossEncoder

import os

# mecab-ko-dic 위치는 환경마다 다르므로, 후보를 순서대로 확인해서 실제 존재하는 경로를 씀
_MECAB_DICPATH_CANDIDATES = [
    os.environ.get("MECAB_DICPATH"),
    os.environ.get("MECAB_DIC_PATH"),
    "/usr/local/lib/mecab/dic/mecab-ko-dic",   # 리눅스: 소스 빌드 (Docker)
    "/opt/homebrew/lib/mecab/dic/mecab-ko-dic",  # 맥: Homebrew
    "/usr/lib/mecab/dic/mecab-ko-dic",         # 리눅스: apt 설치
]


def _resolve_mecab_dicpath():
    for path in _MECAB_DICPATH_CANDIDATES:
        if path and os.path.isdir(path):
            return path
    return None


MECAB_DICPATH = _resolve_mecab_dicpath()


def load_hybrid_retriever(
    faiss_path,
    embedding_model="BAAI/bge-m3",
    reranker_model="BAAI/bge-reranker-v2-m3",
    k=3,
    initial_k=10,
    device="cpu",
    score_threshold=0.5,
    alpha=0.5,
    bm25_min_score=12.0,
    relaxed_score_threshold=0.3,
    relaxed_bm25_min_score=6.0,
    use_reranker=True,
    use_query_expansion=False,   # v3: 쿼리 익스팬션 on/off
    min_chunk_chars=80,          # 이보다 짧은 청크는 근거로 쓰지 않음
):
    """
    FAISS + BM25 하이브리드 검색 + Cross-encoder 리랭킹.

    use_reranker=True:  1차로 initial_k(10)개 뽑고, 리랭커로 재채점 후 상위 k(3)개 반환
    use_reranker=False: 기존 방식 그대로 (리랭킹 전 베이스라인과 비교할 때 사용)
    """
    print("임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("FAISS 벡터 저장소 로드 중...")
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    all_docs = list(vectorstore.docstore._dict.values())
    for i, doc in enumerate(all_docs):
        doc.metadata["_idx"] = i
    print(f"문서 {len(all_docs)}개 로드 완료")

    mecab = Mecab(dicpath=MECAB_DICPATH) if MECAB_DICPATH else Mecab()

    print("BM25 인덱스 구축 중 (Mecab으로 명사 추출)...")
    tokenized_corpus = [mecab.nouns(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 인덱스 구축 완료")

    reranker = None
    if use_reranker:
        print(f"리랭커 로드 중... ({reranker_model})")
        reranker = CrossEncoder(reranker_model)
        print("리랭커 로드 완료")

    def _search_once(question, vec_threshold, bm25_threshold, num_results):
        vec_results = vectorstore.similarity_search_with_relevance_scores(
            question, k=len(all_docs)
        )
        vec_scores = {
            doc.metadata["_idx"]: score
            for doc, score in vec_results
            if score >= vec_threshold
        }

        tokenized_query = mecab.nouns(question)
        bm25_scores_raw = bm25.get_scores(tokenized_query) if tokenized_query else []

        relevant_raw = [s for s in bm25_scores_raw if s >= bm25_threshold]
        max_bm25 = max(relevant_raw) if relevant_raw else 1
        bm25_scores = {
            i: bm25_scores_raw[i] / max_bm25
            for i in range(len(bm25_scores_raw))
            if bm25_scores_raw[i] >= bm25_threshold
        }

        candidate_ids = set(vec_scores) | set(bm25_scores)
        combined = {
            idx: alpha * vec_scores.get(idx, 0) + (1 - alpha) * bm25_scores.get(idx, 0)
            for idx in candidate_ids
        }

        top_ids = sorted(combined, key=combined.get, reverse=True)[:num_results]
        return [all_docs[i] for i in top_ids]

    def _rerank(question, docs):
        if not docs:
            return []
        pairs = [[question, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        filtered = [(doc, score) for doc, score in ranked if score > 0.01]
        return [doc for doc, score in filtered[:k]]

    def _expand_query(question):
        """Gemini를 사용해 질문을 여러 버전으로 확장 (1번 호출로 3개 버전 생성)"""
        from llm import build_llm
        import json
        expand_llm = build_llm()
        prompt = f"""다음 질문을 검색 성능을 높이기 위해 3가지 다른 표현으로 바꿔주세요.
원본 질문의 의미를 유지하면서, 다른 단어/관점으로 표현해주세요.
JSON 배열로만 답하세요 (다른 텍스트 없이):
["변형1", "변형2", "변형3"]

질문: {question}"""
        response = expand_llm.invoke(prompt)
        text = response.content.strip().replace("```json", "").replace("```", "").strip()
        try:
            variants = json.loads(text)
            return [question] + variants  # 원본 + 변형 3개 = 총 4개
        except json.JSONDecodeError:
            return [question]  # 파싱 실패하면 원본만 사용

    def retrieve(question):
        search_k = initial_k if use_reranker else k

        # ---- 쿼리 익스팬션 ----
        if use_query_expansion:
            queries = _expand_query(question)
            print(f"  쿼리 익스팬션: {queries}")
        else:
            queries = [question]

        # ---- 여러 쿼리로 검색해서 결과 합치기 ----
        all_results = []
        seen_ids = set()
        for q in queries:
            results = _search_once(q, score_threshold, bm25_min_score, search_k)
            if not results:
                results = _search_once(q, relaxed_score_threshold, relaxed_bm25_min_score, search_k)
            for doc in results:
                doc_id = doc.metadata.get("_idx")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(doc)

        # ---- 근거가 될 수 없는 초단문 청크 제외 ----
        all_results = [d for d in all_results if len(d.page_content) >= min_chunk_chars]

        # ---- 리랭킹 ----
        if use_reranker and all_results:
            all_results = _rerank(question, all_results)  # 원본 질문 기준으로 재채점

        return all_results[:k]
    return RunnableLambda(retrieve)


load_retriever = load_hybrid_retriever


if __name__ == "__main__":
    import sys
    import os
    _DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    FAISS_PATH = os.path.join(_DATA_DIR, "faiss_index_it")

    retriever = load_hybrid_retriever(FAISS_PATH)

    test_questions = [
        "인공지능이 뭐야?",
        "딥러닝과 머신러닝의 차이는?",
        "클라우드 컴퓨팅의 장점은?",
        "오늘 점심 뭐 먹지?",
    ]
    for q in test_questions:
        results = retriever.invoke(q)
        print(f"\n질문: {q}")
        print(f"검색된 문서 수: {len(results)}개")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.page_content[:100]}...")
