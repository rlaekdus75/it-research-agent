"""
[scripts/collect_wiki.py]

한국어 위키백과에서 IT 관련 문서를 수집하는 스크립트.
결과는 저장소 루트의 data/ 폴더에 저장한다.

실행: 저장소 루트에서 python3 scripts/collect_wiki.py
"""
import json
import os
import time
import wikipediaapi

SEED_TERMS = [
    "알고리즘", "자료 구조", "데이터베이스", "운영 체제", "컴퓨터 네트워크",
    "API", "프로그래밍 언어", "오픈 소스",
    "인공지능", "기계 학습", "딥 러닝", "자연어 처리", "빅 데이터", "데이터 마이닝",
    "클라우드 컴퓨팅", "서버", "HTTP", "웹 브라우저", "도메인 네임 시스템",
    "정보 보안", "암호화", "해킹", "방화벽",
    "블록체인", "사물인터넷", "가상 현실", "5G", "반도체",
    "스타트업", "전자 상거래",
    # --- AI/LLM 계열 (3번 개선: 최신 AI 개발 용어 커버리지 확대) ---
    "랭체인", "대형 언어 모델", "생성형 인공지능", "챗GPT", "오픈AI",
    "트랜스포머 (기계 학습)", "인공 신경망", "강화 학습", "컴퓨터 비전",
    "워드 임베딩", "벡터 공간 모델", "정보 검색", "추천 시스템",
    "프롬프트 엔지니어링", "파운데이션 모델", "허깅 페이스",

    # --- 개발 도구/인프라 계열 ---
    "파이썬", "자바스크립트", "깃 (소프트웨어)", "깃허브", "도커 (소프트웨어)",
    "쿠버네티스", "마이크로서비스", "데브옵스", "리눅스", "리액트 (자바스크립트 라이브러리)",
    "REST", "지속적 통합", "가상 머신",
]

MAX_LINKS_PER_PAGE = 15
MIN_TEXT_LENGTH = 300

# data/ 폴더 경로 (이 파일 위치 기준: scripts/ -> 저장소 루트 -> data/)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(_DATA_DIR, "it_wiki_docs.json")

EXCLUDE_KEYWORDS = ["목록", "연표", "분류:", "틀:", "위키", "동음이의"]

wiki = wikipediaapi.Wikipedia(
    user_agent="rain-research-bot (KTB4 bootcamp project)",
    language="ko",
)


def is_valid_title(title: str) -> bool:
    if any(kw in title for kw in EXCLUDE_KEYWORDS):
        return False
    if title.strip().isdigit():
        return False
    return True


def fetch_page(title: str):
    """위키 문서 1개 가져오기. 일시적 오류는 최대 3번까지 재시도."""
    for attempt in range(3):
        try:
            page = wiki.page(title)
            if not page.exists():
                return None
            if len(page.text) < MIN_TEXT_LENGTH:
                return None
            time.sleep(0.3)
            return page
        except Exception as e:
            print(f"  [재시도 {attempt + 1}/3] {title} - {type(e).__name__}")
            time.sleep(5)
    print(f"  [포기] {title} (3회 시도 실패)")
    return None

def save_docs(collected: dict):
    """지금까지 모은 문서를 파일에 저장 (중간 저장용)"""
    docs = [{"title": t, "text": x} for t, x in collected.items()]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    return len(docs)

def main():
    os.makedirs(_DATA_DIR, exist_ok=True)  # data/ 폴더 없으면 생성

    collected = {}
    failed = []

    print(f"시드 {len(SEED_TERMS)}개 수집 시작...")
    seed_pages = []
    for term in SEED_TERMS:
        page = fetch_page(term)
        if page is None:
            failed.append(term)
            print(f"  [실패] {term} (문서 없음 또는 너무 짧음)")
            continue
        collected[page.title] = page.text
        seed_pages.append(page)
        print(f"  [수집] {page.title} ({len(page.text)}자)")

        save_docs(collected)
    print(f"\n[중간 저장] 시드 수집분 {len(collected)}개 저장 완료")

    print(f"\n링크 확장 수집 시작 (문서당 최대 {MAX_LINKS_PER_PAGE}개)...")
    for page in seed_pages:
        links = [t for t in page.links.keys() if is_valid_title(t)]
        for link_title in links[:MAX_LINKS_PER_PAGE]:
            if link_title in collected:
                continue
            linked = fetch_page(link_title)
            if linked is None:
                continue
            collected[linked.title] = linked.text
        save_docs(collected)
        print(f"  [확장 완료] {page.title} -> 누적 {len(collected)}개 (저장됨)")

    docs = [{"title": t, "text": x} for t, x in collected.items()]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(docs)}개 문서 저장 완료 -> {OUTPUT_FILE}")
    if failed:
        print(f"시드 중 실패: {failed}")


if __name__ == "__main__":
    main()