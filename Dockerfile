# [Dockerfile]
#
# rain-research-bot을 컨테이너로 패키징한다.
# 핵심 이슈: konlpy의 Mecab은 시스템에 mecab-ko + mecab-ko-dic이 설치되어 있어야 한다.
#           로컬(맥)에서는 Homebrew로 깔았지만, 컨테이너(리눅스)에서는 소스 빌드가 필요하다.

FROM python:3.11-slim

# ---- uv 설치 (공식 이미지에서 실행 파일만 가져온다) ----
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

# ---- 시스템 패키지 + mecab-ko 빌드 도구 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    automake \
    autoconf \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# ---- mecab-ko 설치 (형태소 분석 엔진 본체) ----
# tar 압축 해제 후 파일 시각이 뒤섞이면 빌드 시스템이 "다시 생성해야 한다"고
# 착각해서 옛날 도구(aclocal 등)를 찾다가 실패한다. touch로 시각을 통일해서 방지.
RUN curl -L https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz -o /tmp/mecab.tar.gz \
    && tar zxf /tmp/mecab.tar.gz -C /tmp \
    && cd /tmp/mecab-0.996-ko-0.9.2 \
    && find . -exec touch -d "2020-01-01" {} + \
    && ./configure --enable-utf8-only --build=x86_64-linux-gnu \
    && make \
    && make install \
    && ldconfig \
    && rm -rf /tmp/mecab*

# ---- mecab-ko-dic 설치 (한국어 사전) ----
RUN curl -L https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz -o /tmp/mecab-dic.tar.gz \
    && tar zxf /tmp/mecab-dic.tar.gz -C /tmp \
    && cd /tmp/mecab-ko-dic-2.1.1-20180720 \
    && find . -exec touch -d "2020-01-01" {} + \
    && ACLOCAL=true AUTOMAKE=true AUTOCONF=true AUTOHEADER=true ./configure --build=x86_64-linux-gnu \
    && make \
    && make install \
    && rm -rf /tmp/mecab-dic*

# 리눅스에서 mecab-ko-dic이 설치되는 기본 경로 (컨테이너 안에서는 이 경로를 씀)
ENV MECAB_DICPATH=/usr/local/lib/mecab/dic/mecab-ko-dic

WORKDIR /app

# ---- 파이썬 의존성 설치 (코드보다 먼저 복사해서 캐시 활용) ----
# uv.lock에 고정된 버전 그대로 설치한다. CPU 전용 torch 설정도 pyproject.toml에 있다.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# uv가 만든 가상환경을 기본 파이썬으로 쓴다
ENV PATH="/app/.venv/bin:$PATH"

# ---- 애플리케이션 코드 + 데이터 복사 ----
COPY src/ ./src/
COPY static/ ./static/
COPY data/ ./data/

EXPOSE 8003

CMD ["python3", "src/api.py"]