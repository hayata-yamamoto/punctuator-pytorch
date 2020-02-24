FROM python:3.7-slim

WORKDIR app
COPY poetry.lock poetry.lock
COPY pyproject.toml pyproject.toml

RUN set -x \
  && apt-get update -yqq \
  && apt-get upgrade -yqq \
  && apt-get install -yqq --no-install-recommends \
    git \
    curl \
    build-essential \
    python-dev \
    apt-utils \
    libsndfile-dev \
    gcc \
    cmake \
    libyaml-dev \
  && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python \
  && export PATH=$HOME/.poetry/bin:$PATH \
  && poetry self update \
  && poetry config virtualenvs.in-project true \
  && poetry install

COPY . app

CMD ["uvicorn api.main:app --port 8000"]