FROM python:3.7-slim

WORKDIR /var/www

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
  && rm -rf /tmp/* /var/tmp/* \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /var/lib/apt/lists/*

COPY poetry.lock poetry.lock
COPY pyproject.toml pyproject.toml

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 \
  && export PATH=/root/.poetry/bin:$PATH \
  && poetry self update \
  && poetry config create.virtualenvs false \
  && poetry install --no-dev

ENV PATH /root/.poetry/bin:$PATH
COPY . /var/www

CMD ["poetry run uvicorn api.main:app --port 8000"]