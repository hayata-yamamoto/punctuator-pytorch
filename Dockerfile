FROM python:3.7-slim

WORKDIR punctuator/

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
    libyaml-dev

COPY ./ ./

RUN pip3 install pipenv \
  && pipenv sync --dev \
  && pipenv run develop \
  && rm -rf /var/lib/apt/lists/*


CMD ["/bin/bash", "-c", "echo 'python container"]

