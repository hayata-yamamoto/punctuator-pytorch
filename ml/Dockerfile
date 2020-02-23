FROM python:3.7-slim

WORKDIR punctuator-pytorch/

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

COPY .. ./

CMD ["/bin/bash", "-c", "pip3 install pipenv && pipenv sync --dev && pipenv run python punctuator/tasks/create_dataset.py && pipenv run python punctuator/tasks/train.py"]

