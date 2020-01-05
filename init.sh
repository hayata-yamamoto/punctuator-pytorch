# /bin/bash

sudo apt install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    git \
&& git clone https://github.com/pyenv/pyenv.git ~/.pyenv \
&& echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile \
&& echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile \
&& echo 'eval "$(pyenv init -)"' >> ~/.bash_profile \
&& source ~/.bash_profile \
&& pyenv install 3.7.5 \
&& pyenv local 3.7.5 \
&& pip3 install pipenv \
&& pipenv sync --dev \
&& pipenv develop \
&& pipenv run develop \ 
&& mv credentials/.env-sample credentials/.env

