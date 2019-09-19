#!bin/bash

python -m venv venv
source venv/bin/activate
python setup.py develop
python punctuator/task/allennlp_tagger.py