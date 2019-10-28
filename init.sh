#!bin/bash

python3 -m venv venv
source venv/bin/activate
python3 setup.py develop
python3 punctuator/tasks/allennlp_tagger.py
python3 punctuator/tasks/validate.py
