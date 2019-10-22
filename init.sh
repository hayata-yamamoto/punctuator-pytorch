#!bin/bash

python3 -m venv venv
source venv/bin/activate
python3 setup.py develop
python3 punctuator/tasks/allennlp_tagger.py --embed 100 --hidden 100 --epoch 2
