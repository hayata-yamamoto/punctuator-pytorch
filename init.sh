#!bin/bash

python -m venv venv \
&& source venv/bin/activate \
&& python setup.py develop
&& python punctuator/task/allennlp_tagger.py --embed 100 --hidden 100 --epoch 100