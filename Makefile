format:
	yapf -r -i --style pep8 punctuator

format-check:
	yapf -r -d --style pep8 punctuator

import:
	isort -rc -i punctuator

import-check:
	isort -rc -d -c punctuator

lint:
	flake8 punctuator