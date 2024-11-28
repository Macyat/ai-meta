install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

install-Azure:
	pip install --upgrade pip &&\
		pip install -r requirements-Azure.txt

format:
	black *.py
	
test:
	python -m pytest -vv test_hello.py

lint:
	pylint --disable=R,C  hello.py
	
all: install lint test
