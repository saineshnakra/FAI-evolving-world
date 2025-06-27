# python version : 3.11.5

start:
	python -m venv .venv && \
	source .venv/bin/activate && \
	pip install -r requirements.txt && \
	python simulator.py

