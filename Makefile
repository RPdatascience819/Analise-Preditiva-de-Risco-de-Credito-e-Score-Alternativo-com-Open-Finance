.PHONY: install notebook dashboard test

install:
	pip install -r requirements.txt

notebook:
	jupyter notebook

dashboard:
	streamlit run dashboard/app.py

dataset:
	python -m src.build_dataset

test:
	python -m unittest discover -s tests
