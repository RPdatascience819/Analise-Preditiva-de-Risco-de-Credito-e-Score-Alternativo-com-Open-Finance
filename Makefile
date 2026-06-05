.PHONY: install notebook dashboard test

install:
	pip install -r requirements.txt

notebook:
	jupyter notebook

dashboard:
	streamlit run dashboard/app.py

test:
	python -m unittest discover -s tests
