.PHONY: setup lint test ui notebook doctor

setup:
	./scripts/mac_setup.sh

lint:
	ruff check .

test:
	pytest -q

ui:
	qwenfree ui

notebook:
	python scripts/make_notebook.py

doctor:
	qwenfree doctor
