# Company Leadership RAG Agent — common tasks.
#
# Usage:
#   make install                         # create venv + install dependencies
#   make collect URL=https://robinhood.com/   # scrape leadership into SQLite
#   make chat                            # launch the Streamlit chat UI
#   make test                            # run the test suite

VENV := venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

# Default company used by `make collect` if no URL is given.
URL ?= https://meetcampfire.com/

.PHONY: install collect chat test

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

collect:
	$(PY) -m src.scraper "$(URL)"

chat:
	$(VENV)/bin/streamlit run src/app.py --server.headless true

test:
	$(PY) -m pytest -q