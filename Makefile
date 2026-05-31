# Company Leadership RAG Agent — common tasks.
#
# Usage:
#   make install                              # create venv + install dependencies
#   make collect URL=https://robinhood.com/   # scrape leadership into SQLite
#   make collect DOMAIN=robinhood.com         # same, by bare domain
#   make chat                                 # launch the Streamlit chat UI
#   make test                                 # run the test suite

VENV := venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

# Target company for `make collect`. Accept either a full URL or a bare DOMAIN
# (the scraper normalizes both); DOMAIN takes precedence when provided.
URL    ?= https://meetcampfire.com/
DOMAIN ?=
TARGET := $(if $(DOMAIN),$(DOMAIN),$(URL))

.PHONY: install collect chat test

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

collect:
	$(PY) -m src.scraper "$(TARGET)"

chat:
	$(VENV)/bin/streamlit run src/app.py --server.headless true

test:
	$(PY) -m pytest -q