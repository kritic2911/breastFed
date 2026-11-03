PYTHON ?= .venv/bin/python3
PYTEST ?= .venv/bin/pytest
RUFF   ?= .venv/bin/ruff

.PHONY: all test lint format dummy clean cache qc probe install

all: lint test

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	$(PYTEST) tests/ -v

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

cache:
	$(PYTHON) scripts/build_manifest_and_cache.py

qc:
	$(PYTHON) scripts/generate_p1_qc.py

probe:
	$(PYTHON) scripts/site_id_probe.py

dummy:
	$(PYTHON) scripts/dummy_run.py --epochs 2

clean:
	rm -rf .pytest_cache .ruff_cache results/runs/ results/ledger.csv
