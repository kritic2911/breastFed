.PHONY: all test smoke run lint format clean

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
RUFF   ?= .venv/bin/ruff

all: test smoke

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) scripts/run_smoke.py

run:
	$(PYTHON) scripts/run_federated.py --config configs/experiment.yaml --seed 42

lint:
	$(RUFF) check src/ tests/ scripts/

format:
	$(RUFF) format src/ tests/ scripts/

clean:
	rm -rf .pytest_cache .ruff_cache synthetic_data/ results/smoke/

