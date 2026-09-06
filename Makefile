.PHONY: install test analyze figures quick full lint

install:
	pip install -r requirements.txt && pip install -e .

test:
	python -m pytest tests/ -q

analyze:
	python scripts/analyze_results.py --sweep results/sweep_results_v3.csv

figures:
	python scripts/make_figures.py --results results --out figures/regenerated

quick:
	python scripts/run_benchmark.py --root ./runs/quick --quick

full:
	python scripts/run_benchmark.py --root ./runs/full

lint:
	ruff check .
