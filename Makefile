.PHONY: check test lint report

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

check: lint test

report:
	.venv/bin/python scripts/build_results_svg.py
