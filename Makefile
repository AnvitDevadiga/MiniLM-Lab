.PHONY: check test lint report assets

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

check: lint test

report: assets

assets:
	.venv/bin/python scripts/build_report_assets.py
