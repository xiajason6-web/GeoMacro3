PY = .venv/bin/python

.PHONY: refresh ingest portwatch polymarket prices monitor test

# Full M1: land all three sources, then print the basis monitor.
refresh: ingest monitor

ingest: portwatch polymarket prices

portwatch:
	$(PY) -m src.ingest.portwatch

polymarket:
	$(PY) -m src.ingest.polymarket

prices:
	$(PY) -m src.ingest.prices

monitor:
	$(PY) -m src.report.basis_monitor

test:
	$(PY) -m pytest tests/ -q
