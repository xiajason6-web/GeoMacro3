PY = .venv/bin/python

.PHONY: refresh ingest models brief monitor signals backtest stress test \
        portwatch polymarket prices gdelt curve rnd predmkt fingerprint \
        regime events habituation hawkes backfill

# Daily driver: land fresh data, rebuild derived layers, emit the brief.
refresh: ingest models brief

# --- ingest layer ---------------------------------------------------------
ingest: portwatch polymarket prices curve rnd gdelt

portwatch:
	$(PY) -m src.ingest.portwatch

polymarket:
	$(PY) -m src.ingest.polymarket

prices:
	$(PY) -m src.ingest.prices

curve:
	$(PY) -m src.market_implied.curve

rnd:
	$(PY) -m src.market_implied.rnd

# GDELT is rate-limited (1 req/5s, aggressive IP penalties) — failures must
# not break the refresh; the coder just skips a day.
gdelt:
	-$(PY) -m src.ingest.gdelt

# --- derived layers -------------------------------------------------------
models: predmkt regime events habituation hawkes fingerprint

predmkt:
	$(PY) -m src.market_implied.predmkt

regime:
	$(PY) -m src.model.regime_markov

events:
	$(PY) -m src.model.event_study

habituation:
	$(PY) -m src.features.habituation

hawkes:
	$(PY) -m src.model.hawkes

fingerprint:
	$(PY) -m src.market_implied.fingerprint

# --- reports & research ---------------------------------------------------
brief:
	$(PY) -m src.report.daily_brief

monitor:
	$(PY) -m src.report.basis_monitor

signals:
	$(PY) -m src.alpha.signals

backtest:
	$(PY) -m src.alpha.backtest

stress:
	$(PY) -m src.alpha.stress

# --- maintenance ----------------------------------------------------------
# Reload the frozen Feb-Jul 2026 coded history into the lake (idempotent).
backfill:
	$(PY) -m src.coding.load_backfill

test:
	$(PY) -m pytest tests/ -q
