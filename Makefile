PY = .venv/bin/python

.PHONY: refresh ingest models brief monitor signals backtest stress test \
        portwatch polymarket prices gdelt curve rnd predmkt fingerprint \
        regime events habituation hawkes backfill

# Daily driver: land fresh data, code new events, rebuild derived layers, brief.
refresh: ingest code models brief calls

# --- ingest layer ---------------------------------------------------------
ingest: portwatch polymarket prices curve rnd headlines

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

# Headlines for the coder: Google News RSS is the reliable primary; GDELT is
# preferred when its rate limiter cooperates (richer metadata) but must never
# break the refresh — both write to the same gdelt_articles source.
headlines: newsrss gdelt

newsrss:
	-$(PY) -m src.ingest.newsrss

gdelt:
	-$(PY) -m src.ingest.gdelt

# --- coding layer ---------------------------------------------------------
# Live LLM coder (needs ANTHROPIC_API_KEY in .env, or claude CLI). Failure
# tolerated: the day's events just go uncoded until the next run.
code:
	-$(PY) -m src.coding.llm_coder

# --- derived layers -------------------------------------------------------
models: predmkt regime events habituation hawkes fingerprint spread munitions economic will scorecard fundamentals premia equities

equities:
	$(PY) -m src.features.equities

premia:
	$(PY) -m src.features.premia

fundamentals:
	$(PY) -m src.features.fundamentals

scorecard:
	$(PY) -m src.model.scorecard

will:
	$(PY) -m src.features.will

economic:
	$(PY) -m src.features.economic

spread:
	$(PY) -m src.features.horizontal_spread

munitions:
	$(PY) -m src.features.munitions

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
calls:
	$(PY) -m src.report.calls

brief:
	$(PY) -m src.report.daily_brief

monitor:
	$(PY) -m src.report.basis_monitor

signals:
	$(PY) -m src.alpha.signals

replay:
	$(PY) -m src.alpha.replay

sensitivity:
	$(PY) -m src.alpha.sensitivity

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
