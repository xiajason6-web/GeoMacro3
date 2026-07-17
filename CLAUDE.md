# Iran Escalation Pricing Model

OSINT-driven escalation state model for the US–Iran war (P), extraction of what
markets price (Q), and alpha defined as the P−Q divergence, expressed through oil
and oil-adjacent instruments. **Research framework — not financial advice.**

Full methodology lives in the design doc (§1–§10). This file is the operating
manual: conventions, run commands, and the standing weaknesses the implementer
must keep in view.

## Status
- Built to **milestone M1** (ingest trio + A4 basis monitor). M2–M7 not started.
- As of 2026-07-17: sixth day of US strikes; Hormuz re-blockaded; Brent ~$85–86.
  Current regime ≈ **S2** (maritime/chokepoint war) with S3 events occurring.

## Environment
- Python 3.9, isolated venv at `.venv/` (system Python left untouched).
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- No API keys needed for M1 (PortWatch/Polymarket/yfinance are all keyless).
- Secrets, when M2+ adds them, go in `.env` (gitignored) and are referenced by
  env-var NAME only in `config/sources.yaml` — never inline a key.

## Run commands
- `make refresh`    — land all three sources + print the basis monitor
- `make ingest`     — just the three ingests
- `make monitor`    — recompute the A4 basis monitor from latest vintages
- `make test`       — timestamp-discipline tests
- Individual: `.venv/bin/python -m src.ingest.{portwatch,polymarket,prices}`

## Repo layout
- `config/` — `sources.yaml` (endpoints/tickers), `taxonomy.yaml` (S0–S5),
  `priors.yaml` (the Mearsheimer transition priors, consumed by M4).
- `src/ingest/` — one module per source; each writes a dated parquet vintage.
- `src/report/basis_monitor.py` — the A4 three-way Hormuz read (M1 deliverable).
- `src/{coding,features,market_implied,model,alpha}/` — M2–M6 homes (empty).
- `data/` — parquet lake, gitignored, partitioned `<source>/ingest_date=YYYY-MM-DD/`.

## The one non-negotiable convention: vintage / point-in-time discipline
Partitions are keyed by the date **we pulled** the data (`ingest_date=`), and
every row carries `ingested_at`. Never key by observation date; never backfill a
past vintage with data that was not available then. The backtest (M6) must never
see a value before the day it could have been pulled. `src/common.write_partition`
enforces the stamp; `tests/test_timestamps.py` guards it. PortWatch lags ~5 days —
that lag must stay visible, not smoothed away.

## Standing weaknesses (design doc §9 — keep these in code comments and reports)
1. **n = 1 war, ~20 weeks.** Every statistical claim is fragile. Priors and the
   2019/2020/2024/2025 analog windows do real work. Outputs are decision-support,
   not a signal factory. Prefer robustness across specifications over significance.
2. **Q is partly unobservable on free data.** ETF/delayed options are a blurry
   lens on the true crude vol surface. Flag levels as approximate; trust changes.
3. **Prediction-market thinness.** $50k-volume markets move for less than a real
   position. Treat quotes as information, never executable Q. Weight by volume.
4. **P and Q share PortWatch.** It is the S2 input to P *and* Polymarket's
   resolution source, and it lags ~5d. Always decompose a Hormuz "divergence"
   into mechanical-lag vs. real-belief-gap (see `basis_monitor.decompose`) before
   calling it alpha.
5. **The gold anomaly (−28% since war start).** Textbook war playbooks fail —
   fit the §4.3 fingerprint matrix from data, not theory.
6. **LLM coder drift (M2).** Version prompts, freeze coded history, re-code only
   via explicit migration.
7. **The Tanker War lesson (1984–88).** Escalation premium can bleed even while
   the ladder holds if fundamentals (OPEC spare capacity, demand) are loose. Carry
   EIA balances as fundamentals controls, not just conflict features.
8. **Mearsheimer could be wrong.** His argument enters as an *overrulable* prior
   (`priors.yaml`, `prior_strength`). A5 (deal hedge) is mandatory, not optional.

## Milestones
- **M1 ✅** ingest trio + A4 monitor.
- M2 LLM event/rhetoric coder (+ backfill, 100-event QA).
- M3 Q extraction — curve + prediction-market panel first; RND/Breeden–Litzenberger
  **deferred to last** (weakest free data, highest code cost).
- M4 regime Markov model with `priors.yaml`.
- M5 event studies, habituation curve, Hawkes intensity.
- M6 signals A1–A6, walk-forward backtest (as *falsification*, not performance),
  peace-shock stress.
- M7 daily brief: state estimate, P vs Q table, live signals, what changed.

## Verified data endpoints (2026-07-17)
- PortWatch daily chokepoints: ArcGIS `Daily_Chokepoints_Data/FeatureServer/0`,
  filter `portname='Strait of Hormuz'`, dates are epoch-ms.
- Polymarket Gamma search: `gamma-api.polymarket.com/public-search?q=<term>`;
  markets nested under `events[].markets[]`; `outcomes`/`outcomePrices` are
  JSON-encoded strings; store `description` (resolution criteria) verbatim.
- Prices: yfinance. Curve **ladder not available free** — front-month only in M1.
