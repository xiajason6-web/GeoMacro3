# Iran Escalation Pricing Model

OSINT-driven escalation state model for the US–Iran war (P), extraction of what
markets price (Q), and alpha defined as the P−Q divergence, expressed through oil
and oil-adjacent instruments. **Research framework — not financial advice.**

Full methodology lives in the design doc (§1–§10). This file is the operating
manual: conventions, run commands, and the standing weaknesses the implementer
must keep in view.

## Status
- **All milestones M1–M7 built and running** (2026-07-17). Daily driver:
  `make refresh` (ingest -> models -> brief). See Milestones below for what
  each layer does and its known limits.
- As of 2026-07-17: seventh consecutive night of US strikes; Hormuz re-closed
  (7dMA 16.4 calls/day = 22% of baseline); dual blockade reinstated; Brent ~$88.
  Current regime **S2 with S3 events** (Kuwaiti desalination station hit Jul 17).
- War chronology correction vs the design doc (from Wikipedia, coded into the
  backfill): war opened 2026-02-28 at near-S4 intensity (Khamenei assassinated);
  Hormuz closed ~Mar 6; the Apr 8 "ceasefire" was a ceasefire-plus-DUAL-BLOCKADE
  (transits never recovered); Islamabad Memorandum signed Jun 14-17 produced only
  a partial reopening (peak 41% of baseline); MOU collapsed Jul 8.

## Environment
- Python 3.9, isolated venv at `.venv/` (system Python left untouched).
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- No API keys needed for M1 (PortWatch/Polymarket/yfinance are all keyless).
- Secrets, when M2+ adds them, go in `.env` (gitignored) and are referenced by
  env-var NAME only in `config/sources.yaml` — never inline a key.

## Run commands
- `make refresh`    — full daily pipeline: ingest -> derived models -> brief
- `make ingest`     — data layer only (portwatch, polymarket, prices, curve, rnd, gdelt)
- `make models`     — derived layer (predmkt, regime, events, habituation, hawkes, fingerprint)
- `make brief`      — daily brief (state, P vs Q, signals, what-changed diff);
                      snapshots to data/briefs/
- `make monitor`    — A4 basis monitor; `make signals` — A1-A6 with rationale
- `make backtest`   — falsification harness; `make stress` — peace-shock stress
- `make backfill`   — reload frozen Feb-Jul coded history into the lake
- `make test`       — timestamp-discipline tests

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

## Milestones (all built 2026-07-17)
- **M1 ✅** ingest trio + A4 monitor.
- **M2 ✅ (live coder ONLINE)** — ANTHROPIC_API_KEY is in `.env` (gitignored,
  auto-loaded by `src/common.py`); the coder runs claude-sonnet-4-6 in the
  daily chain (`make code`). The Feb–Jul backfill (51 events, 18 rhetoric rows)
  was coded in-session by Claude Opus 4.8 from the Wikipedia chronology and is
  FROZEN in `config/coded_events_backfill.yaml` — canonical through 2026-07-17;
  live codings append strictly after that date (merge logic in llm_coder.main;
  raw live output audited in `coded_events_live_raw`).
  **Prompt migration precedent:** QA of the first live run vs the frozen
  backfill caught S4 over-assignment and escalatory-diplomacy-coded-as-S5 →
  rung_mapper_v2 (v1 stays frozen). v2 verified: spurious S4s went 8 → 0.
  Headlines: Google News RSS (`make newsrss`) is the reliable primary; GDELT's
  DOC API rate-limits brutally (1 req/5s, sticky IP penalties) and is kept as a
  failure-tolerant secondary.
- **M3 ✅** Q extraction: real futures month-ladder (BZ/CL contracts DO exist on
  free yfinance — steep backwardation), USO/BNO Breeden–Litzenberger RND,
  predmkt panel with the full normalization CDF, ridge-anchored fingerprint
  inversion (soft cross-check only; buckets are collinear).
- **M4 ✅** soft-label weekly classifier + conjugate Dirichlet–multinomial
  transition posterior (closed form, no PyMC — Python 3.9 and n=20 weeks both
  argue for it). Posterior currently **79% prior / 21% data** — printed on
  every run; do not present P as data-driven.
- **M5 ✅** event studies, habituation, Hawkes. Empirical verdicts on the doc's
  hypotheses: novelty drift SUPPORTED (+16% vs +10% at 20d, overlapping-window
  caveat); habituation WEAK (half-life ~31 events — A2 held FLAT); Hawkes
  branching 0.89 near-critical but with an uninformatively wide profile range.
- **M6 ✅** signals A1–A6 + falsification backtest + peace-shock stress. A3 and
  A6 survive their perturbation grids; survival earns MONITORING, not capital.
  Surprise finding: A1 currently fires SHORT — the market's normalization CDF
  (24% by Sep 30) is already MORE pessimistic than the prior-driven model at 3m.
  Caveat: the A1 comparison maps "not in S2+" to "normalized", which overstates
  model-implied normalization (S1 can still have transits < 60) — refine before
  trusting the direction.
- **M7 ✅** `make brief` / `make refresh`. Briefs snapshot to `data/briefs/` and
  diff headline numbers against the prior brief.

## Endurance layer (post-M7, the "fuel gauge")
The model measures escalation (state, intensity) but Mearsheimer's thesis is
about *endurance* (can each side sustain). The endurance layer adds that gauge;
its indicators double as transition covariates AND as the live scorecard that
should *derive* the prior-strength slider.
- **8c horizontal-spread index ✅** (`src/features/horizontal_spread.py`,
  `make spread`): weekly third-party fronts (GCC/Iraq/Yemen targets, maritime
  excluded), proxy activation, S3 share, and an S3-attractor test
  (P(S3 recurs next week | S3 now) ≈ 62% — direct support for the v2 prior).
  Serves three jobs: the S3 alpha tracker (unpriced axis), the first endurance
  covariate (fronts multiply depletion), and the scorecard's spread input.
  As of 2026-07-20: trailing-4wk spread ~2× the war average — war re-widening
  after the June-deal contraction; Jul 13 is the peak week.
- **Coder merge fix (2026-07-20):** the live coder now ENRICHES the trailing 21
  days with net-new events (frozen-precedence) instead of dropping everything
  inside the backfill's final week — recent weeks no longer go dark. Frozen
  backfill stays the authoritative spine; live never rewrites deep history.
- **8a munitions/interceptor sustainability ✅** (`src/features/munitions.py`,
  `config/munitions.yaml`, `make munitions`): rule-based extractor over coded
  event TEXT → munitions ledger → **cost-exchange ratio** (defender $ to
  intercept per $1 of Iranian offense, currently ~6.6:1 — Mearsheimer's
  asymmetric escalation as a dollar figure) + a scenario **interceptor runway**
  (grind-vs-S4-breakout discriminator). Honest v1 limits: it's a FLOOR (only
  counts munitions the summaries mention; recent salvo counts are undercounted,
  so the runway is unreliable — the RATIO is the robust output), and the
  inventory band is a scenario, not intelligence. Upgrade paths: LLM munitions
  sub-coder, think-tank depletion estimates.
- **M9 covariate wiring ✅ (thin slice)** (`src/model/covariates.py`): 8a and 8c
  now MOVE P. Bounded, conviction-tuned hazard multipliers (NOT a regression
  fit — pure noise at n=23) on the posterior pseudo-counts: 8a → **S4 gate**
  (suppress `*→S4`/`S4→S4`, boost `S4→S2/S3`), 8c → **S3 pump** (boost `*→S3`).
  `run(use_covariates=True)` is the live default; `False` = static baseline.
  Effect at strength 1: 3m S3 +8pp, S4 −1pp, mean weeks in S2+ 6.5→7.9.
  Dashboard/brief show the static-vs-adjusted delta. The dashed arrows in the
  architecture diagram are now solid for 8a/8c.
- **8b economic endurance ✅** (`src/features/economic.py`, `config/economic.yaml`,
  `make economic`): US oil-pain (Brent vs $100–150 band, a PRICE story) + Iran
  fiscal runway (PortWatch export-loss vs FX buffer, a VOLUME story). Wired into
  the **S5-drift** cell (`covariates.py`, coef 0.4 — deliberately small; face-lock
  keeps deals unstable under strain). Current read: Brent $88 → US-pain 0.00,
  Iran runway ~268d → Iran-pain 0.00, so **p_b≈0**: no deal pull right now — the
  war widens with nothing pushing back, Mearsheimer's endurance asymmetry
  quantified. FREE-DATA PROXY: does NOT yet separate war premium from soft demand
  (the fundamentals-control upgrade needs EIA balances + FRED keys).
- Next: 8d will/casualties (**scorecard-primary** — too soft for a P covariate,
  per Mearsheimer's materiel>politics caveat), then **M10 Mearsheimer scorecard →
  dynamic slider** — the capstone that DERIVES prior_strength from live evidence
  (each restriction graded by a layer; fraction confirmed → strength). M9 now
  wires 8a+8c+8b; the endurance "fuel gauge" moves P.

## What accumulates value from here
Every `make refresh` lands a new dated vintage. A1/A4 backtesting is impossible
until a history of daily P and Q vintages exists — the system starts earning
its keep after ~2-3 weeks of daily runs. The live LLM coder replaces the frozen
backfill path as soon as a backend key is available.

## Verified data endpoints (2026-07-17)
- PortWatch daily chokepoints: ArcGIS `Daily_Chokepoints_Data/FeatureServer/0`,
  filter `portname='Strait of Hormuz'`, dates are epoch-ms.
- Polymarket Gamma search: `gamma-api.polymarket.com/public-search?q=<term>`;
  markets nested under `events[].markets[]`; `outcomes`/`outcomePrices` are
  JSON-encoded strings; store `description` (resolution criteria) verbatim.
- Prices: yfinance. Curve **ladder not available free** — front-month only in M1.
