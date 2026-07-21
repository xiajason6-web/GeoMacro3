# GeoMacro3 — Iran Escalation Pricing Model

**A structural conflict-forecasting model reconciled against market prices, with
a public, auto-graded track record.** Two probability distributions over the
same war — **P** (an OSINT-driven escalation state model) and **Q** (what oil
futures, options, and prediction markets imply) — with the divergence
decomposed by mechanism. Built solo with AI leverage; live since July 2026.

**Research framework — not financial advice.**

## The track record (the part that matters)

[`calls/ledger.yaml`](calls/ledger.yaml) — append-only, git-timestamped,
auto-graded daily by CI, Brier-scored. Every call carries explicit
Polymarket-grade resolution criteria checkable from this repo's own data lake.
Current calls include the model's Hormuz-normalization odds quoted directly
against same-day Polymarket prices (model 20/33/63% vs market 14.5/24.5/54.5%
for Aug/Sep/Dec 2026). The daily brief and commit log are the audit trail.

## Headline findings so far

- **The market capitulated to near-term persistence faster than the model**
  (the July-31 normalization market went 45% → 2% in days) — the naive "market
  underprices the war" thesis was dead on arrival; the remaining divergences
  are in the curve back-end, the horizontal (Gulf-infrastructure) axis, and
  novel-event convexity.
- **Of Brent ~$82, ~$26 is war premium** (fundamentals control: pre-war OLS on
  copper/USD/rates/S&P); the futures curve keeps only ~54% of it at 12M.
- **Horizontal Gulf-infrastructure risk is priced in European gas (TTF +57%
  vs pre-war, Henry Hub flat), not in oil** — visible only cross-market.
- **All-out-war excursions are likely; sustained all-out war is not**
  (P(visit S4 in 3m) ≈ 32% vs S4 occupancy ≈ 3%) — matching the war's two
  observed one-week S4 excursions.
- Habituation — "the market goes numb to repeated strikes" — is **not
  supported** in this war's data; novel rungs still reprice (+16% vs +10%
  abnormal 20d Brent).

## Method, in one paragraph

Weekly regime classification (S0 lull → S5 deal) from IMF PortWatch transits +
an LLM-coded event stream (versioned frozen prompts; inter-coder agreement 82%
on rungs). Transitions follow a conjugate Dirichlet–multinomial posterior with
**three voices**: a Mearsheimer structural prior (whose weight is *derived
live* from a scorecard grading his testable claims against the data), six
analog conflicts coded to the same taxonomy (Tanker War → the 2025 12-day
war), and the live war. Endurance covariates (munitions production asymmetry,
economic runway, horizontal spread) modulate specific transition cells with
effect sizes calibrated by event study where evidence exists. Every judgment
knob is registered, tagged, and adversarially swept in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md) — all headline conclusions survive ±50%
perturbation.

## Live dashboard

Streamlit entrypoint: `streamlit_app.py` — pulls all keyless sources on cold
start, no secrets required.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py     # local dashboard
make refresh                                  # full daily pipeline + brief + graded calls
```

CI runs the pipeline daily (13:30 UTC) and commits the brief + graded ledger
back to this repo. Optional secrets: `ANTHROPIC_API_KEY` (live event coder),
`EIA_API_KEY` (inventories driver).

## Honest limitations

n = 1 war (~24 weekly transitions — the posterior is mostly prior and analogs,
and prints its own mass decomposition); Q from free/delayed data (ETF options,
not CME; trust changes over levels); event-calls in the ledger are self-graded
via the LLM coder against frozen criteria; A1/A4 signals unvalidated until
daily vintages accumulate. Full register: [`ASSUMPTIONS.md`](ASSUMPTIONS.md);
conventions and build history: [`CLAUDE.md`](CLAUDE.md); design rationale:
[`docs/DESIGN.md`](docs/DESIGN.md).
