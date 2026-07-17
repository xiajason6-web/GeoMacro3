# GeoMacro3 — Iran Escalation Pricing Model

OSINT-driven escalation state model (**P**) for the 2026 US–Iran war vs
market-implied scenario pricing (**Q**); alpha defined as the P−Q divergence,
expressed through oil and oil-adjacent instruments. Mearsheimer's structural
argument enters as overrulable Bayesian priors on the regime transition matrix.

**Research framework — not financial advice.**

## Live dashboard

Streamlit entrypoint: `streamlit_app.py` (repo root). It pulls all keyless
sources live on cold start (IMF PortWatch, Polymarket, yfinance) and runs the
full model chain — no secrets required. Deploy on Streamlit Community Cloud by
pointing it at this repo.

```bash
# local
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py
```

## Daily research pipeline (local)

```bash
make refresh   # ingest -> headlines -> LLM event coder -> models -> brief
make brief     # daily brief: state estimate, P vs Q, signals, what changed
make signals   # A1-A6 with rationale and caveats
make stress    # peace-shock stress on the stylized book
```

The LLM event coder needs `ANTHROPIC_API_KEY` in `.env` (see `.env.example`).
Everything else is keyless.

## Architecture

- `config/` — S0–S5 taxonomy, Mearsheimer Dirichlet priors, frozen coded
  event history (Feb–Jul 2026)
- `src/ingest/` — PortWatch, Polymarket, yfinance, GDELT, Google News RSS
- `src/coding/` — LLM event/rhetoric coder with versioned frozen prompts
- `src/market_implied/` — futures curve, Breeden–Litzenberger RND,
  prediction-market panel, fingerprint inversion (**Q**)
- `src/model/` — regime Markov (conjugate Dirichlet–multinomial), event
  studies, Hawkes intensity (**P**)
- `src/alpha/` — signals A1–A6, falsification backtest, peace-shock stress
- `data/` — vintaged parquet lake (gitignored; point-in-time discipline)

See `CLAUDE.md` for conventions and the standing weaknesses — start with
"n = 1 war."
