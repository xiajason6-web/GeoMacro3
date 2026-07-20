# Iran Escalation Pricing Model — Design Doc v2

**v2, 2026-07-18.** Supersedes the v1 doc (2026-07-17) on two fronts:
(1) the system is now **built** (M1–M7 + live LLM coder + Streamlit dashboard),
so design intent is replaced by as-built reality and empirical verdicts;
(2) the Mearsheimer prior is updated with his July "horizontal escalation"
argument, encoded as a second frozen prior vintage (`priors.yaml → v2`) that
can be A/B'd against v1 rather than overwriting it.

**Purpose (unchanged):** (1) an OSINT-driven escalation state model for the
US–Iran war, (2) extraction of what markets currently price, (3) alpha defined
as the divergence between the two, expressed through oil and oil-adjacent
instruments.

**Status as of 2026-07-18:** Seventh consecutive night of US strikes. Hormuz
re-closed (7dMA 16.4 calls/day = 22% of baseline); dual blockade reinstated;
Iran hit a Kuwaiti desalination/electricity station on Jul 17 — the first
confirmed strike on Gulf water infrastructure. Brent ~$88 in steep
backwardation (+$8.22 front-to-6M). Polymarket normalization CDF: 2% by Jul 31
→ 12% Aug → 24% Sep → 52% Dec. The July-31 market the v1 doc quoted at ~45%
now prices ~2% — the market capitulated to persistence within days.

*Not financial advice. This is a research framework.*

---

## 1. Thesis and what "alpha" means here

Two probability distributions over the same future:

- **Q (market-implied):** oil futures curve, ETF options RND, prediction
  markets, tanker equities, cross-asset fingerprint.
- **P (model-implied):** the escalation model, disciplined by OSINT data, with
  Mearsheimer's structural argument encoded as **versioned, overrulable**
  Dirichlet priors.

Alpha = E_P[payoff] − E_Q[payoff] per instrument, harvested where the
divergence is largest per unit of carry. Secondary source: short-horizon
reaction dynamics (drift/habituation), nested inside the regime view.

### The Mearsheimer prior — now two frozen vintages

Prior versioning follows the same discipline as the LLM coder prompts: a
vintage is frozen once used; thesis updates create a **new version**, they
never edit the old one. Both live in `config/priors.yaml`; the dashboard and
`regime_markov.run(prior_version=...)` select between them.

**v1 — "Bombing to Lose" (Feb–Mar argument, frozen 2026-07-17).** The four
original restrictions, unchanged: no coercive leverage (low tit-for-tat →
de-escalation hazard); no coherent objective (fat-tailed duration); asymmetric
escalation options (upward mass ≥ resolution mass from S1/S2); face-lock
(high diagonals, lumpy exits). Vertical-ladder framing: S4, once entered, is
very sticky (diagonal 8).

**v2 — "Horizontal escalation" (July interview, frozen 2026-07-18).** Four
changes, each traceable to a specific claim:

| Claim (July) | Encoding change vs v1 |
|---|---|
| The war **widens, not climbs**: Fujairah cut off, Yanbu threatened, continuous strikes on Jordan/Gulf basing. S3 is the war's *resting state*, not a way-station. | S2→S3 up (3→4.5), S3 diagonal up (7→9), S3→S4 halved (2→1) |
| **No escalation dominance; munitions shortages.** The US cannot sustain a long conventional air war; infrastructure threats (bridges, power) are a substitute for depth, not a rung. All-out war is *unsustainable*, decaying back to the grind. | S4 diagonal cut 8→4; S4→S2/S3 mass up (1,2 → 2.5,3); entries into S4 trimmed |
| **Hardliners ascendant; duration is now a strategy** (compounding economic pressure, forcing US desperation), not merely face-lock. | Escalated diagonals up; S1/S2→S5 mass trimmed (2→1.5) |
| **MOU as functional surrender; blockade yields no new leverage.** Deals remain unstable. | S5 row kept non-absorbing; S5→S2 up slightly (2→2.5) |

**What the A/B shows at prior_strength = 1 (23 observed weekly transitions):**

| Metric (3m unless noted) | v1 vertical | v2 horizontal |
|---|---|---|
| P(S2 chokepoint war) | 33% | 36% |
| P(S3 Gulf infra war) | 15% | **23%** |
| P(S4 all-out war) | 11% | **6%** |
| P(S5 deal) | 11% | 9% |
| P(S2+ at 3m) | 60% | 65% |
| P(touch S4 before S5), 1y | 39% | 37% |
| Median weeks to S5 (when reached) | 4 | 6 |
| Mean weeks in S2+ (1y sim) | 4.5 | **6.6** |

Read: v2 makes the war **wider, longer, and less catastrophic** — more
probability on the grinding horizontal states, half the all-out-war tail,
slower resolution. Note what is *robust across both vintages AND the full
prior-strength sweep (0 → 3)*: P(touch S4 before S5) sits in 37–44% and
P(S2+ at 3m) in 56–79% everywhere. The "sticky war, live S4 tail, no clean
exit" conclusion does not depend on which Mearsheimer you buy — only the
*shape* of the persistence does.

**Slider vs matrix (governance rule).** The dashboard's prior-strength slider
scales total prior mass against the data (50/50 crossover ≈ 0.26 at current
sample size); the version radio selects the thesis *shape*. New arguments that
change relative row magnitudes go in the **matrix as a new vintage**; the
slider only expresses confidence in a fixed shape. A conclusion that appears
only at high strength is prior-driven — distrust it.

---

## 2. The escalation state machine

Six ordinal regimes, unchanged definitions (S0 lull, S1 tit-for-tat, S2
maritime/chokepoint, S3 Gulf infrastructure, S4 all-out, S5 de-escalation).
Everything downstream keys off this.

**As-built:** the classifier (`src/features/state_labels.py`) is two-layer —
a PortWatch transit backbone (always available) plus a coded-event layer that
upgrades weeks with S3/S4/S5 evidence. Ambiguous weeks carry **soft
probabilities**, which propagate as fractional transition counts into the
posterior; S5 requires deal evidence AND transits recovering (conjunction).

**The observed traversal (Feb–Jul 2026), from data — richer than v1 assumed:**

S0 (Feb, transits ~100%) → **S4 excursion** (war opens 2026-02-28 at
near-all-out intensity: ~900 strikes/12h, Khamenei assassinated) → S2/S3 blend
(Hormuz collapses ~Mar 6-8, 7dMA 3.6; Qatar LNG and Ras Tanura hit Mar 2) →
second S4 excursion (Mar 17 decapitations) → long S2 grind (the Apr 8
"ceasefire" was a ceasefire-plus-**dual-blockade** — US blockading Iranian
ports, Iran blockading the strait — so transits never recovered) → S5
(Islamabad Memorandum, Jun 14–17; partial reopening peaking at 41% of
baseline) → S1 (brief) → S2 again (MOU dead Jul 8, strait re-closed Jul 12).

**The S3 question, sharpened.** v1 asked whether S3 is "a rung or a
transition." The v2 prior takes a position: **S3 is the attractor** — the
war's resting state under a no-escalation-dominance regime. This is now a
*testable disagreement between prior vintages*: v1 expects S3 weeks to
resolve up (to S4) or back (to S2); v2 expects them to accumulate. The coded
event stream adjudicates over time. Current tape (Kuwait desalination, Bahrain/
Jordan basing strikes, Fujairah disruption) is v2-consistent.

---

## 3. Data layer — as-built, with survival notes

### 3.1 What's live (all free)

| Series | Source | Status & notes |
|---|---|---|
| Hormuz daily transits by vessel class | IMF PortWatch (ArcGIS `Daily_Chokepoints_Data` FeatureServer) | **Keystone.** ~5-day publication lag; also Polymarket's resolution source — see §9 weakness 3 |
| Brent/WTI **full monthly ladder** | yfinance (`BZU26.NYM`-style) | v1 doc assumed front-month only — the ladder EXISTS free. Real curve shape, 15 contracts/root |
| USO/BNO chains → RND | yfinance | Shipped in M3 (v1 planned to defer). IV + OI usable; levels blurry, changes informative |
| ^OVX, tanker equities (FRO/INSW/TNK/TRMD/NAT/STNG), XLE/XOP/OIH, GLD, ITA/PPA | yfinance | Serial download with per-ticker retry (batch endpoint 429s) |
| Polymarket Iran/Hormuz/deal families | Gamma API | Resolution-criteria text stored verbatim with every price |
| Headlines for the coder | **Google News RSS (primary)**; GDELT DOC API (secondary) | GDELT rate-limits brutally (1 req/5s, sticky IP penalties survive backoff) — it must never gate the pipeline |
| Coded events + rhetoric | LLM coder (live) + frozen backfill | See §3.3 |

### 3.2 Deferred (unchanged rationale): ACLED, FIRMS, OpenSky, NOTAMs, EIA/FRED
fundamentals controls (§9 weakness 6 still stands — EIA balances are the top
missing piece), Metaculus. Paid upgrades unchanged.

### 3.3 LLM-coded features — as-built

Live coder: `src/coding/llm_coder.py`, claude-sonnet-4-6 via API key in `.env`
(claude CLI fallback), running daily in `make refresh`.

- **Frozen backfill:** Feb–Jul 2026 history (51 events, 18 rhetoric rows) was
  coded from the Wikipedia war chronology (GDELT's API being blocked at build
  time was a blessing: curated chronology + PortWatch cross-checks beat
  headline soup for backfill). Canonical in `config/coded_events_backfill.yaml`
  through 2026-07-17; live codings **append strictly after**; raw live output
  audited separately (`coded_events_live_raw`).
- **Prompt migration precedent (QA works):** the first live run was QA'd
  against the frozen backfill and caught two systematic biases — S4
  over-assignment to intense-but-conventional strike waves (8 spurious S4s),
  and escalatory diplomacy coded as S5 because prompt v1 said "diplomatic
  events are rung S5" unqualified. Fix: `rung_mapper_v2` (v1 stays frozen).
  Verified: spurious S4s → 0. **Coder bias flows directly into the transition
  posterior — QA new prompt versions against frozen history before trusting.**
- Novelty is computed downstream from the event stream (recurrence count of
  (actor, target_type, rung)), not asked of the LLM.

---

## 4. Extracting Q — as-built

**4.1 RND.** Breeden–Litzenberger on USO/BNO (weighted polynomial smile in
log-moneyness, dense grid, clipped support, renormalized). Reported: P(~Brent
>100), P(>120), P(<75), ATM IV, 15%-moneyness risk reversal. Current reading:
**downside priced heavier than upside** (P(Brent<75) ≈ 16–22% vs P(>100) ≈
5–9% at ~3 weeks) — the options market fears the deal-drop more than
escalation. Consistent with backwardation and with A1's surprise (below).

**4.2 Curve.** Full ladder lands daily. Front-to-6M +$8.22, front-to-12M
+$12.61 — market prices the disruption as temporary. Event-day curve-shape
changes = the market's revealed duration estimate, now trackable with real
contracts rather than proxies.

**4.3 Fingerprint.** Ridge-anchored constrained least squares
(‖A·q−r‖² + λ‖q−q_anchor‖², λ=0.5, anchor = prediction-market vector). The
ridge is **mandatory**: the asset buckets are collinear (all long-oil-beta),
so unregularized inversion is rank-deficient and whipsaws. Payoff matrix A is
filled empirically from the war's own state-labeled weeks. Treat as a soft
cross-check on Polymarket only — never an independent Q. S3 columns currently
lack evidence weeks (soft labels never give S3 the argmax) — a known gap.

**4.4 Prediction markets.** The liquidity worry in v1 was wrong in a good way:
the Hormuz-normalize family runs $5–17M volume per market (not $54k). The
whole family yields a **normalization CDF** (the term structure of expected
reopening), which is the cleanest single Q object in the system. Resolution
criteria stored verbatim; volume attached to every quote.

---

## 5. Building P — as-built

**Model:** conjugate Dirichlet–multinomial on the weekly transition matrix —
closed form, no MCMC. This is a deliberate downgrade from v1's PyMC plan:
states are ~observable (soft labels), the sample is 23 transitions, and the
closed form makes the prior/data blend **auditable** (pseudo-counts vs real
counts, printed on every run: currently 21% data / 79% prior at strength 1).
PyMC would add machinery, not information, at this n.

**Soft labels propagate:** a 60/40 S2/S3 week contributes fractional counts to
both rows — classifier ambiguity flows into posterior uncertainty instead of
being argmax'd away.

**Outputs:** P(state at 2w/1M/3M/6M); P(touch S4 before S5); median weeks to
S5; expected S2+ occupancy — all via forward simulation with a fixed seed
(reproducible vintages).

**Not yet built (was in v1 §5, now the M8 candidate):** hazard-rate covariates
(rhetoric score, novelty flags, deal-odds momentum, Hawkes intensity
modulating transition probabilities). The covariate *contract* is in
priors.yaml; the multinomial-logistic layer is not. With 23 transitions it
would be pure prior anyway — it becomes estimable as vintages accumulate.

**Analogs (unchanged):** Abqaiq 2019, Soleimani 2020, Apr/Oct 2024, June 2025,
and the 1984–88 Tanker War lesson (premium bleeds when fundamentals are loose
— see §9 weakness 6).

---

## 6. Event layer — built, with verdicts

Three components, all live, now with empirical results:

- **Event studies:** +20d abnormal Brent after **novel** events +16.1% vs
  +10.1% after repeats — post-novel drift **SUPPORTED** (caveat: overlapping
  windows in a dense stream, n≈20/bucket).
- **Habituation:** **WEAK — the v1 hypothesis is not supported in this war.**
  Fitted decay half-life ~31 repetitions; first-of-kind |move| 5.15% vs 4.95%
  for 4th+. The flat tape during repeated strike cycles is more plausibly
  fundamentals (spare capacity, demand) than habituation. A2 is held FLAT
  until the decay strengthens. v1's "directly tradable" claim is retracted.
- **Hawkes:** branching ratio 0.89 (near-critical — escalations breeding
  escalations), current intensity ~2× long-run. Honesty: the profile-likelihood
  range (0.05–0.95) is nearly uninformative at n=47 — this is a live dashboard
  number, not an estimate to size on.

---

## 7. Alpha construction — live readings and falsification results

| # | Signal | Status (2026-07-18) | Falsification verdict |
|---|---|---|---|
| A1 | P−Q scenario divergence | **Fires SHORT** — see below | Not testable yet (needs vintage history) |
| A2 | Habituation vol premium | FLAT — hypothesis weak (§6) | Demoted pending stronger decay |
| A3 | Post-novel-event drift | LONG (2 novel events last 7d: Sanaa/Abha front, Kuwait desalination) | **SURVIVES** — 94% of ±50% perturbation grid positive, hit 68%, worst −19% |
| A4 | Hormuz basis | FLAT — legs agree on persistence | Not testable yet (needs vintage history) |
| A5 | Deal-shock hedge | Always-on with A1 (mandatory) | Stress-validated: April tape costs unhedged book −2.3%, hedged −0.5% |
| A6 | Rhetoric-leads-kinetics | FLAT (sparse series; note Jul 16 WH softening) | Survives grid (100%) but n=5 — monitoring only |

**The A1 surprise (biggest single finding).** v1 assumed the market prices
"contained tit-for-tat with likely resolution" and A1 would fire LONG
escalation tails. As built: model P(S2+ at 3m) = 60–65%, but the market's
normalization CDF implies P(not normal by Sep 30) ≈ 76% — **the market is
already more pessimistic than the Mearsheimer-primed model at 3 months.** The
July-31 market collapsed from ~45% to ~2% within days of the v1 doc. Two
readings, both encoded: (a) the containment consensus died and Q now embeds
persistence — the original A1 edge has been arbitraged to the SHORT side;
(b) caveat before trusting the SHORT: the comparison maps "not in S2+" to
"normalized," which overstates model-implied normalization (S1 can hold with
transits below 60) — refine the mapping before sizing anything.

Portfolio rules unchanged (vol-targeted, defined-risk only, correlation cap,
peace-shock stress on every book). Stress now uses **this war's own tapes**
(Jun 14–18 MOU, Apr 7–11 whiplash, Jul 8–13 collapse) rather than stylized
scenarios. Backtest protocol unchanged in spirit, sharpened in role: it is a
**falsification harness, not a performance estimator** — survival earns live
monitoring, never capital directly.

---

## 8. Repo structure — as-built deltas

v1 layout held, with additions:

```
├── streamlit_app.py           # Streamlit Cloud entrypoint (GeoMacro3 dashboard):
│                              #   keyless live-fetch, 1h cache, prior version radio
│                              #   + strength slider, five tabs
├── docs/DESIGN.md             # this document
├── config/
│   ├── priors.yaml            # v1 AND v2 vintages + strength knob
│   └── coded_events_backfill.yaml  # frozen Feb–Jul history (canonical)
├── src/ingest/newsrss.py      # Google News RSS (primary headline source)
├── src/coding/prompts/        # rung_mapper_v1.md (frozen), v2 (live), rhetoric_v1
└── data/briefs/               # daily brief snapshots with what-changed diffs
```

Removed from plan: PyMC (closed form instead), ACLED/FIRMS/OpenSky modules
(deferred, §3.2). `make refresh` = ingest → headlines → code → models → brief.

**Milestones M1–M7: all shipped 2026-07-17.** Next phase (each one session):

- **M8 — Covariate hazards:** the multinomial-logistic transition layer once
  ≥ 35–40 weekly transitions exist; Hawkes intensity + rhetoric momentum as
  first covariates.
- **M9 — A1 mapping fix + vintage backtests:** proper "normalized" event
  definition (PortWatch 7dMA ≥ 60 within horizon, matching Polymarket
  resolution) via simulation from T; A1/A4 walk-forward on accumulated daily
  vintages.
- **M10 — Fundamentals controls:** EIA balances + FRED breakevens (keys
  needed) to separate escalation premium from supply/demand — the Tanker War
  failure mode.
- **M11 — S3 adjudication:** formal Bayes-factor tracking of v1 vs v2 priors
  as transitions accumulate — let the war itself grade Mearsheimer's revision.

---

## 9. Known weaknesses (CLAUDE.md keeps these in view)

1. **n = 1 war, ~23 weekly transitions.** Posterior is 79% prior at default
   settings — and the model *prints this* on every run. Decision-support, not
   signal factory.
2. **Q is partly unobservable with free data.** Delayed ETF options, ~5-day
   PortWatch lag. Trust changes over levels — everywhere.
3. **P and Q share PortWatch** (new, learned in build): it is simultaneously
   the model's S2 input and Polymarket's resolution source. A naive P−Q gap
   partly reflects the same series at different lags; the basis monitor
   decomposes mechanical-lag vs real-belief-gap before calling anything alpha.
4. **The A1 normalization mapping** ("not in S2+" ≈ "normalized") overstates
   model-implied normalization. Fix in M9 before sizing the SHORT.
5. **LLM coder drift is real, not theoretical** — the first live run produced
   8 spurious S4s and miscoded escalatory diplomacy as S5. Versioned prompts +
   QA-against-frozen-history caught it; keep that loop mandatory.
6. **The Tanker War lesson still cuts against the thesis:** premium bleeds
   when fundamentals are loose even if the ladder holds. Fundamentals controls
   (M10) remain the top structural gap.
7. **Mearsheimer could be wrong — in either vintage.** v1 and v2 are frozen
   precisely so the data can grade them (M11). The A1 surprise shows the
   *market* can out-Mearsheimer the model: encode the thesis, but let Q update
   you too. A5 stays mandatory.
8. **Single-scalar strength slider** can't express row-level confidence
   (high on deal-fragility, lower on S4 behavior). Prior *versions* carry
   shape differences; the slider carries only global confidence. Don't ask it
   to do more.

## 10. Key sources

Mearsheimer: *Bombing to Lose* (Substack) · ScheerPost interview, Mar 2026 ·
Diesen episode, Jul 16 2026 (horizontal escalation; MOU-as-surrender; no
escalation dominance; munitions constraints; hardliner ascendancy) ·
Responsible Statecraft on airpower coercion.
Chronology: Wikipedia *2026 Iran war* + *Timeline* (fetched 2026-07-17, coded
into the frozen backfill).
Market/context: IMF PortWatch · Polymarket Iran/Hormuz families · Goldman
Hormuz scenarios · live dashboard (GeoMacro3 on Streamlit).
