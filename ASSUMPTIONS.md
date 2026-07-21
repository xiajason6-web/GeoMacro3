# Assumptions Register

Living audit of every judgment-based number in the model. Each entry: what it
means in plain language, its grounding tag, how it COULD be estimated, and what
the research (2026-07-20 sweep) found. Update when a knob changes status.

Tags: **FOUNDED** (data/literature/arithmetic) · **DEFENSIBLE** (argued judgment)
· **ARBITRARY** (declared conviction, ungrounded) · **SCENARIO** (explicitly a
what-if input, not a claim).

---

## 1. Transition prior matrix (`priors.yaml`)

**Plain language:** the grid of "if the war is in phase X this week, what's the
chance it's in phase Y next week" — before looking at this war's data. Encodes
Mearsheimer: deals collapse, the war widens rather than climbs, all-out war
doesn't stick.

**Tag:** DEFENSIBLE (shape) / ARBITRARY (exact magnitudes).

**Estimation path:** fit to analog conflicts coded into the same S0–S5 states.
**Research (2026-07-20):** feasible — the Tanker War (1984–88) has directly
codable phases: 1981–83 one-sided anti-shipping (≈S1), 1984 both-sides tanker
war, 71 ships hit (≈S2), May 1984 Iran strikes Kuwaiti/Saudi tankers in their
waters (≈S3 events), 1986–87 reflagging + Earnest Will (external intervention),
Aug 1988 UN ceasefire that HELD (S5 absorbing — unlike our prior!). Add Abqaiq
2019, Soleimani 2020, Apr/Oct 2024 exchanges, Jun 2025 12-day war. The academic
base exists: escalation-dynamics models (arXiv 2503.03945), ViEWS/PRIO
prediction challenge, ConflictForecast.org.

**Base-rate check (IMPORTANT, new):** interstate-war statistics (Correlates of
War, 1823–2003): **median war duration < 5 months; 79% end within 2 years.**
This war is at ~5 months now. Our prior (P(S5 at 3m) ≈ 9%) is FAR more
persistent than the reference class — that deviation IS the Mearsheimer bet,
made explicit. The scorecard is what licenses it; if M falls, the prior should
drift back toward the base rate. The Tanker War cuts both ways: it supports
multi-year S2/S3 persistence (4+ years!) AND a negotiated exit that stuck.

## 2. Prior strength / slider (`STRENGTH_MIN=0.26`, `STRENGTH_MAX=4.0`, M10 map)

**Plain language:** one dial for "trust the theory vs trust the ~23 weeks of
data." Endpoints computed; the mapping M→strength is linear by choice.

**Tag:** FOUNDED (endpoints: 50/50 crossover arithmetic; empirical-Bayes
ceiling) / ARBITRARY (linearity of the map).

**Estimation path:** power priors (Ibrahim & Chen) — estimate the discount on
historical/expert information from prior-data agreement; already approximated
by our empirical-Bayes sweep (optimum ≈ 4.0, agrees with scorecard ≈ 3.2).

## 3. Scorecard weights + grading formulas (M10)

**Plain language:** five Mearsheimer predictions each get a 0–1 grade and equal
votes. Both the grades' formulas and the equal votes are judgment. NOTE: the
scorecard is structured judgment grading structured judgment — its agreement
with empirical Bayes is the independent check, not the scorecard itself.

**Tag:** ARBITRARY (weights, formula constants).

**Estimation path:** structured expert elicitation — AHP pairwise comparisons;
Cooke's Classical Method (performance-weighted experts via calibration
questions); Delphi rounds. Literature: NCBI/ISPOR elicitation protocols.
Do a solo-AHP pass at minimum; a real multi-expert Cooke panel is the upgrade.

**Grounding gained (deals_decay grader):** ceasefire literature (Fortna 48→188
cases; Clayton et al. 2023): collapse hazard is HIGHEST at the start and around
the **six-week mark**; durability rises with implementation mechanisms
(monitoring, peacekeeping) — which US–Iran ceasefires LACK. Our observed
2.9-week median deal half-life sits in the literature's peak-hazard window, and
the no-mechanism structure predicts fragility. The 12-week "decay confirmed"
cutoff is now DEFENSIBLE (6-week critical phase ×2 margin).

## 4. State-classifier soft labels (`state_labels.py`)

**Plain language:** when the transit data is ambiguous about what phase we're
in, how to split the probability (e.g. "70% lull / 30% tit-for-tat").

**Tag:** ARBITRARY (mass splits) / DEFENSIBLE (transit thresholds 30%/80%,
which operationalize the taxonomy).

**Estimation path:** estimate emissions with a Hidden Markov Model (transit
level given true phase) or ordered logit with fitted cutoffs. Standard
machinery; removes the hand-set splits. Needs the analog-conflict corpus (see
#1) to be worth fitting at n>23.

## 5. Covariate effect sizes (`S4_SUPPRESS=0.6`, `S4_DECAY_BOOST=0.8`,
`S3_PUMP=0.7`, `S5_DRIFT=0.4`)

**Plain language:** how hard the fuel gauges bend the forecast (e.g. at full
munitions pressure, entry to all-out war is cut to 40%).

**Tag:** ARBITRARY (magnitudes) / mechanism now EVIDENCED for S4:
**Research (2026-07-20):** June 2025 is direct precedent — WSJ-reported Israeli
Arrow rationing within ~1 week of max-tempo exchange ("conservation became a
structural necessity"); US backfilling with THAAD/SM-3 and burning through its
own. Depletion demonstrably changes behavior at high tempo. Magnitude still
judgment; estimation path = event-study elasticities on analog episodes (how
much did depletion episodes actually shift subsequent escalation?).

## 6. Pressure mappings (gap/20, runway/30, ratio anchor 10:1, spread/1.5,
oil-pain band, Iran buffer)

**Plain language:** the conversion of raw readings (production gap, runway,
spread) into 0–1 "pressure" scores.

**Tags & research:**
- Production gap 15:1, 35-month lead: **FOUNDED** (FPRI, MWI — see
  `config/munitions.yaml`).
- Cost-exchange "alarming" ≈10:1: **FOUNDED** (Apr-2024 real exchange,
  ~$80-100M vs ~$1B; Reuters/Haaretz). Demoted to illustrative weight anyway.
- **US oil-pain band $100–150: NOW FOUNDED.** Oil burden >4% of GDP preceded the
  1973/79/90/2008 recessions (≈Brent $120–140 today); Wells Fargo $130
  sustained ⇒ recession; +10¢ gasoline ≈ −0.6% presidential approval; $10/bbl ≈
  25–30¢ gasoline. Sources in `config/economic.yaml`.
- **Iran FX buffer: RE-GROUNDED $25B → $12.2B** (IMF usable-reserves estimate;
  >$100B frozen; NDF oil fund ~$10B left — Al Jazeera/IMF, Stimson 2025).
  Effect: Iran fiscal runway 268→130 days; Iran-pain 0→0.28; p_b 0→0.14. The
  prior guess was ~2× too generous to Iran — the model now says Iran is
  economically closer to cracking than first built.
- Denominators (/20, /30, /1.5): **ARBITRARY** sensitivity choices; estimation
  path = sensitivity sweep (do conclusions survive ±50%?).

## 7. Munitions cost catalog + inventory band (`munitions.yaml`)

**Plain language:** price tags per munition; how many interceptors exist in
theater.
**Tag:** FOUNDED (catalog ≈ literature) / SCENARIO (inventory 150–500 — real
counts are classified; labeled a scenario, never an estimate).

## 8. Signal/read thresholds (A1 0.10, A3 0.02, A6 0.5; premia cutoffs;
fundamentals "mild alpha" bands; stress book weights)

**Plain language:** when a dashboard light switches from grey to green.
**Tag:** ARBITRARY, low stakes (read layer, doesn't move P).
**Estimation path:** backtest calibration — the falsification harness already
built; set cutoffs where they separate real from false signals historically.

## 9. Structural choices (not numbers)

- Six ordinal states; S3-as-attractor framing — DEFENSIBLE, now empirically
  supported (S3 recurrence 62%, spread trajectory).
- 8d kept out of P (politics too noisy for a covariate) — DEFENSIBLE, per
  Mearsheimer's own materiel>politics caveat.
- Frozen backfill + append-only coding; prompt/prior versioning — method, not
  assumption.
- Double-counting caution: the 6.6:1 exchange feeds BOTH the S4 gate and the
  scorecard (which raises prior strength, whose shape also suppresses S4).
  Thesis-consistent but compounding; the static-covariates toggle is the
  stress lever.

---

## Sensitivity sweep results (2026-07-20, `make sensitivity`)

One-at-a-time ±50% on every arbitrary knob (effect sizes, pressures, soft-label
masses, scorecard weights) + two adversarial combos (ALL-HAWK, ALL-DOVE).
34 perturbations. Verdicts on the headline conclusions:

| Conclusion | Base | Range across sweep | Verdict |
|---|---|---|---|
| C1 P(S2+ @3m) — war persists | 69% | 64–73% | **ROBUST** (never loses majority) |
| C2 P(S3 @3m) — widening/attractor | 34% | 27–42% | **ROBUST** (always ≥1.8× the uninformed 15%) |
| C3 P(S4 @3m) — small all-out tail | 4% | 2–5% | **ROBUST** (never exceeds 5%) |
| C4 P(touch S4 before S5) | 35% | 24–41% | **LEVEL-FRAGILE** — direction fine, level knob-sensitive; quote "one-in-four to two-in-five," never "39%" |

**C4 remediation (2026-07-21):** the fragility is structural — a first-passage
RACE between two rare events ratio-compounds hazard perturbations over ~52
steps. Three-part fix implemented: (1) the race stat is now displayed
everywhere as its knob-uncertainty band (`sensitivity.touch_band()`, currently
24–41%), never a point; (2) stabler MARGINAL statistics added
(`p_visit_s4/s5_3m/6m`, simulated without cross-absorption): P(visit S4 @3m)
33% vs S4 occupancy 4% — brief all-out excursions likely, sustained all-out war
rare (matches the two observed S4 excursion weeks); P(visit S5 @3m) 57% vs S5
occupancy 8% — deal episodes likely, deals decay (matches April/June). The race
stat had conflated visiting with staying. (3) The deep fix — grounding the two
hazards via the analog-conflict fit — remains queue #1.
| C5 derived prior_strength | 3.19 | 3.16–3.23 | **ROBUST** — near-invariant to weights |

Two structural lessons:
1. **No single knob matters much** — the largest single-knob moves are S3_PUMP/
   p_c (C2 ±5pp). Only the correlated adversarial combos move conclusions
   meaningfully. The model's claims rest on the DATA + prior shape, not on any
   one conviction number.
2. **The scorecard's equal-weights worry is currently moot:** derived strength
   is invariant to ±50% reweighting *because all five sub-scores agree*
   (0.69–0.80). Weights only start mattering when restrictions diverge — which
   is exactly when the AHP pass (queue #3) becomes worth doing. Watch for
   sub-score divergence as the trigger.

## Analog-conflict transition fit — DONE (2026-07-21)

`config/analogs.yaml` + `src/model/analogs.py`: six conflicts coded to dated
S0–S5 segments (Tanker War 1981–89, Gulf crisis 2019, Soleimani 2020, Apr/Oct
2024 exchanges, 12-day war 2025), per-conflict normalized to a 20-pseudo-count
budget × relevance weight (0.5–0.8). Total 68 analog pseudo-counts join the
posterior as a third voice: `strength·prior + analogs + live_counts`.

What the corpus taught the model (at strength 1, covariates on):
- **S0 @3m 9% → 24%** — the interstate base rate ("most wars end") finally has
  mass; the queue's base-rate reconciliation is now IN the model, not a memo.
- S3 @3m 34% → 25% — 2019 precedent: Gulf-infra escalation can decay dealless.
- Mean weeks in S2+ 8.1 → 9.5 — conditional S2 persistence UP (Tanker War
  grind) even as overall occupancy shifts toward S0. Both Mearsheimer-relevant
  claims sharpened, in opposite directions — the mark of real evidence.
- P(visit S5 @3m) 57% → 47% — analogs de-escalated *without* deals.
- S4 row: 43%→S1, 26%→S5, 30% self — S4 excursions resolve fast (independent
  support for the v2 unsustainable-S4 claim).
- S5 held in 1988 and 2025 — the corpus argues BOTH sides of deal-decay.

Honest limits: coding the analogs to S0–S5 involved mapping judgments
(documented in analogs.yaml — e.g., Tanker War "S2 by semantics, not transits");
the 20-count budget and relevance weights are declared knobs (sweepable); the
C4 knob-band barely narrowed (23–40%) because it is driven by covariate effect
sizes, not hazard levels — narrowing it needs event-study elasticities (#5).

## Event-study elasticities — DONE (2026-07-21)

`src/model/elasticities.py`. Findings and actions:
- **S4_SUPPRESS 0.6 → 0.15 (calibrated DOWN).** Empirical S4 continuation
  is 44% (5 episodes, 9 weeks, analogs+live: all-out excursions last ~1.8
  weeks) but the model's T[S4,S4] was only 23% — the v2 prior + analog counts
  already encode S4-unsustainability and the old knob DOUBLE-COUNTED it
  (register #9's compounding caution, demonstrated empirically; calibration
  hit the 0.0 boundary). S4_DECAY_BOOST halved to 0.4 for the same reason.
- **S3_PUMP 0.7 → 0.8 (calibrated UP).** This war's own weekly series:
  P(S3 next | spread high) = 30% vs 19% low → implied 0.85, shaded to 0.8
  (n_high = 3 weeks).
- **S5_DRIFT stays declared** — unidentifiable (p_b ~0 all war; 1988
  exhaustion→ceasefire is one qualitative episode).
- **Payoff: the C4 band collapsed 23–40% → 35–41%** (width 17pp → 6pp), and the
  refreshed sweep now shows **ALL FIVE conclusions ROBUST** (C1 52–61%,
  C2 19–32%, C3 3–4%, C4 35–41%, C5 3.14–3.23). Post-analog-fit bases:
  persistence 57%, S3 25%, S0 has real mass (base rate).

## Priority queue — CLOSED (2026-07-21)

1. ~~Analog-conflict transition fit~~ **DONE**.
2. ~~Sensitivity sweep~~ **DONE**; re-run after any knob change (`make sensitivity`).
3. ~~Event-study elasticities~~ **DONE** (above).
4. **Solo-AHP scorecard weights — PARKED with trigger:** currently moot (all
   sub-scores agree, 0.69–0.80, so weights don't bind). Trigger: any two
   sub-scores diverge by >0.25 — then do the AHP pass before trusting M.
5. **HMM soft-label emissions — BLOCKED BY DATA, with trigger:** fitting
   emissions needs an observable series per analog week, but transit data
   doesn't exist pre-2019 (PortWatch starts 2019; no Tanker-War AIS). Trigger:
   ~40+ live weeks accumulated, then fit on live data alone.

Every actionable grounding item is done; the remaining two have explicit
reopen-triggers. The register is maintenance-mode: new knobs enter with a tag.

## Rigor pass (2026-07-21) — uncertainty quantification end-to-end

- **War premium now carries a prediction interval:** $26 ± $10 (pre-war OLS
  residual + leverage term at the extrapolation point; understated — daily
  residuals autocorrelated — but kills the bare-point false precision).
- **Transition forecasts carry 80% credible intervals** (Dirichlet posterior
  row sampling, 400 draws): P(S2+ @3m) = 57% [43%, 71%]; S3 [16%, 35%];
  S4 [1%, 6%]; S5 [3%, 11%]. Parameter uncertainty only — classifier and
  structural uncertainty are additional.
- **Inter-coder agreement chance-corrected:** rung κ = 0.72 ("substantial",
  Landis-Koch), target-type κ = 0.42 ("moderate"). Quote κ, not the raw 82%.
- **A3 DOWNGRADED:** label-permutation test on the novel-vs-repeated drift
  gives p ≈ 0.44 — not significant. A3 survived the ±50% parameter grid but
  fails the significance hurdle; these are different tests and clearing one is
  not clearing the other. Confidence forced to "low" in signals.py.
- **Variant tilts stated as within-noise:** all five bands include zero; the
  tilts rank buckets, they do not size positions (stated on the dashboard).
- **Triangulation legs now beta-adjusted** (per-bucket S&P beta, not naive 1×
  subtraction).
- **Scorecard circularity disclosed in the note itself** (§III): structured
  judgment grading structured judgment; the independent check is
  empirical-Bayes convergence (~3 vs 3.2) — convergence, not proof.
- Remaining below-scratch items requiring resources: EIA key (user, free),
  CME settlement data (paid).
