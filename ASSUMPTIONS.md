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

## Priority queue for further grounding

1. **Analog-conflict transition fit** (#1) — biggest object, clearest path;
   Tanker War + 2019/2020/2024/2025 episodes coded to S0–S5. Also yields the
   HMM emissions (#4) and event-study elasticities (#5) from the same corpus.
2. **Sensitivity sweep** on the arbitrary denominators and effect sizes (#5,
   #6) — which conclusions survive ±50%?
3. **Solo-AHP pass** on scorecard weights (#3), documented in this file.
4. Base-rate reconciliation note in the daily brief: report P(S5) vs the
   interstate-war base rate so the Mearsheimer bet stays visible.
