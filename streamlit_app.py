"""Iran Escalation Pricing Model — Streamlit dashboard (GeoMacro3).

Streamlit Cloud entrypoint. Cold start pulls all keyless sources live, lands
them in the ephemeral lake, runs the model chain; cached 1h. Optional secrets:
ANTHROPIC_API_KEY (live event coder), EIA_API_KEY (inventories driver).

Layout follows the model's narrative, one idea per tab:
  📍 Now — what is happening (transits, regime, spread, events)
  🔮 Forecast — what P says (state odds, touch marginals, endurance gauges)
  💹 Market — what Q says (normalization CDF, curve, RND, premium decomposition)
  ⚡ Divergence — where P and Q disagree (head-to-head, signals, stress)
  🧭 Trust — why believe any of it (scorecard, track record, data status, limits)
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iran Escalation Model", page_icon="🛢️",
                   layout="wide")

# ---- university-press styling (after the Ricci cover) ---------------------- #
# Deep green field + sage stripes + vermillion serif display. Also fixes the
# stock-Streamlit annoyances: metric values/labels truncating to "…", oversized
# vertical gaps, and sans-serif headings.
st.markdown("""
<style>
:root {
  --sage: #BFD8C7;
  --sage-muted: rgba(191, 216, 199, 0.55);
  --vermillion: #D14D28;
  --green-raised: #1A3A2E;
  --hairline: rgba(191, 216, 199, 0.18);
}
/* page geometry */
.block-container { padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1250px; }
/* serif display type throughout */
h1, h2, h3, h4, [data-testid="stMetricValue"], [data-testid="stMarkdownContainer"] p {
  font-family: Georgia, "Palatino Linotype", "Book Antiqua", serif;
}
/* masthead */
.masthead-kicker {
  font-family: Georgia, serif; font-variant: small-caps; letter-spacing: 0.22em;
  color: var(--sage); font-size: 0.95rem; margin: 0 0 0.2rem 0;
}
.masthead-title {
  font-family: Georgia, serif; color: var(--vermillion); font-weight: 400;
  font-size: 2.6rem; line-height: 1.08; margin: 0 0 0.15rem 0;
}
.masthead-sub {
  font-family: Georgia, serif; font-style: italic; color: var(--vermillion);
  opacity: 0.85; font-size: 1.15rem; margin: 0 0 1.1rem 0;
}
/* section headings: clean sage serif, no ornament */
h3 {
  color: var(--sage) !important; font-weight: 500 !important;
  font-size: 1.25rem !important;
  margin-top: 1.4rem !important; margin-bottom: 0.4rem !important;
}
/* metrics: card on raised green, NO ellipsis truncation, serif values */
[data-testid="stMetric"] {
  background: var(--green-raised); border: 1px solid var(--hairline);
  border-radius: 2px; padding: 0.7rem 0.9rem;
}
/* kill ellipsis truncation at EVERY level Streamlit nests it */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
  white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.4rem !important; line-height: 1.25 !important; color: #EDF4EF;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  font-variant: small-caps; letter-spacing: 0.06em; color: var(--sage-muted);
}
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
/* tabs: small-caps serif, vermillion active underline */
button[data-baseweb="tab"] {
  font-family: Georgia, serif !important; letter-spacing: 0.05em;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--vermillion) !important;
}
[data-baseweb="tab-highlight"] { background-color: var(--vermillion) !important; }
/* captions in muted sage; tighter */
[data-testid="stCaptionContainer"] {
  color: var(--sage-muted) !important; margin-top: -0.3rem;
}
/* dividers as sage hairlines, tighter rhythm */
hr { border-color: var(--hairline) !important; margin: 1.1rem 0 !important; }
/* expanders on raised green */
[data-testid="stExpander"] {
  background: var(--green-raised); border: 1px solid var(--hairline);
  border-radius: 2px;
}
/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--hairline); }
[data-testid="stSidebar"] h1 { font-size: 1.35rem; color: var(--sage); }
</style>
""", unsafe_allow_html=True)

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]
STATE_NAMES = {
    "S0": "Lull", "S1": "Tit-for-tat", "S2": "Chokepoint war",
    "S3": "Gulf infra war", "S4": "All-out war", "S5": "De-escalation/deal",
}


# --------------------------------------------------------------------------- #
# Data + model layer (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner="Pulling live data (PortWatch, Polymarket, prices)...")
def build_lake() -> dict:
    from src.common import write_partition
    from src.ingest import polymarket, portwatch, prices

    status = {}
    for name, mod in [("portwatch", portwatch), ("polymarket", polymarket),
                      ("prices", prices)]:
        try:
            df = mod.fetch()
            write_partition(df, name)
            status[name] = f"{len(df)} rows"
        except Exception as exc:  # noqa: BLE001 — degrade per-source, keep app up
            status[name] = f"FAILED: {str(exc)[:80]}"

    try:
        from src.market_implied import curve
        write_partition(curve.fetch(), "futures_curve")
        status["curve"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["curve"] = f"FAILED: {str(exc)[:80]}"
    try:
        from src.market_implied import rnd as rnd_mod
        frames = [f for s in ("USO", "BNO") if len(f := rnd_mod.fetch(s))]
        if frames:
            write_partition(pd.concat(frames, ignore_index=True), "rnd")
            status["rnd"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["rnd"] = f"FAILED: {str(exc)[:80]}"

    # coded history: frozen YAML -> lake (no LLM needed)
    from src.coding import load_backfill
    load_backfill.main()
    status["coded_events"] = "frozen backfill"

    # optional live enrichment when a key is present (Streamlit secret or env)
    import os
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            os.environ.setdefault("ANTHROPIC_API_KEY", st.secrets["ANTHROPIC_API_KEY"])
    except Exception:  # noqa: BLE001 — no secrets file is fine
        pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from src.ingest import newsrss
            from src.coding import llm_coder
            write_partition(newsrss.fetch(), "gdelt_articles")
            llm_coder.main()
            status["coded_events"] = "frozen + live-enriched"
        except Exception as exc:  # noqa: BLE001
            status["coded_events"] = f"frozen (live enrich failed: {str(exc)[:40]})"

    from src.common import write_partition as wp
    try:
        from src.market_implied.predmkt import build_panel
        wp(build_panel(), "predmkt_panel")
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.features.horizontal_spread import weekly_spread
        wp(weekly_spread(), "horizontal_spread")
        status["horizontal_spread"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["horizontal_spread"] = f"FAILED: {str(exc)[:60]}"
    try:
        from src.features import munitions as _mun
        led = _mun.build_ledger()
        wp(_mun.weekly(led), "munitions_weekly")
        wp(led, "munitions_ledger")
        status["munitions"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["munitions"] = f"FAILED: {str(exc)[:60]}"
    try:
        from src.features.economic import readings as _econ
        e = _econ()
        status["economic (8b)"] = f"p_b={e['economic_pressure']:.2f}"
    except Exception as exc:  # noqa: BLE001
        status["economic (8b)"] = f"FAILED: {str(exc)[:50]}"
    try:
        from src.model.scorecard import compute as _sc
        r = _sc()
        wp(pd.DataFrame([{**{k: v["score"] for k, v in r["sub_scores"].items()},
                          "M": r["M"], "derived_strength": r["derived_strength"]}]),
           "mearsheimer_scorecard")
        status["scorecard (M10)"] = f"M={r['M']:.2f} -> strength {r['derived_strength']:.1f}"
    except Exception as exc:  # noqa: BLE001
        status["scorecard (M10)"] = f"FAILED: {str(exc)[:50]}"
    try:
        from src.features.fundamentals import decompose
        fd = decompose()
        wp(pd.DataFrame([{k: v for k, v in fd.items() if not isinstance(v, dict)}]),
           "fundamentals")
        status["fundamentals"] = f"war premium ${fd['war_premium']:+.0f} (R2={fd['prewar_r2']:.2f})"
    except Exception as exc:  # noqa: BLE001
        status["fundamentals"] = f"FAILED: {str(exc)[:50]}"
    try:
        from src.features.premia import decompose as _premia
        wp(pd.DataFrame([_premia()]), "premia")
        status["premia"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["premia"] = f"FAILED: {str(exc)[:50]}"
    return {"status": status, "as_of": dt.datetime.utcnow().isoformat()}


@st.cache_data(ttl=3600, show_spinner="Running regime model...")
def regime(prior_strength: float, _lake_key: str) -> dict:
    # reload defends against stale-module skew after Cloud git pulls
    import importlib
    import inspect

    import src.model.regime_markov as rm
    if "use_covariates" not in inspect.signature(rm.run).parameters:
        rm = importlib.reload(rm)
    run = rm.run
    if "use_covariates" in inspect.signature(run).parameters:
        r = run(prior_strength, use_covariates=True)
        static = run(prior_strength, use_covariates=False)
    else:
        r = run(prior_strength)
        static = r
    try:
        from src.alpha.sensitivity import touch_band
        band = touch_band()
    except Exception:  # noqa: BLE001
        band = None
    r["touch"]["race_band"] = band

    # model-implied normalization CDF (first passage to S0) — the Divergence
    # tab's head-to-head against the market's own CDF
    T, p0 = r["T"], r["p0"]
    rng = np.random.default_rng(20260721)
    N = 20000
    stv = rng.choice(6, size=N, p=p0 / p0.sum())
    first = np.full(N, 999)
    for wk in range(1, 30):
        u = rng.random(N)
        cum = T[stv].cumsum(axis=1)
        stv = np.clip((u[:, None] > cum).sum(axis=1), 0, 5)
        hit = (stv == 0) & (first == 999)
        first[hit] = wk
    today = dt.date.today()
    model_cdf = {}
    for label, d in [("2026-08-31", dt.date(2026, 8, 31)),
                     ("2026-09-30", dt.date(2026, 9, 30)),
                     ("2026-12-31", dt.date(2026, 12, 31))]:
        wks = max(1, round((d - today).days / 7))
        model_cdf[label] = float((first <= wks).mean())

    return {
        "labels": r["labels"], "p0": r["p0"],
        "forecasts": {k: list(map(float, v)) for k, v in r["forecasts"].items()},
        "static_forecasts": {k: list(map(float, v)) for k, v in static["forecasts"].items()},
        "touch": r["touch"], "data_weight": r["data_weight"],
        "covariates": r.get("covariates") or {}, "model_cdf": model_cdf,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def scorecard_derived(_lake_key: str) -> dict:
    from src.model.scorecard import compute
    r = compute()
    return {"strength": r["derived_strength"], "M": r["M"],
            "sub_scores": {k: v["score"] for k, v in r["sub_scores"].items()},
            "details": {k: v["detail"] for k, v in r["sub_scores"].items()}}


lake = build_lake()
lake_key = lake["as_of"]
# --------------------------------------------------------------------------- #
# Additional styles for the essay layout (major sections, lede paragraphs)
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
h2 {
  color: var(--sage) !important; font-weight: 500 !important;
  font-size: 1.55rem !important;
  margin-top: 2.4rem !important; margin-bottom: 0.5rem !important;
}
h3 { font-style: italic; font-weight: 400 !important; }
.lede { font-size: 1.02rem; line-height: 1.65; color: #D9E7DD; max-width: 62rem; }
.lede em { color: var(--vermillion); font-style: normal; }
.keybox { border: 1px solid var(--hairline); background: var(--green-raised);
          padding: 0.8rem 1.1rem; border-radius: 2px; max-width: 62rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar — minimal. Prior strength is ALWAYS scorecard-derived.
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛢️ Iran Escalation Model")
    st.caption("A structural war model (**P**) against market pricing (**Q**); "
               "the divergence is the edge. **Research framework — not "
               "financial advice.**")
    derived = scorecard_derived(lake_key)
    st.caption(f"Prior weight is set by the live Mearsheimer scorecard "
               f"(§III): currently **{derived['strength']:.1f}** "
               f"(fit M = {derived['M']:.0%}). No manual override — if the war "
               "stops behaving like the thesis, the weight falls on its own.")
    st.divider()
    if st.button("↻ Force data refresh"):
        build_lake.clear()
        st.rerun()
    st.caption("[GitHub](https://github.com/xiajason6-web/GeoMacro3) · data "
               "cached 1h · daily CI-graded "
               "[calls ledger](https://github.com/xiajason6-web/GeoMacro3/blob/main/calls/ledger.yaml)")

prior_strength = derived["strength"]

try:
    reg = regime(prior_strength, lake_key)
except Exception as exc:  # noqa: BLE001 — surface the real error; Cloud redacts it
    st.error(f"Regime model failed: {type(exc).__name__}: {exc}")
    st.stop()
labels = reg["labels"]
cur = labels.iloc[-1]

from src.common import read_latest  # noqa: E402

# --------------------------------------------------------------------------- #
# Masthead
# --------------------------------------------------------------------------- #
st.markdown("""
<p class="masthead-kicker">GeoMacro · a structural war model against the market</p>
<h1 class="masthead-title">The Pricing of Escalation</h1>
<p class="masthead-sub">Iran, the Strait of Hormuz, and what the oil market believes</p>
""", unsafe_allow_html=True)

# =========================================================================== #
# I. THE CONCLUSION
# =========================================================================== #
st.markdown("## I. The conclusion")

f3 = reg["forecasts"]["3m"]
t = reg["touch"]
p_s2plus = sum(f3[2:5])
try:
    fd_h = read_latest("fundamentals").iloc[-1]
    war_prem = float(fd_h["war_premium"])
    kept = float(fd_h["frac_premium_persisting"]) if pd.notna(
        fd_h.get("frac_premium_persisting")) else None
except Exception:  # noqa: BLE001
    war_prem, kept = None, None
mdl_dec = reg["model_cdf"].get("2026-12-31")
try:
    pm_ = read_latest("predmkt_panel")
    hz_ = pm_[(pm_["family"] == "hormuz_normalize")
              & (pm_["question"].str.contains("December", na=False))]
    mkt_dec = float(hz_.iloc[0]["yes_prob"])
except Exception:  # noqa: BLE001
    mkt_dec = None

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("The war persists", f"{p_s2plus:.0%}",
          "odds the strait is still a war zone in three months", delta_color="off")
k2.metric("It widens, not climbs", f"{f3[3]:.0%} vs {f3[4]:.0%}",
          "odds of a Gulf-wide infrastructure war vs an all-out war",
          delta_color="off")
k3.metric("Deals happen — and decay", f"{t.get('p_visit_s5_3m', 0):.0%}",
          f"odds of a ceasefire attempt; only {f3[5]:.0%} that one holds",
          delta_color="off")
if war_prem is not None:
    k4.metric("War premium in oil", f"${war_prem:+.0f}",
              f"per barrel of Brent; {kept:.0%} still priced a year out" if kept
              else "per barrel of Brent", delta_color="off")
if mdl_dec is not None and mkt_dec is not None:
    k5.metric("Strait reopens by Dec 31", f"P {mdl_dec:.0%} · Q {mkt_dec:.0%}",
              "this model vs the betting market", delta_color="off")

st.markdown(f"""
<p class="lede">
<em>This war does not end soon, and it grows outward, not upward.</em> Three
months from now, the most likely world still has the Strait of Hormuz closed
and strikes landing on Gulf infrastructure — Kuwaiti desalination plants, Saudi
terminals, US bases in Jordan. All-out war stays unlikely ({f3[4]:.0%}), and so
does lasting peace ({f3[5]:.0%}). A ceasefire attempt is actually
<b>probable</b> — but the war has already produced two, in April and in June,
and each collapsed within about three weeks. The model expects that pattern to
repeat: deals arrive suddenly, and then they die.
</p>
<p class="lede">
<b>Where this view differs from the market's:</b> not on whether the war
continues — the market gave up on a quick reopening weeks ago, and there is no
money in agreeing with everyone. The remaining disagreements are three. First,
oil for delivery <b>next year</b> is priced as if most of the war premium will
be gone by then; this model expects the grind to outlast that optimism. Second,
the war's real growth axis — strikes on Gulf <b>water, power, and export
facilities</b> — barely shows up in oil prices at all; the only market pricing
it is European natural gas. Third, oil options pay heavily for a <b>peace that
sticks</b>, which neither the betting markets nor the two dead ceasefires
support. Each disagreement is developed below, and the boldest ones are
standing bets in the public ledger (§VII), graded automatically as the dates
pass.
</p>
""", unsafe_allow_html=True)

# =========================================================================== #
# II. THE WAR TODAY
# =========================================================================== #
st.markdown("## II. The war today")
st.markdown("""
<p class="lede">
The single most important series in this system: <b>how many ships pass the
Strait of Hormuz each day.</b> Roughly a fifth of the world's oil transits this
strait; the count below (IMF PortWatch, the same source Polymarket uses to
settle its markets) is the war's pulse. Normal is ~75 calls/day; the red line
(60) is the market's own definition of "back to normal."
</p>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    st.subheader("Strait of Hormuz daily transits")
    pw = read_latest("portwatch").copy()
    pw["obs_date"] = pd.to_datetime(pw["obs_date"])
    pw = pw[pw["obs_date"] >= "2026-01-01"].set_index("obs_date")
    chart_df = pd.DataFrame({"daily": pd.to_numeric(pw["n_total"], errors="coerce")})
    chart_df["7d MA"] = chart_df["daily"].rolling(7).mean()
    chart_df["normal (60)"] = 60.0
    st.line_chart(chart_df, height=300)
    st.caption(f"Latest {pw.index.max().date()} — PortWatch publishes ~5 days "
               "late. Read the story left to right: normal traffic into early "
               "March, the collapse when Iran closed the strait, four months "
               "of near-zero, the partial reopening after the June deal — and "
               "the re-closure when that deal died in July.")
with right:
    st.subheader("Weekly regime")
    lab = labels.copy()
    lab["week"] = pd.to_datetime(lab["week"]).dt.date
    show = lab[["week", "state", "frac"]].tail(12).iloc[::-1]
    show["frac"] = (show["frac"] * 100).round(0).astype(int).astype(str) + "%"
    show.columns = ["week", "state", "transits"]
    st.dataframe(show, hide_index=True, height=300)
    st.caption("The war coded into six phases: S0 lull · S1 military "
               "tit-for-tat · S2 chokepoint war · S3 Gulf-infrastructure war · "
               "S4 all-out war · S5 deal. Everything downstream keys off this "
               "classification.")

st.subheader("Is the war widening? (the S3 axis)")
sc1, sc2 = st.columns([3, 2])
with sc1:
    try:
        hs = read_latest("horizontal_spread").copy()
        hs["week"] = pd.to_datetime(hs["week"])
        st.bar_chart(hs.set_index("week")[
            ["third_party_fronts", "proxy_active", "broad_hit"]], height=240)
        st.caption("Distinct third-country targets struck per week — Kuwait, "
                   "UAE, Saudi, Yemen fronts — excluding the belligerents "
                   "themselves and pure maritime targets. This is Mearsheimer's "
                   "'horizontal escalation' made countable.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"spread index unavailable: {exc}")
with sc2:
    try:
        from src.features.horizontal_spread import spread_now
        sp = spread_now()
        st.metric("Fronts this week", sp["third_party_fronts"],
                  sp["third_party_list"] or "—", delta_color="off")
        st.metric("Trailing 4-week spread", f"{sp['trailing_4wk_index']:.1f}",
                  f"war average {sp['war_avg_index']:.1f}", delta_color="off")
        if sp["trailing_4wk_index"] > sp["war_avg_index"] * 1.3:
            st.warning("**Widening.** Recent spread is well above the war's "
                       "average — the July re-escalation went broader than any "
                       "earlier phase, not just hotter.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"spread reading unavailable: {exc}")

with st.expander("Recent coded events — the raw material"):
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev_show = ev.sort_values("date", ascending=False)[
        ["date", "actor", "rung", "target_type", "target_country", "severity", "action"]
    ].head(25)
    ev_show["date"] = ev_show["date"].dt.date
    st.dataframe(ev_show, hide_index=True, height=300)
    st.caption("News headlines coded into structured events by an LLM under "
               "frozen, versioned prompts (a second model agrees on the phase "
               "coding 82% of the time). Hand-verified history through July 17 "
               "is frozen; the live coder appends but never rewrites.")

# =========================================================================== #
# III. THE ENDURANCE ARGUMENT
# =========================================================================== #
st.markdown("## III. The endurance argument — why the forecast looks this way")
st.markdown("""
<p class="lede">
The forecast is built on an argument, not a curve-fit. The argument is John
Mearsheimer's, and in plain terms it runs like this. <b>Bombing does not make
countries surrender</b> — no state fighting for its survival has ever been
coerced by airpower alone, and Iran is fighting for its survival while America
is fighting for leverage, so Iran can absorb pain longer than Washington can
sustain interest. <b>The bombing cannot be sustained anyway</b>: precision
munitions and interceptors are being spent far faster than any factory can
replace them. <b>Iran's counter-moves are cheap</b>: a drone that costs less
than a car forces a defense that costs more than a house, and tankers,
pipelines, and desalination plants cannot shoot back. And <b>neither government
can afford to be seen quitting</b> — every strike is framed as retaliation, so
ending the war requires one leader to accept a public humiliation that neither
can survive politically. Put together: a war that cannot be won, cannot be
sustained at full intensity, is cheapest to fight sideways, and has no
dignified exit — so it <em>grinds on and spreads outward</em>.
</p>
<p class="lede">
Each of those claims is testable, and the bars below score them against what
is actually happening. The average sets how heavily the thesis leans on the
forecast. This cuts both ways: if a ceasefire holds or Iran starts conceding,
the scores fall and the model drifts back toward the historical base rate —
which, it should be said plainly, is against this thesis: <b>the median war
between states ends in under five months</b>. This forecast is a deliberate,
evidence-graded bet against that average.
</p>
""", unsafe_allow_html=True)

sc = scorecard_derived(lake_key)
m1, m2, m3 = st.columns(3)
m1.metric("Thesis fit", f"{sc['M']:.0%}",
          "how Mearsheimer-shaped the war is right now", delta_color="off")
m2.metric("Thesis weight in forecast", f"{sc['strength']:.1f} / 4",
          "derived from the fit — not hand-set", delta_color="off")
_tot = 23.0 + 68.0 + sc["strength"] * 88.0
m3.metric("What the forecast rests on",
          f"{sc['strength']*88/_tot:.0%} thesis · {68/_tot:.0%} history · {23/_tot:.0%} this war",
          "six analog conflicts supply the history", delta_color="off")

LABELS = {
    "no_coercive_leverage": ("No coercive leverage",
                             "US strikes are not producing Iranian concessions"),
    "deals_decay": ("Deals decay", "both 2026 ceasefires collapsed in ~3 weeks"),
    "asymmetric_escalation": ("Horizontal escalation",
                              "the war spreads to cheap third-country targets"),
    "face_lock": ("Face-lock", "neither side can afford to be seen backing down"),
    "endurance_asymmetry": ("Endurance asymmetry", "Iran out-endures US patience"),
}
for k, (label, gloss) in LABELS.items():
    s = sc["sub_scores"].get(k, 0.5)
    colA, colB = st.columns([1, 3])
    colA.markdown(f"**{label}**  \n{s:.2f}")
    with colB:
        st.progress(min(1.0, s))
        st.caption(f"{gloss} — {sc['details'].get(k, '')}")

st.subheader("The physical mechanism: production, money, will")
ci = reg.get("covariates") or {}
g1, g2, g3 = st.columns(3)
g1.metric("Interceptor production gap",
          f"{ci.get('munitions', {}).get('production_gap', 0):.0f} : 1",
          "Iran builds missiles ~15× faster than the US builds interceptors",
          delta_color="off")
try:
    from src.features.economic import readings as _econ_r
    _e = _econ_r()
    g2.metric("Iran's fiscal runway", f"{_e['iran_runway_days']:.0f} days"
              if _e.get("iran_runway_days") else "long",
              "usable reserves ÷ blocked-export losses (IMF figure)",
              delta_color="off")
    g3.metric("US oil pain", f"{_e['us_oil_pain']:.0%}",
              "political pressure from pump prices — none below ~$100 Brent",
              delta_color="off")
except Exception:  # noqa: BLE001
    pass
st.markdown("""
<p class="lede">
These three numbers carry consequences beyond the war itself. Because an
interceptor takes about three years to build, every missile fired in defense
of the Gulf is gone for the duration — which rules out sustained all-out war,
and quietly spends stock America also counts on for deterring China. Because
Iran's cash runs out in roughly four months of blocked exports, economic
gravity will eventually pull it toward a deal — but because neither side can
afford to be seen quitting, the deal won't hold. That is not speculation; it
is the April-to-July record, projected forward.
</p>
<p class="lede">
Follow the chain one step further and it reaches ordinary life. Each
deal-and-collapse cycle teaches traders to shrug at ceasefire headlines, so
peace gets cheaper to announce and easier to abandon. And each week of
interceptor drawdown leaves Gulf cities — whose drinking water comes from
coastal desalination plants — a little less defended against the next cheap
drone. That is why a strike on a desalination plant is this model's tripwire:
it is the moment the war stops being about oil logistics and starts being
about whether Kuwait City has water.
</p>
""", unsafe_allow_html=True)

# =========================================================================== #
# IV. THE FORECAST (P)
# =========================================================================== #
st.markdown("## IV. The forecast — where the war goes from here")
lcol, rcol = st.columns([3, 2])
with lcol:
    rows = []
    for h in ("2w", "1m", "3m", "6m"):
        for s, p in zip(STATES, reg["forecasts"][h]):
            rows.append({"horizon": h, "state": f"{s} {STATE_NAMES[s]}", "prob": p})
    fdf = pd.DataFrame(rows).pivot(index="state", columns="horizon", values="prob")
    st.dataframe(fdf[["2w", "1m", "3m", "6m"]].style.format("{:.0%}"), height=260)
    st.caption("Probability of being in each phase at each horizon. Read the 3m "
               "column top to bottom: the mass sits in the chokepoint/"
               "infrastructure grind.")
with rcol:
    tm = reg["touch"]
    st.metric("Odds of touching all-out war",
              f"{tm.get('p_visit_s4_3m', 0):.0%}",
              f"within 3 months — but only {f3[4]:.0%} that it lasts",
              delta_color="off")
    st.metric("Odds of a ceasefire attempt", f"{tm.get('p_visit_s5_3m', 0):.0%}",
              f"within 3 months — but only {f3[5]:.0%} that it holds",
              delta_color="off")
    band = tm.get("race_band")
    st.metric("All-out war arrives before a deal",
              f"{band['lo']:.0%}–{band['hi']:.0%}" if band else "n/a",
              "a range, honestly — this number resists precision",
              delta_color="off")
st.markdown("""
<p class="lede">
The crucial distinction is <em>visiting versus staying</em>. The model gives a
real chance of touching all-out war briefly — the war has already done it
twice, in the opening decapitation campaign and the March strikes — but almost
no chance of staying there, because §III's arithmetic makes maximum tempo
unsustainable. Same asymmetry for peace: deal <b>episodes</b> are likelier
than not within months, but deals <b>holding</b> is rare. If you remember one
thing from this page: escalation spikes and ceasefires are both, on this
model's read, <em>episodes inside a grind</em> — not endings.
</p>
""", unsafe_allow_html=True)

# =========================================================================== #
# V. THE MARKET (Q)
# =========================================================================== #
st.markdown("## V. What the market believes")
st.markdown("""
<p class="lede">
Four independent windows into the market's mind. Prediction markets state
reopening odds directly. The futures curve reveals how long traders expect
disruption to last (today's barrel trading far above next year's = "this is
temporary"). Options prices imply the odds of extreme moves. And the premium
decomposition asks how much of today's oil price is war at all — versus
ordinary supply and demand.
</p>
""", unsafe_allow_html=True)

qa, qb = st.columns(2)
with qa:
    st.subheader("Prediction markets — reopening odds")
    try:
        pm = read_latest("predmkt_panel")
        hz = pm[pm["family"] == "hormuz_normalize"].copy()
        hz = hz[(hz["end_date"].notna()) & (hz["volume"].fillna(0) > 100_000)]
        hz = hz[hz["end_date"] > dt.date.today().isoformat()].sort_values("end_date")
        cdf = hz[["end_date", "yes_prob"]].rename(
            columns={"end_date": "by", "yes_prob": "P(normalized)"})
        st.line_chart(cdf.set_index("by")["P(normalized)"], height=220)
        st.caption("Millions of dollars of real-money volume. The near months "
                   "collapsed to ~zero weeks ago — the market has fully "
                   "conceded that the strait stays shut near-term.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"prediction markets unavailable: {exc}")
with qb:
    st.subheader("Futures curve — how long is 'temporary'?")
    try:
        fc = read_latest("futures_curve")
        bzl = fc[fc["root"] == "BZ"].sort_values("contract_month")
        st.line_chart(bzl.set_index("contract_month")["close"], height=220)
        spread6 = float(bzl["close"].iloc[0] - bzl["close"].iloc[min(6, len(bzl) - 1)])
        st.caption(f"Brent by delivery month. The steep downslope "
                   f"(front-to-6M {spread6:+.0f}) says the market prices this "
                   "disruption as mostly gone within a year — the single "
                   "clearest thing this model disagrees with.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"curve unavailable: {exc}")

st.subheader("How much of the oil price is war — and which war?")
try:
    fd = read_latest("fundamentals").iloc[-1]
    pr = read_latest("premia").iloc[-1]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("War premium", f"${fd['war_premium']:+.0f}",
              f"Brent trades at ${fd['brent_fred']:.0f}; without the war it "
              f"would be near ${fd['fundamentals_fair']:.0f}", delta_color="off")
    q2.metric("Still priced a year out", f"{fd['frac_premium_persisting']:.0%}"
              if pd.notna(fd.get("frac_premium_persisting")) else "n/a",
              "share of the premium the futures curve keeps at 12 months",
              delta_color="off")
    q3.metric("European gas premium", f"{pr['gas_war_premium_proxy']:+.0%}",
              "the only market pricing Gulf-infrastructure risk", delta_color="off")
    q4.metric("Gold since the war began", f"{pr['gold_since_war']:+.0%}",
              "falling gold = an oil event, not a world event", delta_color="off")
    st.caption("Fair value comes from oil's pre-war relationship to copper, the "
               "dollar, rates, and equities — what oil *would* cost in this "
               "economy with no war. The gap is the war premium. Copper matters "
               "because it shares oil's demand drivers but not its war: it is "
               "the counterfactual twin. Note gold: in a war the textbook says "
               "buy gold, yet it has *fallen* — the market classifies this as a "
               "supply disruption, not a systemic crisis. If gold ever starts "
               "rising *with* escalation, that re-classification is the single "
               "scariest signal on this page.")
except Exception as exc:  # noqa: BLE001
    st.info(f"premium decomposition unavailable: {exc}")

# =========================================================================== #
# VI. THE DIVERGENCE — and what the positions mean
# =========================================================================== #
st.markdown("## VI. The divergence — model vs market, and the positions")

st.subheader("Head-to-head on the question both sides price")
try:
    pm = read_latest("predmkt_panel")
    hz = pm[pm["family"] == "hormuz_normalize"].copy()
    hz = hz[hz["end_date"].notna()]
    rows = []
    for by, mdl in reg["model_cdf"].items():
        sub = hz[hz["end_date"].astype(str).str.startswith(by[:7])]
        mkt = float(sub.sort_values("volume", ascending=False).iloc[0]["yes_prob"]) \
            if len(sub) else None
        rows.append({"strait normal by": by, "model P": f"{mdl:.0%}",
                     "market Q": f"{mkt:.0%}" if mkt is not None else "—",
                     "P − Q": f"{mdl - mkt:+.0%}" if mkt is not None else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Positive P−Q: the model is *more* optimistic about reopening "
               "than the market at that date. These are not hypotheticals — "
               "each row is a timestamped call in the public ledger (§VII), "
               "graded automatically when the date passes.")
except Exception as exc:  # noqa: BLE001
    st.info(f"head-to-head unavailable: {exc}")

st.markdown("""
<div class="keybox">
<b>How to read the positions below.</b> 🟢 <b>LONG</b> = the position profits
if escalation/persistence exceeds what the market has priced (e.g. owning oil
exposure or upside options). 🔴 <b>SHORT</b> = profits if resolution comes
faster or risk premium deflates. ⚪ <b>FLAT</b> = the model sees no edge —
which is a conclusion, not an absence of one. <b>Confidence</b> measures how
much evidence backs the signal's <em>logic</em> (did it survive robustness
tests? how much data supports it?), not the odds of profit — a
<b>low-confidence</b> signal is a hypothesis to watch, a <b>high-confidence</b>
one has survived deliberate attempts to break it. None of them are sized
positions, and the standing rule is that any escalation-long position is
paired with cheap deal-shock protection, because peace arrives overnight here
(April 7 went from "a whole civilization will die tonight" to a ceasefire in
hours).
</div>
""", unsafe_allow_html=True)

try:
    from src.alpha.signals import compute_all
    dirmap = {-1: "🔴 SHORT", 0: "⚪ FLAT", 1: "🟢 LONG"}
    for s in compute_all():
        with st.expander(f"{dirmap[s['direction']]}  **{s['signal']}**  "
                         f"(confidence: {s['confidence']})", expanded=False):
            st.write(s["rationale"])
            for k, v in (s["value"] or {}).items():
                if isinstance(v, dict):
                    v = v.get("yes_prob", v)
                st.markdown(f"- `{k}` = {v}")
            st.warning(s["caveats"])
except Exception as exc:  # noqa: BLE001
    st.error(f"signal computation failed: {exc}")

st.subheader("The stress test that disciplines everything")
try:
    from src.alpha.stress import WINDOWS, _window_moves, book_pnl
    rows = []
    for name, (lo, hi) in WINDOWS.items():
        m = _window_moves(lo, hi)
        if not m:
            continue
        pnl = book_pnl(m)
        rows.append({"scenario (this war's own tapes)": name,
                     "brent": f"{m['brent']:+.1%}", "tankers": f"{m['tankers']:+.1%}",
                     "hedged book": f"{pnl['TOTAL']:+.2%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("A stylized escalation-long book replayed against the war's own "
               "deal windows. The April tape costs an *unhedged* book ~-2.3%; "
               "with the deal hedge it loses ~-0.5%. That difference is why the "
               "hedge is mandatory, not optional: the biggest risk in being "
               "long this war is peace, however briefly it lasts.")
except Exception as exc:  # noqa: BLE001
    st.error(f"stress unavailable: {exc}")

# =========================================================================== #
# VII. TRUST
# =========================================================================== #
st.markdown("## VII. Why believe any of this — the track record")
st.markdown("""
<p class="lede">
A model that never risks being wrong is an opinion. This one publishes
timestamped, falsifiable calls with explicit resolution criteria, graded
automatically every day by CI against the same data sources the markets
settle on. The Brier score (0 = clairvoyant, 0.25 = coin-flipping) accumulates
below as calls resolve — and the ledger is append-only: no call, once made,
can be edited or quietly deleted.
</p>
""", unsafe_allow_html=True)

try:
    from src.report.calls import load, summary
    doc = load()
    s = summary(doc)
    st.markdown(f"**{s['n_calls']} public calls** since {s['first_call']} — "
                f"{s['n_open']} open, {s['n_resolved']} resolved"
                + (f", **Brier {s['brier']:.3f}**" if s["brier"] is not None else "")
                + " · [ledger on GitHub](https://github.com/xiajason6-web/GeoMacro3/blob/main/calls/ledger.yaml)")
    rows = [{"made": c["made"], "p": f"{c['p']:.0%}", "claim": c["claim"],
             "status": c.get("outcome", "open")} for c in doc["calls"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"claim": st.column_config.TextColumn(
                     "claim", width="large")})
except Exception as exc:  # noqa: BLE001
    st.info(f"ledger unavailable: {exc}")

with st.expander("Data status (cached 1h)"):
    cols = st.columns(3)
    for i, (k, v) in enumerate(lake["status"].items()):
        icon = "🟢" if "FAIL" not in str(v) else "🔴"
        cols[i % 3].caption(f"{icon} **{k}**: {v}")

with st.expander("Honest limitations — read before believing any number"):
    st.markdown("""
1. **One war, ~24 observed weeks.** The forecast leans on the thesis and on
   six analog conflicts, and says exactly how much (§III).
2. **Market data is free and delayed** — ETF options, ~5-day transit lag.
   Trust day-over-day changes more than absolute levels.
3. **The model and Polymarket share a data source** (PortWatch), so part of any
   gap can be timing, not disagreement — the pipeline decomposes this before
   calling anything an edge.
4. **Deals decay** is a thesis claim with n=2 (April, June). If a ceasefire
   holds, the scorecard falls and the whole forecast softens on its own.

Every judgment knob in the model is registered, tagged (founded / defensible /
arbitrary), and adversarially perturbed ±50% —
[ASSUMPTIONS.md](https://github.com/xiajason6-web/GeoMacro3/blob/main/ASSUMPTIONS.md).
All §I conclusions survive that sweep. **Not financial advice.**
""")
