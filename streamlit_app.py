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
    try:
        from src.features.equities import readthrough as _eqrt
        wp(_eqrt(), "equities_readthrough")
        status["equities"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["equities"] = f"FAILED: {str(exc)[:50]}"
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
# Styles for the research-note layout
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
h2 {
  color: var(--sage) !important; font-weight: 500 !important;
  font-size: 1.5rem !important;
  margin-top: 2.4rem !important; margin-bottom: 0.5rem !important;
}
h3 { font-style: italic; font-weight: 400 !important; }
.lede { font-size: 1.02rem; line-height: 1.65; color: #D9E7DD; max-width: 62rem; }
.lede em { color: var(--vermillion); font-style: normal; }
.keybox { border: 1px solid var(--hairline); background: var(--green-raised);
          padding: 0.8rem 1.1rem; border-radius: 2px; max-width: 62rem; }
.notemeta { font-family: Georgia, serif; color: var(--sage-muted);
            font-size: 0.9rem; letter-spacing: 0.04em; margin-bottom: 0.2rem; }
.source { color: var(--sage-muted); font-size: 0.78rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛢️ GeoMacro Research")
    st.caption("**Coverage:** the US–Iran war and the oil complex. A structural "
               "escalation model (P) marked against market pricing (Q); the "
               "variant view is the product. **Research framework — not "
               "financial advice.**")
    derived = scorecard_derived(lake_key)
    st.caption(f"Framework weight is set mechanically by the thesis scorecard "
               f"(§III): currently **{derived['strength']:.1f}** on a 0–4 scale "
               f"(thesis fit {derived['M']:.0%}). There is no manual override: "
               "if incoming data stops confirming the framework, its weight in "
               "the forecast declines automatically.")
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


@st.cache_data(ttl=3600, show_spinner=False)
def brent_spot_history(_lake_key: str) -> pd.DataFrame:
    """FRED Europe Brent spot (DCOILBRENTEU), keyless — the series behind the
    war-premium decomposition, displayed rather than merely asserted."""
    import io
    import requests
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": "DCOILBRENTEU", "cosd": "2025-07-01"},
                     timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "brent"]
    df["date"] = pd.to_datetime(df["date"])
    df["brent"] = pd.to_numeric(df["brent"], errors="coerce")
    return df.dropna().set_index("date")


# --------------------------------------------------------------------------- #
# Masthead
# --------------------------------------------------------------------------- #
st.markdown(f"""
<p class="masthead-kicker">GeoMacro Research · Global Macro · Energy & Geopolitics</p>
<h1 class="masthead-title">The Pricing of Escalation</h1>
<p class="masthead-sub">Iran, the Strait of Hormuz, and the term structure of a war</p>
<p class="notemeta">{dt.date.today():%B %d, %Y} · Stance: <b>persistence over
resolution</b>, expressed at the back of the curve and always hedged for the
deal · Horizon: 3–6 months · All calls public and machine-graded (§IX)</p>
""", unsafe_allow_html=True)

# =========================================================================== #
# I. INVESTMENT SUMMARY
# =========================================================================== #
st.markdown("## I. Investment summary")

f3 = reg["forecasts"]["3m"]
t = reg["touch"]
p_s2plus = sum(f3[2:5])
try:
    fd_h = read_latest("fundamentals").iloc[-1]
    war_prem = float(fd_h["war_premium"])
    fair = float(fd_h["fundamentals_fair"])
    spot_fred = float(fd_h["brent_fred"])
    kept = float(fd_h["frac_premium_persisting"]) if pd.notna(
        fd_h.get("frac_premium_persisting")) else None
except Exception:  # noqa: BLE001
    war_prem = fair = spot_fred = kept = None
mdl_dec = reg["model_cdf"].get("2026-12-31")
try:
    pm_ = read_latest("predmkt_panel")
    hz_ = pm_[(pm_["family"] == "hormuz_normalize")
              & (pm_["question"].str.contains("December", na=False))]
    mkt_dec = float(hz_.iloc[0]["yes_prob"])
except Exception:  # noqa: BLE001
    mkt_dec = None

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Base case: the grind persists", f"{p_s2plus:.0%}",
          "probability the strait remains disrupted at 3 months", delta_color="off")
k2.metric("Shape of the risk", f"{f3[3]:.0%} vs {f3[4]:.0%}",
          "Gulf-infrastructure widening vs all-out escalation", delta_color="off")
k3.metric("Ceasefire risk is episodic", f"{t.get('p_visit_s5_3m', 0):.0%}",
          f"probability of a truce attempt; {f3[5]:.0%} that it durably holds",
          delta_color="off")
if war_prem is not None:
    k4.metric("Geopolitical premium in Brent", f"${war_prem:+.0f}/bbl",
              f"{kept:.0%} retained by the 12M contract" if kept else "",
              delta_color="off")
if mdl_dec is not None and mkt_dec is not None:
    k5.metric("Hormuz normal by Dec-31", f"P {mdl_dec:.0%} · Q {mkt_dec:.0%}",
              "our estimate vs prediction-market consensus", delta_color="off")

st.markdown(f"""
<p class="lede">
<b>Our view.</b> We expect the conflict to persist through the investment
horizon and to broaden laterally — toward Gulf critical infrastructure and
third-country basing — rather than escalate vertically into sustained
theater-wide war. We assign {p_s2plus:.0%} to continued disruption at three
months against {f3[4]:.0%} to durable all-out escalation and {f3[5]:.0%} to a
ceasefire that holds. Truce announcements are likely within the horizon
({t.get('p_visit_s5_3m', 0):.0%}) but are, in our framework, <em>episodes
rather than terminal events</em>: both 2026 precedents decayed inside a month.
</p>
<p class="lede">
<b>Where we differ from consensus.</b> Consensus, as expressed in market
pricing, has already capitulated on near-term reopening; we see no residual
edge in the front. Our variant view sits in three places. <b>(1) Duration:</b>
the 12-month Brent contract retains only ~{kept:.0%} of the current
${war_prem:+.0f}/bbl geopolitical premium — the curve is underwriting a
resolution our framework does not expect on that timeline. <b>(2) The lateral
axis:</b> infrastructure risk to Gulf water, power, and export capacity is
priced in European gas (TTF) but is essentially absent from oil instruments —
a cross-market inconsistency. <b>(3) Resolution durability:</b> the oil
options market pays for downside consistent with a durable settlement;
prediction markets price a final deal in single digits, and the empirical
half-life of 2026 ceasefires is ~3 weeks. We fade the durable-peace tail while
carrying cheap deal-shock protection against its arrival.
</p>
""", unsafe_allow_html=True)

st.subheader("Scenario framework, 3-month horizon")
if fair is not None:
    scen = pd.DataFrame([
        {"scenario": "Bear (for the premium): settlement holds",
         "probability": f"~{f3[5]:.0%}",
         "path for Brent": f"premium unwinds toward fundamentals (${fair:.0f}–{fair+10:.0f})",
         "positioning consequence": "deal-shock hedges pay; escalation longs stopped"},
        {"scenario": "Base: grind persists and widens",
         "probability": f"~{p_s2plus:.0%}",
         "path for Brent": f"range-bound near spot (${spot_fred-5:.0f}–{spot_fred+10:.0f}); back-of-curve repricing higher",
         "positioning consequence": "long deferred contracts / calendar expressions carry positively"},
        {"scenario": "Tail: vertical excursion (S4 touch)",
         "probability": f"~{t.get('p_visit_s4_3m', 0):.0%} touch",
         "path for Brent": "transient spike well above $110; mean-reverts as tempo proves unsustainable",
         "positioning consequence": "convexity (OTM calls) monetized into strength, not held"},
    ])
    st.dataframe(scen, hide_index=True,
                 column_config={"scenario": st.column_config.TextColumn(width="medium"),
                                "path for Brent": st.column_config.TextColumn(width="large")})
    st.markdown('<p class="source">Probabilities: this model (§IV). Price paths: '
                'scenario logic anchored to the premium decomposition (§V), not '
                'point forecasts — the fair-value anchor carries a wide error '
                'band.</p>', unsafe_allow_html=True)

# =========================================================================== #
# II. THE TAPE
# =========================================================================== #
st.markdown("## II. What the tape shows")
st.markdown("""
<p class="lede">
The controlling series for the entire complex is physical: <b>daily transit
calls at the Strait of Hormuz</b> — roughly a fifth of global oil supply. It
is also the settlement source for the prediction-market contracts we mark
against, which makes it the cleanest bridge between the physical war and its
price. Normal throughput is ~75 calls/day; 60 is the market's own resolution
threshold for "normalized."
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
    chart_df["normalization threshold (60)"] = 60.0
    st.line_chart(chart_df, height=300)
    st.markdown(f'<p class="source">Source: IMF PortWatch (satellite AIS), '
                f'latest {pw.index.max().date()}; publishes with ~5-day lag. '
                'The 2026 narrative left to right: normal into early March; '
                'closure; four months near zero; partial reopening on the June '
                'memorandum; re-closure on its collapse in July.</p>',
                unsafe_allow_html=True)
with right:
    st.subheader("Weekly regime classification")
    lab = labels.copy()
    lab["week"] = pd.to_datetime(lab["week"]).dt.date
    show = lab[["week", "state", "frac"]].tail(12).iloc[::-1]
    show["frac"] = (show["frac"] * 100).round(0).astype(int).astype(str) + "%"
    show.columns = ["week", "state", "transits"]
    st.dataframe(show, hide_index=True, height=300)
    st.markdown('<p class="source">Six-state taxonomy: S0 lull · S1 reciprocal '
                'strikes · S2 chokepoint interdiction · S3 Gulf-infrastructure '
                'war · S4 general war · S5 settlement. Classified from transit '
                'data plus the coded event stream.</p>', unsafe_allow_html=True)

st.subheader("Lateral escalation: distinct third-country fronts per week")
sc1, sc2 = st.columns([3, 2])
with sc1:
    try:
        hs = read_latest("horizontal_spread").copy()
        hs["week"] = pd.to_datetime(hs["week"])
        st.bar_chart(hs.set_index("week")[
            ["third_party_fronts", "proxy_active", "broad_hit"]], height=240)
        st.markdown('<p class="source">Source: LLM-coded event stream (frozen '
                    'prompts; inter-coder rung agreement 82%). Counts distinct '
                    'GCC/Iraq/Yemen targets; belligerent homelands and pure '
                    'maritime targets excluded.</p>', unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"spread index unavailable: {exc}")
with sc2:
    try:
        from src.features.horizontal_spread import spread_now
        sp = spread_now()
        st.metric("Active fronts this week", sp["third_party_fronts"],
                  sp["third_party_list"] or "—", delta_color="off")
        st.metric("Trailing 4-week intensity", f"{sp['trailing_4wk_index']:.1f}",
                  f"war average {sp['war_avg_index']:.1f}", delta_color="off")
        if sp["trailing_4wk_index"] > sp["war_avg_index"] * 1.3:
            st.warning("**Lateral escalation confirmed** — the July phase is "
                       "the broadest of the war, consistent with the framework's "
                       "central prediction.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"spread reading unavailable: {exc}")

with st.expander("Exhibit: recent coded events (the primary source material)"):
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev_show = ev.sort_values("date", ascending=False)[
        ["date", "actor", "rung", "target_type", "target_country", "severity", "action"]
    ].head(25)
    ev_show["date"] = ev_show["date"].dt.date
    st.dataframe(ev_show, hide_index=True, height=300)

# =========================================================================== #
# III. STRATEGIC FRAMEWORK
# =========================================================================== #
st.markdown("## III. The strategic framework — Mearsheimer, stated formally")
st.markdown("""
<p class="lede">
Our structural prior is John Mearsheimer's realist account of this war, and it
is worth stating in the discipline's own terms, because each proposition is
independently testable. The American campaign is an exercise in <b>coercion by
punishment</b> — an attempt to compel policy change by imposing costs from the
air without contesting control on the ground. The empirical record of
punishment campaigns against consolidated states is poor, and the theoretical
reason is the <b>balance of resolve</b>: Iran's survival interests are engaged
and America's are not, so the target's cost tolerance systematically exceeds
the coercer's political patience. Second, the United States lacks
<b>escalation dominance</b> — the capacity to raise the adversary's costs
faster than one's own at each rung of the ladder. Interceptor inventories
deplete on a ~35-month production cycle against missile output an order of
magnitude faster; the cost-exchange ratio of defense runs several multiples
against the defender. Third, Iran's <b>counter-escalation options are
asymmetric and lateral</b>: interdiction of commercial shipping and strikes on
Gulf critical infrastructure impose regional and global costs at trivial
marginal expense, along an axis where defense is thinnest. Fourth,
<b>war-termination theory</b>: wars end when both parties' expectations
converge on a settlement each leadership can survive domestically. Here,
mutual framing of every strike as proportionate retaliation generates
<b>audience costs</b> that keep the bargaining range empty — settlements can
be reached (April, June) but not sustained, because compliance is read as
capitulation. The structural prediction follows: <em>a punishment stalemate of
extended duration, expanding laterally toward the coercee's cheapest options,
punctuated by settlement episodes that fail</em>.
</p>
<p class="lede">
We hold this framework accountable rather than assume it. Each proposition is
scored continuously against the observed record below; the average sets the
framework's weight in the forecast. We note explicitly that the base rate cuts
against us — <b>the median interstate war terminates inside five months</b> —
so the persistence view is a deliberate, monitored deviation from the
reference class, licensed only while the scores hold.
</p>
""", unsafe_allow_html=True)

sc = scorecard_derived(lake_key)
m1, m2, m3 = st.columns(3)
m1.metric("Framework fit", f"{sc['M']:.0%}",
          "share of the thesis currently confirmed by data", delta_color="off")
m2.metric("Framework weight", f"{sc['strength']:.1f} / 4",
          "set mechanically by the fit — not discretionary", delta_color="off")
_tot = 23.0 + 68.0 + sc["strength"] * 88.0
m3.metric("Forecast composition",
          f"{sc['strength']*88/_tot:.0%} framework · {68/_tot:.0%} historical analogs · {23/_tot:.0%} this war",
          "analogs: Tanker War, 2019, 2020, 2024×2, 2025", delta_color="off")

LABELS = {
    "no_coercive_leverage": ("Punishment is not coercing",
                             "strike tempo has produced no Iranian concessions"),
    "deals_decay": ("Settlements decay",
                    "both 2026 ceasefires collapsed within ~3 weeks"),
    "asymmetric_escalation": ("Lateral escalation dominates",
                              "the war widens toward cheap third-country targets"),
    "face_lock": ("Audience costs bind",
                  "neither leadership can absorb visible capitulation"),
    "endurance_asymmetry": ("Resolve favors the target",
                            "Iran's endurance vs American political patience"),
}
for k, (label, gloss) in LABELS.items():
    s = sc["sub_scores"].get(k, 0.5)
    colA, colB = st.columns([1, 3])
    colA.markdown(f"**{label}**  \n{s:.2f}")
    with colB:
        st.progress(min(1.0, s))
        st.caption(f"{gloss} — {sc['details'].get(k, '')}")

st.subheader("The material basis: production, reserves, political tolerance")
ci = reg.get("covariates") or {}
g1, g2, g3 = st.columns(3)
g1.metric("Interceptor production deficit",
          f"{ci.get('munitions', {}).get('production_gap', 0):.0f} : 1",
          "Iranian missile output vs US interceptor output, per month",
          delta_color="off")
try:
    from src.features.economic import readings as _econ_r
    _e = _econ_r()
    g2.metric("Iranian fiscal runway", f"{_e['iran_runway_days']:.0f} days"
              if _e.get("iran_runway_days") else "extended",
              "usable FX reserves against interdicted export revenue (IMF)",
              delta_color="off")
    g3.metric("US price-pain reading", f"{_e['us_oil_pain']:.0%}",
              "approval-relevant gasoline passthrough begins near $100 Brent",
              delta_color="off")
except Exception:  # noqa: BLE001
    pass
st.markdown("""
<p class="lede">
<b>Second-order implications.</b> The production deficit makes Gulf air
defense a wasting asset: every intercept is inventory unreplaceable inside the
horizon, which caps sustained vertical escalation and — beyond this theater —
draws down the same magazines that underwrite Pacific deterrence. The Iranian
fiscal runway implies settlement pressure builds on a months-not-years clock;
combined with binding audience costs, this reproduces the observed cycle:
settlement, decay, re-escalation. <b>Third-order:</b> repetition of that cycle
conditions markets to discount truce headlines (successively smaller deal
rallies), while progressive interceptor depletion leaves desalination and
power assets — potable water for major Gulf cities — increasingly exposed. A
strike on desalination is accordingly our tripwire for regime reclassification
rather than an incremental data point.
</p>
""", unsafe_allow_html=True)

# =========================================================================== #
# IV. BASE-CASE PATH
# =========================================================================== #
st.markdown("## IV. The base-case path")
lcol, rcol = st.columns([3, 2])
with lcol:
    rows = []
    for h in ("2w", "1m", "3m", "6m"):
        for s, p in zip(STATES, reg["forecasts"][h]):
            rows.append({"horizon": h, "state": f"{s} {STATE_NAMES[s]}", "prob": p})
    fdf = pd.DataFrame(rows).pivot(index="state", columns="horizon", values="prob")
    st.dataframe(fdf[["2w", "1m", "3m", "6m"]].style.format("{:.0%}"), height=260)
    st.markdown('<p class="source">State-occupancy probabilities by horizon. '
                'The 3-month mass concentrates in S2/S3 — the interdiction-and-'
                'infrastructure grind.</p>', unsafe_allow_html=True)
with rcol:
    tm = reg["touch"]
    st.metric("P(touch S4 within 3m)", f"{tm.get('p_visit_s4_3m', 0):.0%}",
              f"occupancy at 3m only {f3[4]:.0%} — excursions, not residence",
              delta_color="off")
    st.metric("P(settlement episode within 3m)",
              f"{tm.get('p_visit_s5_3m', 0):.0%}",
              f"holding at 3m only {f3[5]:.0%}", delta_color="off")
    band = tm.get("race_band")
    st.metric("P(S4 precedes settlement)",
              f"{band['lo']:.0%}–{band['hi']:.0%}" if band else "n/a",
              "reported as an interval; the point estimate is not robust",
              delta_color="off")
st.markdown("""
<p class="lede">
The operative distinction is <b>touch versus occupancy</b>. The model assigns
material probability to brief S4 excursions — the war has produced two — and
minimal probability to sustained S4 residence, because the material
constraints in §III bind at maximum tempo. The same asymmetry governs
settlement: episodes are probable, durability is not. For positioning, both
tails are therefore <em>trading events within a persistent regime</em> — the
escalation tail is monetized into strength, the settlement tail is hedged and
then faded — rather than regime changes to re-underwrite.
</p>
""", unsafe_allow_html=True)

# =========================================================================== #
# V. OIL
# =========================================================================== #
st.markdown("## V. Oil: spot, curve, and the geopolitical premium")

st.subheader("Brent spot against the fundamentals anchor")
try:
    bh = brent_spot_history(lake_key)
    bchart = pd.DataFrame({"Brent spot": bh["brent"]})
    if fair is not None:
        bchart["fundamentals fair value"] = fair
    st.line_chart(bchart, height=280)
    st.markdown(f'<p class="source">Source: FRED series DCOILBRENTEU (Europe '
                f'Brent spot, latest ${spot_fred:.0f}). The fair-value line '
                f'(${fair:.0f}) is a pre-war OLS of Brent on copper, the broad '
                'dollar, 10y breakevens, real yields, and the S&amp;P (Jan-2024 '
                'to Jan-2026, R²≈0.7), evaluated at today\'s macro readings — '
                'i.e., our estimate of the price absent the war, which is low '
                'because copper says demand is soft. The vertical gap is the '
                'geopolitical premium. Treat the anchor as a decomposition '
                'device with a wide error band, not a target. Note: the front '
                'ICE future (§ curve below) trades a few dollars above FRED '
                'spot; each module is internally consistent.</p>',
                unsafe_allow_html=True)
except Exception as exc:  # noqa: BLE001
    st.warning(f"Brent history unavailable: {exc}")

qa, qb = st.columns(2)
with qa:
    st.subheader("The forward curve: how long is 'temporary'?")
    try:
        fc = read_latest("futures_curve")
        bzl = fc[fc["root"] == "BZ"].sort_values("contract_month")
        st.line_chart(bzl.set_index("contract_month")["close"], height=220)
        spread6 = float(bzl["close"].iloc[0] - bzl["close"].iloc[min(6, len(bzl) - 1)])
        st.markdown(f'<p class="source">Source: ICE Brent futures via Yahoo. '
                    f'Backwardation of ${spread6:.0f} front-to-6M prices the '
                    'disruption as substantially resolved within the year — '
                    'the single clearest expression of the duration view we '
                    'fade.</p>', unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"curve unavailable: {exc}")
with qb:
    st.subheader("Prediction markets: reopening odds by date")
    try:
        pm = read_latest("predmkt_panel")
        hz = pm[pm["family"] == "hormuz_normalize"].copy()
        hz = hz[(hz["end_date"].notna()) & (hz["volume"].fillna(0) > 100_000)]
        hz = hz[hz["end_date"] > dt.date.today().isoformat()].sort_values("end_date")
        cdf = hz[["end_date", "yes_prob"]].rename(
            columns={"end_date": "by", "yes_prob": "P(normalized)"})
        st.line_chart(cdf.set_index("by")["P(normalized)"], height=220)
        st.markdown('<p class="source">Source: Polymarket (Gamma API); '
                    'multi-million-dollar volume in the December contract; '
                    'settles on the same PortWatch series as §II.</p>',
                    unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"prediction markets unavailable: {exc}")

st.subheader("Premium decomposition: which war is priced, where")
try:
    fd = read_latest("fundamentals").iloc[-1]
    pr = read_latest("premia").iloc[-1]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Geopolitical premium", f"${fd['war_premium']:+.0f}/bbl",
              f"spot ${fd['brent_fred']:.0f} against fair ${fd['fundamentals_fair']:.0f}",
              delta_color="off")
    q2.metric("Retained at 12M", f"{fd['frac_premium_persisting']:.0%}"
              if pd.notna(fd.get("frac_premium_persisting")) else "n/a",
              "balance priced to decay along the curve", delta_color="off")
    q3.metric("TTF war premium", f"{pr['gas_war_premium_proxy']:+.0%}",
              "European gas — the only venue pricing the lateral axis",
              delta_color="off")
    q4.metric("Gold since outbreak", f"{pr['gold_since_war']:+.0%}",
              "declining: classified as a supply event, not systemic",
              delta_color="off")
    st.markdown('<p class="source">Sources: FRED + model estimates; ICE TTF and '
                'Henry Hub via Yahoo vs pre-war means; COMEX gold. Copper '
                'functions as the demand counterfactual — it shares oil\'s '
                'macro drivers but not its war exposure. A gold rally '
                'concurrent with escalation would mark reclassification from '
                'supply-local to systemic and is monitored as a tripwire '
                '(§VIII).</p>', unsafe_allow_html=True)
except Exception as exc:  # noqa: BLE001
    st.info(f"premium decomposition unavailable: {exc}")

try:
    rnd = read_latest("rnd")
    r = rnd[rnd["symbol"] == "USO"].iloc[-1]
    o1, o2, o3 = st.columns(3)
    o1.metric("Options: P(≈Brent > 100)", f"{r['p_up16']:.0%}")
    o2.metric("P(≈Brent < 75)", f"{r['p_dn13']:.0%}")
    o3.metric("At-the-money implied vol", f"{r['atm_iv']:.0%}")
    st.markdown('<p class="source">Source: USO/BNO option chains (delayed), '
                'Breeden–Litzenberger density; levels approximate, changes '
                'informative. The downside tail trades rich to the upside — '
                'the options market pays for durable resolution.</p>',
                unsafe_allow_html=True)
except Exception:  # noqa: BLE001
    pass

# =========================================================================== #
# VI. VARIANT VIEW & POSITIONING
# =========================================================================== #
# =========================================================================== #
# VI. EQUITIES READ-THROUGH
# =========================================================================== #
st.markdown("## VI. Equities read-through")
st.markdown("""
<p class="lede">
The macro thesis maps onto equities unevenly, and the tape — bucket returns
since the outbreak, through the June détente, and through the July
re-escalation, plus measured sensitivity on days with coded Gulf-infrastructure
(S3) strikes — separates the buckets that <em>say</em> they trade the war from
those that actually do.
</p>
""", unsafe_allow_html=True)

try:
    eq = read_latest("equities_readthrough").copy()
    has_tilt = "tilt_3m" in eq.columns and eq["tilt_3m"].notna().any()
    disp = pd.DataFrame({
        "bucket": eq["bucket"],
        "since outbreak": eq["since_war"].map("{:+.1%}".format),
        "June détente": eq["detente_jun"].map("{:+.1%}".format),
        "July re-escalation": eq["reescalation_jul"].map("{:+.1%}".format),
        "S3-day edge (per day)": eq["s3_sensitivity"].map("{:+.2%}".format),
    })
    if has_tilt and "tilt_lo" in eq.columns:
        disp["variant tilt (3m)"] = [
            f"{m:+.2%}  ({lo:+.1%} … {hi:+.1%})"
            for m, lo, hi in zip(eq["tilt_3m"], eq["tilt_lo"], eq["tilt_hi"])]
    elif has_tilt:
        disp["variant tilt (3m)"] = eq["tilt_3m"].map("{:+.2%}".format)
    st.dataframe(disp, hide_index=True)
    if has_tilt:
        r0 = eq.iloc[0]
        st.markdown(f"""
<p class="lede">
The first four columns are descriptive — each bucket's measured behavior. The
<b>variant tilt</b> is where our model meets the market: the expected
three-month return differential <em>if our war probabilities are right and the
market's are wrong</em> — each probability gap times the bucket's payoff when
that scenario materializes. Two gaps drive it: we assign
<b>{r0['p_res_model']:.0%}</b> to Hormuz normalizing by September against the
market's <b>{r0['p_res_mkt']:.0%}</b> (identical contract terms), and
<b>{r0['p_esc_model']:.0%}</b> to touching all-out war within three months
against an options-implied <b>{r0['p_esc_mkt']:.0%}</b>. The payoffs are
<b>triangulated</b>, not read off a single week: the median across this war's
own episodes, six analog-war episodes (Abqaiq 2019 through the 2025 twelve-day
war), and a third, mechanically independent leg — the bucket's Brent beta
applied to Brent's typical episode move — all market-adjusted. The
parenthetical band spans the legs. Triangulation mattered: single-window
estimates had flattered tankers and condemned defense; across nine episodes
those readings <em>reversed</em>.
</p>
""", unsafe_allow_html=True)
        with st.expander("Exhibit: the triangulation detail (per episode, market-adjusted)"):
            try:
                tri = read_latest("payoff_triangulation")
                piv = tri[tri["bucket"] != "brent"].pivot_table(
                    index=["scenario", "bucket"], columns="episode",
                    values="abnormal_return")
                st.dataframe(piv.style.format("{:+.1%}", na_rep="—"))
                st.markdown('<p class="source">Abnormal (S&amp;P-adjusted) '
                            'bucket returns per episode window. The July-2026 '
                            'defense selloff is visible as the outlier; in '
                            'every other escalation episode defense rallied. '
                            'Resolution episodes reliably cost tankers and '
                            'energy 6-10% market-adjusted.</p>',
                            unsafe_allow_html=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"triangulation detail unavailable: {exc}")
    st.markdown('<p class="source">Source: Yahoo daily closes, equal-weight '
                'buckets (tankers FRO/INSW/TNK/TRMD/NAT/STNG; defense ITA/PPA; '
                'energy XLE/XOP/OIH; Gulf KSA/UAE/QAT ETFs as the free proxy '
                'for Gulf sovereign risk — true CDS is paid data; airlines '
                'JETS). S3-day edge = mean return on coded S3-event days minus '
                'all other war days (n≈7 event days). Tilt caveats: episode '
                'payoffs come from single windows; the escalation-side market '
                'proxy is options-implied P(≈Brent&gt;100) — an imperfect object '
                'match, flagged rather than hidden. Indicative, not '
                'inferential.</p>', unsafe_allow_html=True)

    st.markdown("""
<p class="lede">
<b>Defense — the variant overweight, and triangulation's chief finding.</b>
On raw 2026 returns the bucket looks like dead money, and on the July window
alone it looked like a tactical avoid. Nine episodes say otherwise: July was
the outlier, and in every other escalation episode on record — Abqaiq,
Soleimani, October 2024, the 2026 outbreak — defense rallied, while losing
nothing in resolution weeks. With our escalation odds twenty points above the
options market and the §III restock arithmetic underneath (magazines rebuilt
at a 15:1 replacement gap in <em>every</em> scenario), defense carries the
only decisively positive tilt on the page. Structural and tactical now point
the same way; a week ago, on one window, they appeared opposed — which is the
argument for triangulating.
<br><br>
<b>Tankers — the all-weather story, demoted to neutral.</b> The +11% raw run
conceals what market-adjustment reveals: resolution episodes reliably cost
tankers 7–10% abnormal (April, June 2025), and our reopening odds run
<em>above</em> the market's. The escalation leg still pays, and the two legs
now roughly cancel. Long duration, yes — but no longer the page's best
expression of our variant, and hulls remain S2 targets.
<br><br>
<b>Energy equity — clean war-beta, wrong-way tilt.</b> Still the cleanest
tactical escalation instrument (sold the June deal, bought its collapse,
largest S3-day edge). But precisely because it loses 6–9% market-adjusted in
resolution weeks, our above-market reopening odds make the <em>variant</em>
tilt negative: the right vehicle for a pure escalation view is, on our own
probabilities, not a bucket to be overweight.
<br><br>
<b>Gulf markets — the lateral axis, priced in Riyadh.</b> A persistent ~10%
war discount, a negative S3-day edge, little détente recovery, and a modestly
negative tilt. The "unpriced lateral axis" argument finds its exception here —
unpriced in oil, priced in Gulf equity — and a widening-war view expresses as
a Gulf underweight rather than a fresh short.
<br><br>
<b>Airlines — the settlement long, roughly fairly priced against us.</b> The
mirror-image bucket (up in détentes, down in escalations). Our higher
reopening odds argue for it; our higher escalation odds argue against; the
legs nearly cancel. It remains the natural hedge leg rather than a view.
</p>
""", unsafe_allow_html=True)
except Exception as exc:  # noqa: BLE001
    st.info(f"equities read-through unavailable: {exc}")

st.markdown("## VII. The variant view, quantified — and positioning")

st.subheader("Our probabilities against consensus, same contract terms")
try:
    pm = read_latest("predmkt_panel")
    hz = pm[pm["family"] == "hormuz_normalize"].copy()
    hz = hz[hz["end_date"].notna()]
    rows = []
    for by, mdl in reg["model_cdf"].items():
        sub = hz[hz["end_date"].astype(str).str.startswith(by[:7])]
        mkt = float(sub.sort_values("volume", ascending=False).iloc[0]["yes_prob"]) \
            if len(sub) else None
        rows.append({"Hormuz normal by": by, "our estimate": f"{mdl:.0%}",
                     "consensus (Polymarket)": f"{mkt:.0%}" if mkt is not None else "—",
                     "variant": f"{mdl - mkt:+.0%}" if mkt is not None else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown('<p class="source">Identical resolution criteria on both '
                'columns (PortWatch 7-day average ≥ 60). Every row is a '
                'standing entry in the graded ledger (§IX).</p>',
                unsafe_allow_html=True)
except Exception as exc:  # noqa: BLE001
    st.info(f"variant table unavailable: {exc}")

st.markdown("""
<div class="keybox">
<b>Reading the signal sleeve.</b> 🟢 <b>LONG</b>: the expression profits if
persistence/escalation exceeds market pricing. 🔴 <b>SHORT</b>: profits if
resolution arrives faster than priced. ⚪ <b>FLAT</b>: no exploitable variant —
itself a finding. <b>Conviction</b> grades the evidential support for the
signal's logic (robustness-test survival, sample depth), not expected return;
low conviction means monitor, never size. House rule: every escalation-long
carries deal-shock protection, because settlement risk here is an overnight
gap — April 7 repriced from maximal escalation rhetoric to a ceasefire inside
a session.
</div>
""", unsafe_allow_html=True)

try:
    from src.alpha.signals import compute_all
    dirmap = {-1: "🔴 SHORT", 0: "⚪ FLAT", 1: "🟢 LONG"}
    for s in compute_all():
        with st.expander(f"{dirmap[s['direction']]}  **{s['signal']}**  "
                         f"(conviction: {s['confidence']})", expanded=False):
            st.write(s["rationale"])
            for k, v in (s["value"] or {}).items():
                if isinstance(v, dict):
                    v = v.get("yes_prob", v)
                st.markdown(f"- `{k}` = {v}")
            st.warning(s["caveats"])
except Exception as exc:  # noqa: BLE001
    st.error(f"signal computation failed: {exc}")

st.subheader("Settlement-shock stress (the book's governing risk)")
try:
    from src.alpha.stress import WINDOWS, _window_moves, book_pnl
    rows = []
    for name, (lo, hi) in WINDOWS.items():
        m = _window_moves(lo, hi)
        if not m:
            continue
        pnl = book_pnl(m)
        rows.append({"scenario (replayed from 2026 tapes)": name,
                     "Brent": f"{m['brent']:+.1%}", "tankers": f"{m['tankers']:+.1%}",
                     "hedged book": f"{pnl['TOTAL']:+.2%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown('<p class="source">A stylized persistence book replayed '
                'against the war\'s own settlement windows. Unhedged, the '
                'April tape costs ~-2.3%; hedged, ~-0.5%. The hedge is '
                'structural, not tactical.</p>', unsafe_allow_html=True)
except Exception as exc:  # noqa: BLE001
    st.error(f"stress unavailable: {exc}")

# =========================================================================== #
# VII. CATALYSTS & TRIPWIRES
# =========================================================================== #
st.markdown("## VIII. Catalysts and tripwires")
st.markdown("""
<p class="lede">
<b>Upside catalysts for the persistence view:</b> a strike on Gulf
desalination or power (regime reclassification toward sustained S3 — our
primary tripwire); interceptor-rationing reports from Gulf capitals;
Polymarket December odds fading below ~45%. <b>Falsifiers:</b> a ceasefire
surviving past the six-week hazard peak identified in the ceasefire
literature; transits holding above 60 for two consecutive weeks (grades our
ledger entries against us); the scorecard's settlement-decay or audience-cost
scores breaking below 0.5, which mechanically de-weights the framework.
<b>Systemic reclassification:</b> gold rallying concurrent with escalation —
the market abandoning its supply-local read — would dominate every other
signal on this page and warrants immediate de-risking of carry expressions.
</p>
""", unsafe_allow_html=True)

# =========================================================================== #
# VIII. RISKS, TRACK RECORD, METHODOLOGY
# =========================================================================== #
st.markdown("## IX. Risks to our view — and the graded record")
st.markdown("""
<p class="lede">
The honest core: this is one war, roughly twenty-four coded weeks. The
forecast therefore leans on the framework and on six historical analogs, and
§III displays exactly how much. Our published calls are the discipline —
timestamped, resolution criteria fixed in advance, machine-graded daily
against the same sources the markets settle on, Brier-scored as they resolve.
The ledger is append-only: no entry can be edited or withdrawn after commit.
</p>
""", unsafe_allow_html=True)

try:
    from src.report.calls import load, summary
    doc = load()
    s = summary(doc)
    st.markdown(f"**{s['n_calls']} public calls** since {s['first_call']} — "
                f"{s['n_open']} open, {s['n_resolved']} resolved"
                + (f", **Brier {s['brier']:.3f}**" if s["brier"] is not None else "")
                + " · [ledger](https://github.com/xiajason6-web/GeoMacro3/blob/main/calls/ledger.yaml)")
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

with st.expander("Methodology notes and standing limitations"):
    st.markdown("""
- **Sample**: n=1 war, ~24 coded weeks; posterior composition disclosed in §III.
- **Market data**: free/delayed feeds (ETF options; ~5-day transit lag) — we
  weight changes over levels throughout.
- **Shared settlement source**: our model and Polymarket both read PortWatch;
  the pipeline decomposes timing overlap from genuine disagreement before any
  variant is claimed.
- **Framework risk**: the persistence view is a monitored bet against the
  interstate-war base rate (median duration < 5 months); the scorecard is the
  monitoring instrument and de-weights the framework on contrary evidence.
- Every judgment parameter is registered and tagged
  ([ASSUMPTIONS.md](https://github.com/xiajason6-web/GeoMacro3/blob/main/ASSUMPTIONS.md));
  headline conclusions survive ±50% perturbation of all of them. **Not
  financial advice.**
""")
