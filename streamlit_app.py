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
/* section headings: sage double-stripe motif (the cover's L) */
h3 {
  color: var(--sage) !important; font-weight: 500 !important;
  font-size: 1.25rem !important;
  border-left: 3px solid var(--sage); padding-left: 14px;
  box-shadow: inset 8px 0 0 -6px var(--sage);
  margin-top: 1.4rem !important; margin-bottom: 0.4rem !important;
  padding-top: 0.1rem; padding-bottom: 0.1rem;
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
# Sidebar — controls only (status lives in the Trust tab)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛢️ Iran Escalation Model")
    st.caption("A structural war model (**P**) vs market pricing (**Q**); alpha "
               "is their divergence. **Research framework — not financial advice.**")
    derived = scorecard_derived(lake_key)
    auto = st.toggle(
        f"Auto prior strength (scorecard → {derived['strength']:.1f})", value=True,
        help="Derive the Mearsheimer-prior weight from the live scorecard "
             "(Trust tab). Off = set it manually.")
    manual_strength = st.slider(
        "Prior strength", 0.0, 4.0, 1.0, 0.25, disabled=auto,
        help="0 = data only (degenerate at n this small); 0.26 = 50/50 "
             "crossover; ~3 = current scorecard; 4 = empirical-Bayes ceiling.")
    prior_strength = derived["strength"] if auto else manual_strength
    st.caption(f"prior_strength = **{prior_strength:.2f}** "
               f"({'derived' if auto else 'manual'})")
    st.divider()
    if st.button("↻ Force data refresh"):
        build_lake.clear()
        st.rerun()
    st.caption("[GitHub](https://github.com/xiajason6-web/GeoMacro3) · "
               "data cached 1h · daily CI-graded "
               "[calls ledger](https://github.com/xiajason6-web/GeoMacro3/blob/main/calls/ledger.yaml)")

try:
    reg = regime(prior_strength, lake_key)
except Exception as exc:  # noqa: BLE001 — surface the real error; Cloud redacts it
    st.error(f"Regime model failed: {type(exc).__name__}: {exc}")
    st.stop()
labels = reg["labels"]
cur = labels.iloc[-1]

from src.common import read_latest  # noqa: E402

# --------------------------------------------------------------------------- #
# Masthead + the five numbers that matter
# --------------------------------------------------------------------------- #
st.markdown("""
<p class="masthead-kicker">GeoMacro · a structural war model against the market</p>
<h1 class="masthead-title">The Pricing of Escalation</h1>
<p class="masthead-sub">Iran, the Strait of Hormuz, and what the oil market believes</p>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
state_now = max(zip(reg["p0"], STATES))[1]
c1.metric("Regime now", f"{state_now} · {STATE_NAMES[state_now]}",
          f"{max(reg['p0']):.0%} probability", delta_color="off")
c2.metric("Hormuz transits", f"{cur['frac']:.0%} of normal",
          f"7dMA {cur['ma7']:.0f} calls/day", delta_color="off")
try:
    px = read_latest("prices")
    bz = px[px["ticker"] == "BZ=F"].sort_values("obs_date")
    brent_now = float(bz["close"].iloc[-1])
    fd_head = read_latest("fundamentals").iloc[-1]
    c3.metric("Brent", f"${brent_now:.0f}",
              f"${fd_head['war_premium']:+.0f} war premium", delta_color="off")
except Exception:  # noqa: BLE001
    c3.metric("Brent", "n/a")
try:
    pm = read_latest("predmkt_panel")
    hz = pm[(pm["family"] == "hormuz_normalize")
            & (pm["question"].str.contains("December", na=False))]
    mkt_dec = float(hz.iloc[0]["yes_prob"])
    mdl_dec = reg["model_cdf"].get("2026-12-31")
    c4.metric("Normal by Dec 31", f"P {mdl_dec:.0%} · Q {mkt_dec:.0%}",
              "model vs market", delta_color="off")
except Exception:  # noqa: BLE001
    c4.metric("Normal by Dec 31", "n/a")
c5.metric("Mearsheimer-fit", f"{derived['M']:.0%}",
          f"prior strength {derived['strength']:.1f}", delta_color="off")

tab_now, tab_p, tab_q, tab_div, tab_trust = st.tabs(
    ["📍 Now", "🔮 Forecast · P", "💹 Market · Q", "⚡ Divergence", "🧭 Trust"])

# =========================================================================== #
with tab_now:
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
        st.caption(f"Latest {pw.index.max().date()} — PortWatch publishes ~5d late.")
    with right:
        st.subheader("Weekly regime")
        lab = labels.copy()
        lab["week"] = pd.to_datetime(lab["week"]).dt.date
        show = lab[["week", "state", "frac"]].tail(12).iloc[::-1]
        show["frac"] = (show["frac"] * 100).round(0).astype(int).astype(str) + "%"
        show.columns = ["week", "state", "transits"]
        st.dataframe(show, hide_index=True, height=300)

    st.subheader("Horizontal spread — is the war widening?")
    sc1, sc2 = st.columns([3, 2])
    with sc1:
        try:
            hs = read_latest("horizontal_spread").copy()
            hs["week"] = pd.to_datetime(hs["week"])
            st.bar_chart(hs.set_index("week")[
                ["third_party_fronts", "proxy_active", "broad_hit"]], height=240)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"spread index unavailable: {exc}")
    with sc2:
        try:
            from src.features.horizontal_spread import spread_now
            sp = spread_now()
            st.metric("Fronts this week", sp["third_party_fronts"],
                      sp["third_party_list"] or "—", delta_color="off")
            st.metric("Trailing-4wk spread", f"{sp['trailing_4wk_index']:.1f}",
                      f"war average {sp['war_avg_index']:.1f}", delta_color="off")
            if sp["trailing_4wk_index"] > sp["war_avg_index"] * 1.3:
                st.warning("**War widening** — spread well above the war average.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"spread reading unavailable: {exc}")
    with st.expander("ⓘ What this measures"):
        st.markdown("Distinct third-country targets struck per week (GCC/Iraq/"
                    "Yemen; belligerent homelands and maritime chokepoint targets "
                    "excluded). This is the S3 axis — Mearsheimer's horizontal "
                    "escalation, and the dimension no oil instrument prices.")

    with st.expander("📋 Recent coded events (LLM-coded, frozen prompts)"):
        ev = read_latest("coded_events").copy()
        ev["date"] = pd.to_datetime(ev["date"])
        ev_show = ev.sort_values("date", ascending=False)[
            ["date", "actor", "rung", "target_type", "target_country", "severity", "action"]
        ].head(25)
        ev_show["date"] = ev_show["date"].dt.date
        st.dataframe(ev_show, hide_index=True, height=300)

# =========================================================================== #
with tab_p:
    st.subheader("State probabilities by horizon")
    rows = []
    for h in ("2w", "1m", "3m", "6m"):
        for s, p in zip(STATES, reg["forecasts"][h]):
            rows.append({"horizon": h, "state": f"{s} {STATE_NAMES[s]}", "prob": p})
    fdf = pd.DataFrame(rows).pivot(index="state", columns="horizon", values="prob")
    st.dataframe(fdf[["2w", "1m", "3m", "6m"]].style.format("{:.0%}"), height=260)

    st.subheader("Touch probabilities — visiting vs staying")
    t = reg["touch"]
    tm1, tm2, tm3, tm4 = st.columns(4)
    if t.get("p_visit_s4_3m") is not None:
        tm1.metric("Visit S4 within 3m", f"{t['p_visit_s4_3m']:.0%}",
                   f"6m {t['p_visit_s4_6m']:.0%}", delta_color="off")
        tm2.metric("Visit S5 within 3m", f"{t['p_visit_s5_3m']:.0%}",
                   f"6m {t['p_visit_s5_6m']:.0%}", delta_color="off")
    band = t.get("race_band")
    tm3.metric("S4 before S5", f"{band['lo']:.0%}–{band['hi']:.0%}" if band
               else f"~{t['p_touch_s4_before_s5']:.0%}",
               "range — level is knob-sensitive", delta_color="off")
    tm4.metric("Median weeks to S5",
               f"{t['median_weeks_to_s5']:.0f}" if t.get("median_weeks_to_s5") else "n/a",
               "when reached; deals then decay", delta_color="off")
    st.caption("Brief **excursions** into all-out war are likely; **sustained** "
               "all-out war is rare (visit ≫ occupancy). Deal episodes likely "
               "and short-lived — matching April and June.")

    st.subheader("Endurance gauges (move the forecast via M9)")
    ci = reg.get("covariates") or {}
    g1, g2, g3 = st.columns(3)
    g1.metric("Munitions p_a", f"{ci.get('p_a', 0):.2f}",
              f"production gap {ci.get('munitions', {}).get('production_gap', 0):.0f}:1 → S4 gate",
              delta_color="off")
    g2.metric("Spread p_c", f"{ci.get('p_c', 0):.2f}", "→ S3 pump", delta_color="off")
    g3.metric("Economic p_b", f"{ci.get('p_b', 0):.2f}", "→ S5 drift", delta_color="off")
    if ci and "static_forecasts" in reg:
        s3d = reg["forecasts"]["3m"][3] - reg["static_forecasts"]["3m"][3]
        s4d = reg["forecasts"]["3m"][4] - reg["static_forecasts"]["3m"][4]
        st.caption(f"Net effect vs static prior at 3m: **S3 {s3d:+.0%}, S4 {s4d:+.0%}** "
                   "— the war widens; the all-out tail shrinks.")
    with st.expander("ⓘ How P is built"):
        md = ci.get("mass_decomposition")
        st.markdown(
            "Conjugate Dirichlet–multinomial over weekly S0–S5 transitions, three "
            "voices: the Mearsheimer prior (weight set by the scorecard), six "
            "analog conflicts (Tanker War → 2025), and this war's observed weeks. "
            + (f"Current mass: **{md['prior']:.0%} prior / {md['analogs']:.0%} "
               f"analogs / {md['live_data']:.0%} live**. " if md else "")
            + "Endurance gauges multiply specific transition cells (bounded, "
              "calibrated where evidence exists — see ASSUMPTIONS.md). Effect "
              "sizes for S4 were calibrated DOWN after the empirical continuation "
              "check showed double-counting.")

# =========================================================================== #
with tab_q:
    left, right = st.columns(2)
    with left:
        st.subheader("Prediction markets — normalization odds")
        try:
            pm = read_latest("predmkt_panel")
            hz = pm[pm["family"] == "hormuz_normalize"].copy()
            hz = hz[(hz["end_date"].notna()) & (hz["volume"].fillna(0) > 100_000)]
            hz = hz[hz["end_date"] > dt.date.today().isoformat()].sort_values("end_date")
            cdf = hz[["end_date", "yes_prob", "volume"]].rename(
                columns={"end_date": "by", "yes_prob": "P(normalized)"})
            st.line_chart(cdf.set_index("by")["P(normalized)"], height=220)
            with st.expander("markets detail"):
                st.dataframe(
                    cdf.assign(volume=cdf["volume"].map(lambda v: f"${v:,.0f}"))
                       .style.format({"P(normalized)": "{:.1%}"}),
                    hide_index=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"prediction-market leg unavailable: {exc}")
    with right:
        st.subheader("Futures curve — revealed duration")
        try:
            fc = read_latest("futures_curve")
            bzl = fc[fc["root"] == "BZ"].sort_values("contract_month")
            st.line_chart(bzl.set_index("contract_month")["close"], height=220)
            spread6 = float(bzl["close"].iloc[0] - bzl["close"].iloc[min(6, len(bzl) - 1)])
            st.caption(f"Front-to-6M {spread6:+.2f}: backwardation = disruption "
                       "priced as temporary.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"curve unavailable: {exc}")

    st.subheader("What kind of war is priced, where")
    try:
        fd = read_latest("fundamentals").iloc[-1]
        pr = read_latest("premia").iloc[-1]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("War premium in Brent", f"${fd['war_premium']:+.0f}",
                  f"fair ${fd['fundamentals_fair']:.0f} + premium", delta_color="off")
        q2.metric("Kept at 12M by curve",
                  f"{fd['frac_premium_persisting']:.0%}"
                  if pd.notna(fd.get("frac_premium_persisting")) else "n/a",
                  "persistence already priced", delta_color="off")
        q3.metric("Gas war premium (S3)", f"{pr['gas_war_premium_proxy']:+.0%}",
                  f"TTF {pr['ttf_elevation']:+.0%} vs HH {pr['hh_elevation']:+.0%}",
                  delta_color="off")
        q4.metric("Gold since war", f"{pr['gold_since_war']:+.0%}",
                  "supply-local" if "supply-local" in str(pr["gold_regime"])
                  else "⚠ SYSTEMIC", delta_color="off")
    except Exception as exc:  # noqa: BLE001
        st.info(f"premium decomposition unavailable: {exc}")

    try:
        rnd = read_latest("rnd")
        r = rnd[rnd["symbol"] == "USO"].iloc[-1]
        o1, o2, o3 = st.columns(3)
        o1.metric("Options: P(~Brent>100)", f"{r['p_up16']:.0%}")
        o2.metric("P(~Brent<75)", f"{r['p_dn13']:.0%}")
        o3.metric("ATM implied vol", f"{r['atm_iv']:.0%}")
        st.caption("Downside priced heavier than upside — the options market "
                   "fears the deal-drop more than escalation.")
    except Exception:  # noqa: BLE001
        pass
    with st.expander("ⓘ How Q is read"):
        st.markdown(
            "War premium = Brent minus a pre-war OLS fair value (copper/USD/"
            "rates/S&P — copper is the demand twin that shares oil's drivers but "
            "not its war). Brent−WTI localizes the premium to the chokepoint; "
            "TTF-vs-HenryHub is the only instrument pricing the horizontal (S3) "
            "axis; gold classifies the war as supply-local vs systemic. ETF "
            "options are delayed — trust changes over levels.")

# =========================================================================== #
with tab_div:
    st.subheader("Head-to-head: model vs market on Hormuz normalization")
    try:
        pm = read_latest("predmkt_panel")
        hz = pm[pm["family"] == "hormuz_normalize"].copy()
        hz = hz[hz["end_date"].notna()]
        rows = []
        for by, mdl in reg["model_cdf"].items():
            sub = hz[hz["end_date"].astype(str).str.startswith(by[:7])]
            mkt = float(sub.sort_values("volume", ascending=False).iloc[0]["yes_prob"]) \
                if len(sub) else None
            rows.append({"normalized by": by, "model P": f"{mdl:.0%}",
                         "market Q": f"{mkt:.0%}" if mkt is not None else "—",
                         "P − Q": f"{mdl - mkt:+.0%}" if mkt is not None else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Positive P−Q = model more optimistic on reopening than the "
                   "market. These rows are live calls in the public ledger "
                   "(Trust tab) — they get graded.")
    except Exception as exc:  # noqa: BLE001
        st.info(f"head-to-head unavailable: {exc}")

    st.subheader("Signals")
    try:
        from src.alpha.signals import compute_all
        dirmap = {-1: "🔴 SHORT", 0: "⚪ FLAT", 1: "🟢 LONG"}
        for s in compute_all():
            with st.expander(f"{dirmap[s['direction']]}  **{s['signal']}**  "
                             f"(confidence: {s['confidence']})",
                             expanded=False):
                st.write(s["rationale"])
                for k, v in (s["value"] or {}).items():
                    if isinstance(v, dict):
                        v = v.get("yes_prob", v)
                    st.markdown(f"- `{k}` = {v}")
                st.warning(s["caveats"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"signal computation failed: {exc}")
    st.caption("A5 (deal hedge) is mandatory whenever A1 is on. A3 survived 94% "
               "of its perturbation grid; survival earns monitoring, not capital.")

    st.subheader("Peace-shock stress — the trade's true risk")
    try:
        from src.alpha.stress import WINDOWS, _window_moves, book_pnl, BOOK
        rows = []
        for name, (lo, hi) in WINDOWS.items():
            m = _window_moves(lo, hi)
            if not m:
                continue
            pnl = book_pnl(m)
            rows.append({"scenario": name, "brent": f"{m['brent']:+.1%}",
                         "tankers": f"{m['tankers']:+.1%}",
                         "book P&L": f"{pnl['TOTAL']:+.2%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        with st.expander("ⓘ Stylized book"):
            st.dataframe(pd.DataFrame([{"sleeve": k, "weight": v} for k, v in BOOK.items()]),
                         hide_index=True)
            st.markdown("Windows are this war's own tapes (June MOU, April "
                        "whiplash, July collapse). Rule: if the deal-shock row is "
                        "worse than −2%, escalation sleeves are oversized.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"stress unavailable: {exc}")

# =========================================================================== #
with tab_trust:
    st.subheader("Mearsheimer scorecard — is the war behaving as the thesis predicts?")
    sc = scorecard_derived(lake_key)
    m1, m2, m3 = st.columns(3)
    m1.metric("Mearsheimer-fit M", f"{sc['M']:.0%}", "fraction of restrictions confirmed",
              delta_color="off")
    m2.metric("Derived prior strength", f"{sc['strength']:.2f}",
              "drives the sidebar (auto mode)", delta_color="off")
    _tot = 23.0 + 68.0 + sc["strength"] * 88.0
    m3.metric("Posterior mass",
              f"{23/_tot:.0%} live · {68/_tot:.0%} analogs · {sc['strength']*88/_tot:.0%} prior",
              delta_color="off")
    LABELS = {
        "no_coercive_leverage": "No coercive leverage",
        "deals_decay": "Deals decay",
        "asymmetric_escalation": "Asymmetric / horizontal escalation",
        "face_lock": "Face-lock",
        "endurance_asymmetry": "Endurance asymmetry",
    }
    for k, label in LABELS.items():
        s = sc["sub_scores"].get(k, 0.5)
        colA, colB = st.columns([1, 3])
        colA.markdown(f"**{label}**  \n{s:.2f}")
        with colB:
            st.progress(min(1.0, s))
            st.caption(sc["details"].get(k, ""))
    with st.expander("ⓘ Why derive the prior weight this way"):
        st.markdown("Each restriction is graded from data (deal half-lives, "
                    "spread trajectory, cost asymmetry, face-lock, runways). High "
                    "M → trust the prior; if the war deviates (a deal holds), the "
                    "score falls and the prior stops dominating. It cannot "
                    "overstay. Weights are equal (judgment at n=1) — currently "
                    "moot because all sub-scores agree.")

    st.divider()
    st.subheader("📋 Public track record")
    try:
        from src.report.calls import load, summary
        doc = load()
        s = summary(doc)
        st.markdown(f"**{s['n_calls']} calls** since {s['first_call']} — "
                    f"{s['n_open']} open, {s['n_resolved']} resolved"
                    + (f", **Brier {s['brier']:.3f}**" if s["brier"] is not None else "")
                    + ". Append-only, git-timestamped, auto-graded daily "
                      "([ledger](https://github.com/xiajason6-web/GeoMacro3/blob/main/calls/ledger.yaml)).")
        rows = [{"made": c["made"], "p": f"{c['p']:.0%}",
                 "claim": c["claim"],
                 "status": c.get("outcome", "open")} for c in doc["calls"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"claim": st.column_config.TextColumn(
                         "claim", width="large")})
    except Exception as exc:  # noqa: BLE001
        st.info(f"ledger unavailable: {exc}")

    st.divider()
    st.subheader("Data status")
    cols = st.columns(3)
    for i, (k, v) in enumerate(lake["status"].items()):
        icon = "🟢" if "FAIL" not in str(v) else "🔴"
        cols[i % 3].caption(f"{icon} **{k}**: {v}")

    with st.expander("ⓘ Honest limitations — read before believing any number"):
        st.markdown("""
1. **n = 1 war** (~24 weekly transitions): the posterior is mostly prior +
   analogs, and says so above.
2. **Q is free/delayed data** — ETF options, ~5-day PortWatch lag. Trust
   changes over levels.
3. **P and Q share PortWatch** (model input AND Polymarket's resolution
   source) — divergences are decomposed before being called alpha.
4. **Deals decay** (April, June precedents) — the peace-shock hedge is
   mandatory, and the stress table shows why.

Full register: [ASSUMPTIONS.md](https://github.com/xiajason6-web/GeoMacro3/blob/main/ASSUMPTIONS.md)
— every judgment knob tagged founded / defensible / arbitrary, adversarially
swept. All headline conclusions survive ±50% perturbation.
""")
