"""Iran Escalation Pricing Model — Streamlit dashboard (GeoMacro3).

Streamlit Cloud entrypoint (repo root, per GeoMacro2 convention). On cold start
it pulls all keyless sources live (PortWatch, Polymarket, yfinance), lands them
in the ephemeral local lake, and runs the model chain. Cached 1h.

The LLM coder does NOT run here (no key in the cloud); coded history comes from
the frozen backfill YAML in git. Run `make refresh` locally to append live
codings and push updated vintages.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iran Escalation Model", page_icon="🛢️",
                   layout="wide")

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]
STATE_NAMES = {
    "S0": "Lull", "S1": "Tit-for-tat", "S2": "Chokepoint war",
    "S3": "Gulf infra war", "S4": "All-out war", "S5": "De-escalation/deal",
}


# --------------------------------------------------------------------------- #
# Data layer — live fetch, cached, landed into the (ephemeral) lake
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

    # optional live enrichment: only if an API key is present (Streamlit secret
    # or env). Without it, the cloud shows the hand-verified frozen spine; recent
    # weeks are then sparse (run `make refresh` locally for full live coding).
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

    from src.market_implied.predmkt import build_panel
    from src.common import write_partition as wp
    try:
        wp(build_panel(), "predmkt_panel")
    except Exception:  # noqa: BLE001
        pass
    from src.features.horizontal_spread import weekly_spread
    try:
        wp(weekly_spread(), "horizontal_spread")
        status["horizontal_spread"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["horizontal_spread"] = f"FAILED: {str(exc)[:60]}"
    return {"status": status, "as_of": dt.datetime.utcnow().isoformat()}


@st.cache_data(ttl=3600, show_spinner="Running regime model...")
def regime(prior_strength: float, _lake_key: str) -> dict:
    from src.model.regime_markov import run
    r = run(prior_strength)
    return {
        "labels": r["labels"], "p0": r["p0"], "T": r["T"],
        "forecasts": {k: list(map(float, v)) for k, v in r["forecasts"].items()},
        "touch": r["touch"], "data_weight": r["data_weight"],
    }


lake = build_lake()
lake_key = lake["as_of"]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛢️ Iran Escalation Model")
    st.caption("P (OSINT regime model) vs Q (market-implied) on the US–Iran "
               "war, expressed through oil. **Research framework — not "
               "financial advice.**")
    st.caption("**Prior:** Mearsheimer *horizontal escalation* thesis — the "
               "war widens (Gulf infrastructure, basing) rather than climbs; "
               "no US escalation dominance, so all-out war is unsustainable and "
               "decays back to the grind. S3 is the attractor.")
    prior_strength = st.slider(
        "Prior strength", 0.0, 3.0, 1.0, 0.25,
        help="0 = let the ~20 weeks of data speak alone (several transition "
             "rows go degenerate at n this small); 0.26 = 50/50 data-prior "
             "crossover; 1 = prior as written; 3 = prior-dominated. A "
             "conclusion that only appears at high strength is prior-driven — "
             "distrust it.")
    st.divider()
    st.caption("**Data status** (cached 1h)")
    for k, v in lake["status"].items():
        icon = "🟢" if "FAIL" not in str(v) else "🔴"
        st.caption(f"{icon} {k}: {v}")
    if st.button("Force refresh"):
        build_lake.clear()
        st.rerun()

try:
    reg = regime(prior_strength, lake_key)
except Exception as exc:  # noqa: BLE001 — surface the real error; cloud redacts it
    st.error(f"Regime model failed: {type(exc).__name__}: {exc}")
    st.stop()
labels = reg["labels"]
cur = labels.iloc[-1]

# --------------------------------------------------------------------------- #
# Header metrics
# --------------------------------------------------------------------------- #
st.title("Iran Escalation Pricing — P vs Q")

from src.common import read_latest  # noqa: E402

c1, c2, c3, c4, c5 = st.columns(5)
state_now = max(zip(reg["p0"], STATES))[1]
c1.metric("Regime", f"{state_now} {STATE_NAMES[state_now]}",
          f"{max(reg['p0']):.0%} prob")
c2.metric("Hormuz 7dMA", f"{cur['ma7']:.1f} calls/d",
          f"{cur['frac']:.0%} of baseline", delta_color="off")
try:
    px = read_latest("prices")
    bz = px[px["ticker"] == "BZ=F"].sort_values("obs_date")
    brent_now = float(bz["close"].iloc[-1])
    brent_prev = float(bz["close"].iloc[-6])
    c3.metric("Brent", f"${brent_now:.2f}", f"{brent_now/brent_prev-1:+.1%} 5d")
except Exception:  # noqa: BLE001
    c3.metric("Brent", "n/a")
try:
    pm = read_latest("predmkt_panel")
    hz = pm[(pm["family"] == "hormuz_normalize")
            & (pm["question"].str.contains("December", na=False))]
    c4.metric("Mkt P(normal by Dec 31)", f"{float(hz.iloc[0]['yes_prob']):.0%}")
except Exception:  # noqa: BLE001
    c4.metric("Mkt P(normal by Dec 31)", "n/a")
c5.metric("P(touch S4 before S5)", f"{reg['touch']['p_touch_s4_before_s5']:.0%}",
          f"{reg['data_weight']:.0%} data-weighted", delta_color="off")

tab_state, tab_pq, tab_signals, tab_stress, tab_about = st.tabs(
    ["📊 State & transits", "⚖️ P vs Q", "🚨 Signals", "🕊️ Peace-shock stress", "📖 About"])

# --------------------------------------------------------------------------- #
with tab_state:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Strait of Hormuz daily transits")
        pw = read_latest("portwatch").copy()
        pw["obs_date"] = pd.to_datetime(pw["obs_date"])
        pw = pw[pw["obs_date"] >= "2026-01-01"].set_index("obs_date")
        chart_df = pd.DataFrame({
            "daily": pd.to_numeric(pw["n_total"], errors="coerce"),
        })
        chart_df["7d MA"] = chart_df["daily"].rolling(7).mean()
        chart_df["normal threshold (60)"] = 60.0
        st.line_chart(chart_df, height=320)
        st.caption(f"Latest observation {pw.index.max().date()} "
                   f"(~{(dt.date.today() - pw.index.max().date()).days}d lag — "
                   "PortWatch publishes late; this lag is why P-Q gaps get "
                   "decomposed before being called alpha).")
    with right:
        st.subheader("Weekly regime labels")
        lab = labels.copy()
        lab["week"] = pd.to_datetime(lab["week"]).dt.date
        show = lab[["week", "state", "ma7", "frac"]].tail(14).iloc[::-1]
        show.columns = ["week", "state", "7dMA", "% baseline"]
        show["% baseline"] = (show["% baseline"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(show, hide_index=True, height=320)

    st.subheader("Horizontal escalation — the war widening (Mearsheimer's asymmetric options)")
    sc1, sc2 = st.columns([3, 2])
    with sc1:
        try:
            hs = read_latest("horizontal_spread").copy()
            hs["week"] = pd.to_datetime(hs["week"])
            st.bar_chart(hs.set_index("week")[["third_party_fronts", "proxy_active", "broad_hit"]],
                         height=260)
            st.caption("Third-party fronts = distinct GCC/Iraq/Yemen/etc. targets "
                       "struck per week (excludes Iran/Israel/US homelands). This "
                       "is the S3 axis — the escalation dimension **no market "
                       "instrument prices** (alpha #2).")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"spread index unavailable: {exc}")
    with sc2:
        try:
            from src.features.horizontal_spread import spread_now
            sp = spread_now()
            st.metric("Spread index (this week)", sp["spread_index"],
                      f"trailing-4wk {sp['trailing_4wk_index']:.1f} vs war-avg "
                      f"{sp['war_avg_index']:.1f}", delta_color="off")
            st.metric("Fronts this week", sp["third_party_fronts"],
                      sp["third_party_list"] or "—", delta_color="off")
            if sp["p_s3_persists_next_week"] is not None:
                st.metric("S3 recurs next week | S3 now",
                          f"{sp['p_s3_persists_next_week']:.0%}",
                          "v2 predicts high (attractor)", delta_color="off")
            if sp["trailing_4wk_index"] > sp["war_avg_index"] * 1.3:
                st.warning("**War widening** — recent spread well above the war "
                           "average. Live confirmation of the horizontal-"
                           "escalation prior; the scorecard slider ticks up.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"spread reading unavailable: {exc}")

    st.subheader("Coded escalation events (frozen backfill + live appends)")
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev_show = ev.sort_values("date", ascending=False)[
        ["date", "actor", "rung", "target_type", "target_country", "severity", "action"]
    ].head(25)
    ev_show["date"] = ev_show["date"].dt.date
    st.dataframe(ev_show, hide_index=True, height=300)

# --------------------------------------------------------------------------- #
with tab_pq:
    left, right = st.columns(2)
    with left:
        st.subheader("P — model state forecasts")
        rows = []
        for h in ("2w", "1m", "3m", "6m"):
            for s, p in zip(STATES, reg["forecasts"][h]):
                rows.append({"horizon": h, "state": f"{s} {STATE_NAMES[s]}", "prob": p})
        fdf = pd.DataFrame(rows).pivot(index="state", columns="horizon", values="prob")
        fdf = fdf[["2w", "1m", "3m", "6m"]]
        st.dataframe(fdf.style.format("{:.0%}"), height=260)
        mwks = reg["touch"]["median_weeks_to_s5"]
        st.caption(f"Horizontal-escalation prior, posterior = "
                   f"{reg['data_weight']:.0%} data / "
                   f"{1-reg['data_weight']:.0%} prior at strength={prior_strength}. "
                   + (f"Median weeks to S5 when reached: {mwks:.0f}."
                      if mwks is not None else "S5 not reached in simulation."))
    with right:
        st.subheader("Q — market normalization CDF")
        try:
            pm = read_latest("predmkt_panel")
            hz = pm[pm["family"] == "hormuz_normalize"].copy()
            hz = hz[(hz["end_date"].notna()) & (hz["volume"].fillna(0) > 100_000)]
            hz = hz[hz["end_date"] > dt.date.today().isoformat()].sort_values("end_date")
            cdf = hz[["end_date", "yes_prob", "volume"]].rename(
                columns={"end_date": "by", "yes_prob": "P(normalized)"})
            st.line_chart(cdf.set_index("by")["P(normalized)"], height=220)
            st.dataframe(
                cdf.assign(volume=cdf["volume"].map(lambda v: f"${v:,.0f}"))
                   .style.format({"P(normalized)": "{:.1%}"}),
                hide_index=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"prediction-market leg unavailable: {exc}")

    st.divider()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Futures curve (revealed duration)")
        try:
            fc = read_latest("futures_curve")
            bzl = fc[fc["root"] == "BZ"].sort_values("contract_month")
            st.line_chart(bzl.set_index("contract_month")["close"], height=220)
            spread6 = float(bzl["close"].iloc[0] - bzl["close"].iloc[min(6, len(bzl)-1)])
            st.caption(f"Front-to-6M: {spread6:+.2f} — backwardation means the "
                       "market prices the disruption as temporary.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"curve unavailable: {exc}")
    with cc2:
        st.subheader("Options-implied tails (USO RND)")
        try:
            rnd = read_latest("rnd")
            r = rnd[rnd["symbol"] == "USO"].iloc[-1]
            t1, t2, t3 = st.columns(3)
            t1.metric("P(~Brent>100)", f"{r['p_up16']:.0%}")
            t2.metric("P(~Brent<75)", f"{r['p_dn13']:.0%}")
            t3.metric("ATM IV", f"{r['atm_iv']:.0%}")
            st.caption(f"Expiry {r['expiry']} ({r['days']}d). ETF options are a "
                       "blurry lens — trust changes over levels. Downside "
                       "currently priced heavier than upside: the market fears "
                       "the deal-drop more than escalation.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"RND unavailable: {exc}")

# --------------------------------------------------------------------------- #
with tab_signals:
    st.subheader("Live signals A1–A6")
    try:
        from src.alpha.signals import compute_all
        dirmap = {-1: "🔴 SHORT", 0: "⚪ FLAT", 1: "🟢 LONG"}
        for s in compute_all():
            with st.expander(f"{dirmap[s['direction']]}  **{s['signal']}**  "
                             f"(confidence: {s['confidence']})",
                             expanded=s["direction"] != 0):
                st.write(s["rationale"])
                st.json(s["value"])
                st.warning(s["caveats"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"signal computation failed: {exc}")
    st.caption("A5 is mandatory whenever A1 is on. Falsification verdicts "
               "(walk-forward + ±50% perturbation): A3 SURVIVES (94% of grid "
               "positive), A6 SURVIVES (100%, n=5 — tiny). Survival earns "
               "monitoring, not capital.")

# --------------------------------------------------------------------------- #
with tab_stress:
    st.subheader("Stylized A1+A5 book vs this war's own tapes")
    st.caption("The trade's true risk is the overnight deal. Windows measured "
               "from actual 2026 episodes.")
    try:
        from src.alpha.stress import WINDOWS, _window_moves, book_pnl, BOOK
        st.json(BOOK)
        rows = []
        for name, (lo, hi) in WINDOWS.items():
            m = _window_moves(lo, hi)
            if not m:
                continue
            pnl = book_pnl(m)
            rows.append({"scenario": name,
                         "brent": f"{m['brent']:+.1%}",
                         "tankers": f"{m['tankers']:+.1%}",
                         "book TOTAL": f"{pnl['TOTAL']:+.2%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Rule: if the deal-shock TOTAL is worse than −2%, escalation "
                   "sleeves are oversized relative to the hedge.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"stress unavailable: {exc}")

# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown("""
### What this is
An OSINT-driven escalation state model (**P**) for the 2026 US–Iran war vs
what markets price (**Q**), with alpha defined as their divergence — expressed
through oil and oil-adjacent instruments. Mearsheimer's structural argument
(airpower doesn't coerce; no defined win condition; asymmetric escalation
options; face-lock) enters as **overrulable Dirichlet priors** on the regime
transition matrix — the sidebar slider shows exactly how much work the prior
is doing.

### Honest limitations (read before believing any number)
1. **n = 1 war, ~20 weeks.** The posterior is mostly prior. Outputs are
   decision-support, not a signal factory.
2. **Q is partly unobservable with free data** — delayed ETF options, thin
   prediction markets. Trust changes over levels.
3. **P and Q share PortWatch** (it is both the model's S2 input and
   Polymarket's resolution source, with ~5d publication lag) — divergences are
   decomposed into mechanical-lag vs real-belief-gap before being called alpha.
4. **The June deal decayed in 3 weeks.** The peace-shock hedge (A5) is
   mandatory, not optional — and the stress tab shows why.

Repo: [xiajason6-web/GeoMacro3](https://github.com/xiajason6-web/GeoMacro3) ·
Daily pipeline: `make refresh` (local, includes the LLM event coder).
**Not financial advice.**
""")
