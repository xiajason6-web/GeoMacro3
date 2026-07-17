"""Signals A1–A6 — live readings with explicit economic rationale (§7).

Each signal returns {value, direction, confidence, rationale, caveats}.
Direction: -1 short / 0 flat / +1 long (of the stated expression).
Confidence: qualitative {low, medium, high} — with n=1 war, none earn "high"
unless multiple independent legs agree.

These are RESEARCH readings, not orders. A5 (deal hedge) is mandatory whenever
A1 is on — enforced in portfolio.py, restated here.

Run:  python -m src.alpha.signals
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import read_latest
from src.model.regime_markov import run as run_regime, STATES


def _latest(source):
    try:
        return read_latest(source)
    except FileNotFoundError:
        return None


def a1_pq_divergence() -> dict:
    """P-Q on the S2-persistence question: model P(still S2+ at 3m) vs the
    market's normalization CDF read at the same horizon."""
    reg = run_regime()
    p_s2plus_3m = float(sum(reg["forecasts"]["3m"][2:5]))  # S2+S3+S4
    pm = _latest("predmkt_panel")
    q_normal_sep = None
    if pm is not None:
        hz = pm[(pm["family"] == "hormuz_normalize")
                & (pm["question"].str.contains("September", case=False, na=False))]
        if len(hz):
            q_normal_sep = float(hz.iloc[0]["yes_prob"])
    p_normal_3m = 1 - p_s2plus_3m  # crude complement: not-in-S2+ ~ normalizing
    edge = (p_s2plus_3m - (1 - q_normal_sep)) if q_normal_sep is not None else None
    return {
        "signal": "A1 P-Q scenario divergence",
        "value": {"model_P_S2plus_3m": round(p_s2plus_3m, 3),
                  "market_P_normal_by_Sep30": q_normal_sep,
                  "model_P_normal_3m": round(p_normal_3m, 3)},
        "direction": 0 if edge is None or abs(edge) < 0.10 else (1 if edge > 0 else -1),
        "confidence": "low",
        "rationale": "Mearsheimer prior says exits are lumpy and upward mass "
                     "exceeds resolution mass; long convexity if model P(S2+) "
                     "materially exceeds market's non-normalization odds.",
        "caveats": "P is 79% prior at current data weight; the comparison "
                   "conflates 'normalized' with 'not in S2+'. Expression: "
                   "long-dated OTM Brent call spreads, NEVER naked.",
    }


def a2_habituation_vol() -> dict:
    hab = _latest("habituation")
    b = float(hab["b"].iloc[0]) if hab is not None and "b" in hab else np.nan
    supported = np.isfinite(b) and b > 0.10
    return {
        "signal": "A2 habituation vol premium",
        "value": {"decay_rate_b": round(b, 3) if np.isfinite(b) else None},
        "direction": -1 if supported else 0,
        "confidence": "low",
        "rationale": "If repetition decays event-day moves, short-dated vol "
                     "into repeated-pattern strike cycles is systematically rich.",
        "caveats": "Backfill fit finds WEAK decay (half-life ~31 events) — the "
                   "hypothesis is NOT well supported in this war so far; signal "
                   "held at flat until decay strengthens. Defined-risk only.",
    }


def a3_novelty_drift() -> dict:
    es = _latest("event_study")
    ev = _latest("coded_events")
    drift_edge = None
    recent_novel = False
    if es is not None:
        w20 = es[es["window"] == 20]
        drift_edge = float(w20[w20["novel"]]["abn_brent"].mean()
                           - w20[~w20["novel"]]["abn_brent"].mean())
    if ev is not None:
        ev = ev.copy()
        ev["date"] = pd.to_datetime(ev["date"])
        key = ev["actor"].astype(str) + "|" + ev["target_type"].astype(str) + "|" + ev["rung"].astype(str)
        ev["recurrence"] = key.groupby(key).cumcount()
        last7 = ev[ev["date"] >= ev["date"].max() - pd.Timedelta(days=7)]
        novel_events = last7[last7["recurrence"] == 0]
        recent_novel = len(novel_events) > 0
        novel_list = novel_events["action"].tolist()
    return {
        "signal": "A3 post-novel-event drift",
        "value": {"novel_minus_repeated_20d_drift": round(drift_edge, 4) if drift_edge else None,
                  "novel_event_in_last_7d": recent_novel,
                  "recent_novel_events": novel_list if recent_novel else []},
        "direction": 1 if (recent_novel and (drift_edge or 0) > 0.02) else 0,
        "confidence": "medium" if (drift_edge or 0) > 0.04 else "low",
        "rationale": "Backfill shows +20d abnormal Brent after first-of-kind "
                     "events exceeds repeated-pattern events (+16% vs +10%) — "
                     "underreaction to novel rungs. Long oil/tankers 5-20d "
                     "after a novelty flag fires.",
        "caveats": "Overlapping windows in a dense event stream inflate the "
                   "apparent edge; n~20 per bucket.",
    }


def a4_hormuz_basis() -> dict:
    from src.report.basis_monitor import portwatch_leg, polymarket_leg
    from src.common import load_config
    pw = portwatch_leg(load_config("sources")["portwatch"])
    pm = polymarket_leg()
    gap_real = None
    if pm and pm["yes_prob"] is not None:
        data_recovering = (pw["trend"] or 0) > 0.5
        market_bullish = pm["yes_prob"] >= 0.40
        gap_real = market_bullish != data_recovering
    return {
        "signal": "A4 Hormuz basis",
        "value": {"portwatch_7dma": round(pw["ma7"], 1),
                  "transit_trend_per_day": round(pw["trend"], 1),
                  "market_P_normalize_nearest_fwd": pm["yes_prob"] if pm else None,
                  "data_lag_days": pw["lag_days"]},
        "direction": 0 if not gap_real else (-1 if pm["yes_prob"] >= 0.40 else 1),
        "confidence": "medium",
        "rationale": "Three-way read (transits / prediction market / tanker "
                     "equities) disagrees only when someone is wrong; trade the "
                     "laggard leg.",
        "caveats": "PortWatch is Polymarket's resolution source with ~5d lag — "
                   "mechanical-lag gaps are excluded by the decomposition.",
    }


def a5_deal_hedge() -> dict:
    from src.market_implied.predmkt import deal_odds, build_panel
    dl = deal_odds(build_panel())
    return {
        "signal": "A5 deal-shock hedge (MANDATORY with A1)",
        "value": {"deal_market": dl},
        "direction": 1,  # always on when any long-escalation sleeve is on
        "confidence": "high",
        "rationale": "S5 is lumpy: the April 7 whiplash ('civilization will die "
                     "tonight' to ceasefire in hours) and June MOU are direct "
                     "precedent. Every A1 position pairs with cheap downside "
                     "(puts financed by call spreads, or deal-market YES).",
        "caveats": "The June deal decayed within 3 weeks — hedge the shock, "
                   "don't bet on deal durability either way.",
    }


def a6_rhetoric_tilt() -> dict:
    rh = _latest("rhetoric")
    momentum = None
    detail = {}
    if rh is not None:
        rh = rh.copy()
        rh["date"] = pd.to_datetime(rh["date"])
        rh["rhetoric_score"] = pd.to_numeric(rh["rhetoric_score"])
        recent = rh[rh["date"] >= rh["date"].max() - pd.Timedelta(days=10)]
        prior = rh[(rh["date"] < rh["date"].max() - pd.Timedelta(days=10))
                   & (rh["date"] >= rh["date"].max() - pd.Timedelta(days=30))]
        if len(recent) and len(prior):
            momentum = float(recent["rhetoric_score"].mean() - prior["rhetoric_score"].mean())
        detail = {a: float(g["rhetoric_score"].mean())
                  for a, g in recent.groupby("actor")}
    return {
        "signal": "A6 rhetoric-leads-kinetics tilt",
        "value": {"momentum_10d_vs_prior": round(momentum, 2) if momentum is not None else None,
                  "recent_mean_by_actor": detail},
        "direction": 0 if momentum is None or abs(momentum) < 0.5 else (1 if momentum > 0 else -1),
        "confidence": "low",
        "rationale": "Coded rhetoric shifts precede transitions (Mar 22 "
                     "ultimatum preceded infrastructure phase; Jun 14 signing "
                     "preceded reopening). Tilt A1 sizing on momentum.",
        "caveats": "Backfill rhetoric is sparse (18 rows); live coder needed "
                   "for a real daily series. Note Jul 16 White House softening.",
    }


ALL_SIGNALS = [a1_pq_divergence, a2_habituation_vol, a3_novelty_drift,
               a4_hormuz_basis, a5_deal_hedge, a6_rhetoric_tilt]


def compute_all() -> list[dict]:
    return [f() for f in ALL_SIGNALS]


def main() -> int:
    dirmap = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    for s in compute_all():
        print(f"\n{s['signal']}  [{dirmap[s['direction']]}, confidence={s['confidence']}]")
        print(f"  value: {s['value']}")
        print(f"  why:   {s['rationale']}")
        print(f"  BUT:   {s['caveats']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
