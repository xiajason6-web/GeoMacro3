"""A4 Hormuz-basis monitor — the first shippable signal (design doc §7, A4).

Three-way read on the Strait of Hormuz, plus the timing-artifact decomposition:

  1. PortWatch   : actual transit calls (7-day MA) as % of pre-war baseline.
  2. Polymarket  : market-implied P(traffic normalizes by the market's end date).
  3. Tankers     : equity risk premium proxy (tanker basket vs. energy basket).

The decomposition (weakness #1 from planning): PortWatch is BOTH the S2 input to
P and Polymarket's resolution source, and it lags ~5 days. So before calling any
PortWatch-vs-Polymarket gap "alpha", we separate:
  - mechanical lag  : Polymarket is just tracking the same lagged transit data;
  - real belief gap : Polymarket implies a recovery (or non-recovery) the transit
                      trajectory does not yet support.
Only the second is tradable. We surface both and flag the laggard leg.

Run:  python -m src.report.basis_monitor
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.common import load_config, read_latest, today_utc

BASELINE_NORMAL = 60  # Polymarket "normal" threshold: 7dMA transit calls >= 60


# --------------------------------------------------------------------------- #
# Leg 1: PortWatch transit reality
# --------------------------------------------------------------------------- #
def portwatch_leg(cfg) -> dict:
    df = read_latest("portwatch").copy()
    df = df.sort_values("obs_date")
    df["n_total"] = pd.to_numeric(df["n_total"], errors="coerce")

    last7 = df.tail(7)
    ma7 = last7["n_total"].mean()
    prev7 = df.iloc[-14:-7]["n_total"].mean() if len(df) >= 14 else np.nan

    lo, hi = cfg["baseline_window"]
    base = df[(df["obs_date"] >= lo) & (df["obs_date"] <= hi)]["n_total"]
    baseline = base.mean() if len(base) else np.nan

    latest_obs = df.iloc[-1]["obs_date"]
    lag_days = (today_utc() - pd.to_datetime(latest_obs).date()).days
    frac = ma7 / baseline if baseline and not np.isnan(baseline) else np.nan

    # required daily recovery slope to reach the "normal" threshold
    return {
        "latest_obs": latest_obs,
        "lag_days": lag_days,
        "ma7": ma7,
        "prev_ma7": prev7,
        "trend": ma7 - prev7 if not np.isnan(prev7) else np.nan,
        "baseline": baseline,
        "frac_of_baseline": frac,
        "threshold_ma7": BASELINE_NORMAL,
        "gap_to_normal": BASELINE_NORMAL - ma7,
    }


# --------------------------------------------------------------------------- #
# Leg 2: Polymarket implied normalization probability
# --------------------------------------------------------------------------- #
def _yes_price(row) -> float | None:
    try:
        outcomes = json.loads(row["outcomes"]) if row["outcomes"] else []
        prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else []
        for o, p in zip(outcomes, prices):
            if str(o).strip().lower() == "yes":
                return float(p)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    if row.get("last_trade_price") is not None:
        try:
            return float(row["last_trade_price"])
        except (ValueError, TypeError):
            return None
    return None


def polymarket_leg() -> dict | None:
    """Extract the Hormuz-normalization TERM STRUCTURE — the market's implied
    cumulative probability that traffic normalizes (7dMA transit >= 60) by each
    horizon. Far richer than a single market: it's Q's whole reopening-time CDF.
    """
    df = read_latest("polymarket").copy()
    # "traffic returns to normal by <date>" family only (the clean CDF markets)
    mask = df["question"].str.contains("traffic returns? to normal by", case=False, na=False)
    hz = df[mask].copy()
    if len(hz) == 0:
        return None

    hz["end_dt"] = pd.to_datetime(hz["end_date"], errors="coerce", utc=True)
    hz["yes"] = hz.apply(_yes_price, axis=1)
    hz = hz.dropna(subset=["end_dt", "yes"]).sort_values("end_dt")

    today = pd.Timestamp(today_utc(), tz="UTC")
    term_structure = [
        {
            "horizon": r["end_dt"].date().isoformat(),
            "yes_prob": r["yes"],
            "volume": r["volume"],
            "question": r["question"],
        }
        for _, r in hz.iterrows()
    ]

    # headline = nearest FORWARD market (end date strictly after today) with
    # non-trivial volume — the market's belief on near-term normalization, which
    # is what the transit-trajectory decomposition compares against.
    fwd = hz[hz["end_dt"] > today]
    fwd = fwd[fwd["volume"].fillna(0) > 10_000]
    headline = fwd.iloc[0] if len(fwd) else hz.iloc[-1]

    return {
        "question": headline["question"],
        "yes_prob": headline["yes"],
        "volume": headline["volume"],
        "liquidity": headline["liquidity"],
        "end_date": headline["end_dt"].date().isoformat(),
        "criteria": (headline["resolution_criteria"] or "")[:200],
        "term_structure": term_structure,
    }


# --------------------------------------------------------------------------- #
# Leg 3: tanker equity risk-premium proxy
# --------------------------------------------------------------------------- #
def tanker_leg() -> dict:
    df = read_latest("prices").copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    def basket_return(group, days):
        sub = df[df["group"] == group]
        rets = []
        for t, g in sub.groupby("ticker"):
            g = g.sort_values("obs_date")
            if len(g) > days:
                rets.append(g["close"].iloc[-1] / g["close"].iloc[-1 - days] - 1)
        return float(np.nanmean(rets)) if rets else np.nan

    tank_5d = basket_return("tanker_equity", 5)
    tank_20d = basket_return("tanker_equity", 20)
    energy_20d = basket_return("energy_equity", 20)
    # tanker premium = tanker basket outperformance vs energy basket (war/rerouting
    # premium shows up as tankers > energy). Rough proxy, not a clean read.
    premium_20d = (
        tank_20d - energy_20d
        if not (np.isnan(tank_20d) or np.isnan(energy_20d))
        else np.nan
    )
    return {"tanker_5d": tank_5d, "tanker_20d": tank_20d, "premium_vs_energy_20d": premium_20d}


# --------------------------------------------------------------------------- #
# Decomposition + report
# --------------------------------------------------------------------------- #
def decompose(pw: dict, pm: dict | None) -> str:
    if pm is None or pm["yes_prob"] is None:
        return "  n/a — no Hormuz normalization market landed this vintage."

    yes = pm["yes_prob"]
    trend = pw["trend"]
    gap = pw["gap_to_normal"]  # how far 7dMA is below 60 (positive = below)

    # Is the transit data itself already recovering toward normal?
    data_recovering = (not np.isnan(trend)) and trend > 0.5
    # Does the market imply a recovery the data has not shown?
    market_bullish = yes >= 0.40

    lines = [
        f"  PortWatch 7dMA={pw['ma7']:.1f} vs normal={BASELINE_NORMAL} "
        f"(gap {gap:+.1f}), 7d trend {trend:+.1f}/day, data lag {pw['lag_days']}d.",
        f"  Polymarket P(normalize)={yes:.0%} (vol ${pm['volume']:,.0f}).",
    ]
    if market_bullish and not data_recovering:
        lines.append(
            "  -> REAL BELIEF GAP: market prices normalization the transit "
            "trajectory does not yet support. Divergence is not just data lag — "
            "the market expects a deal/reopening ahead of the OSINT. Tradable (A4)."
        )
    elif market_bullish and data_recovering:
        lines.append(
            "  -> MOSTLY MECHANICAL: transit data is itself recovering; Polymarket "
            "is largely tracking the same (lagged) series. Discount the 'divergence'."
        )
    elif not market_bullish and data_recovering:
        lines.append(
            "  -> MARKET LAGGARD: transit data turning up but market still doubts "
            "normalization. Prediction market is the laggard leg — watch for catch-up."
        )
    else:
        lines.append(
            "  -> ALIGNED PESSIMISM: both the data and the market say no near-term "
            "normalization. Consistent with a sticky S2 (Mearsheimer prior)."
        )
    return "\n".join(lines)


def build_report() -> str:
    cfg = load_config("sources")["portwatch"]
    pw = portwatch_leg(cfg)
    pm = polymarket_leg()
    tk = tanker_leg()

    frac = pw["frac_of_baseline"]
    state_hint = (
        "S2 (maritime/chokepoint war)"
        if not np.isnan(frac) and frac < 0.30
        else "S1 or milder" if not np.isnan(frac) and frac > 0.30 else "unknown"
    )

    out = []
    out.append("=" * 68)
    out.append(f"  HORMUZ BASIS MONITOR (A4)   —   {today_utc().isoformat()}")
    out.append("=" * 68)
    out.append("")
    out.append("LEG 1 — PortWatch transit reality")
    out.append(
        f"  7dMA {pw['ma7']:.1f} calls/day = {frac:.0%} of baseline "
        f"({pw['baseline']:.1f}).  regime hint: {state_hint}"
    )
    out.append(
        f"  latest obs {pw['latest_obs']} ({pw['lag_days']}d lag), "
        f"7d trend {pw['trend']:+.1f}/day"
    )
    out.append("")
    out.append("LEG 2 — Polymarket implied normalization (reopening-time CDF)")
    if pm and pm["yes_prob"] is not None:
        out.append("  P(traffic normalizes, 7dMA transit>=60, by horizon):")
        for pt in pm["term_structure"]:
            thin = (pt["volume"] or 0) < 100_000
            flag = "  [thin]" if thin else ""
            out.append(
                f"    by {pt['horizon']}:  {pt['yes_prob']:>5.1%}"
                f"   (vol ${pt['volume']:,.0f}){flag}"
            )
        out.append(
            f"  headline (nearest forward): P(Yes)={pm['yes_prob']:.0%} "
            f"by {pm['end_date']}, vol ${pm['volume']:,.0f}"
        )
    else:
        out.append("  no Hormuz normalization market found this vintage")
    out.append("")
    out.append("LEG 3 — Tanker equity risk premium (proxy)")
    out.append(
        f"  tanker basket 5d={tk['tanker_5d']:+.1%}  20d={tk['tanker_20d']:+.1%}  "
        f"vs energy 20d={tk['premium_vs_energy_20d']:+.1%}"
    )
    out.append("")
    out.append("DECOMPOSITION — mechanical lag vs. real belief gap")
    out.append(decompose(pw, pm))
    out.append("")
    out.append("=" * 68)
    out.append(
        "  Caveat: Q levels are APPROXIMATE (thin/lagged free data); trust "
        "CHANGES over levels. Not financial advice — research framework."
    )
    out.append("=" * 68)
    return "\n".join(out)


def main() -> int:
    print(build_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
