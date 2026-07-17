"""Walk-forward falsification harness (§7) — NOT a performance estimator.

With one war and ~20 weeks, any Sharpe computed here is noise. What this
harness answers instead: does a signal's rule SURVIVE
  (a) walk-forward application with strict information timing (events dated
      the day they happened; prices from the next trading day — you cannot
      trade the close of a day whose events you haven't seen), and
  (b) +/-50% perturbation of every free parameter (robustness grid)?
A signal that flips sign under perturbation is a research note, not capital.

Signals testable on the backfill: A3 (novelty drift) and A6 (rhetoric tilt).
A1/A4 are P-vs-Q levels (no historical Q panel yet — needs daily vintages,
which only accumulate from today forward). A2 fit already reported weak. A5 is
a hedge rule, not a return stream.

Run:  python -m src.alpha.backtest
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import read_latest, write_partition


def _brent_returns() -> pd.Series:
    px = read_latest("prices")
    b = px[px["ticker"] == "BZ=F"].copy()
    b["obs_date"] = pd.to_datetime(b["obs_date"])
    b = b.set_index("obs_date").sort_index()
    return b["close"].pct_change(fill_method=None).dropna()


def _events() -> pd.DataFrame:
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"])
    key = ev["actor"].astype(str) + "|" + ev["target_type"].astype(str) + "|" + ev["rung"].astype(str)
    ev = ev.sort_values("date").reset_index(drop=True)
    ev["recurrence"] = key.groupby(key).cumcount()
    return ev


def a3_walk_forward(hold_days: int = 10, entry_lag: int = 1,
                    min_severity: int = 3) -> dict:
    """Rule: after a novel event of severity >= min_severity, long Brent from
    close of (event + entry_lag) trading day, hold `hold_days`. Long-only —
    the hypothesis is underreaction to novel escalation, not symmetry."""
    rets = _brent_returns()
    days = rets.index
    ev = _events()
    novel = ev[(ev["recurrence"] == 0)
               & (pd.to_numeric(ev["severity"]) >= min_severity)
               & (ev["rung"] != "S5")]  # S5 novelty is the deal shock, not escalation
    trades = []
    for _, e in novel.iterrows():
        pos = days.searchsorted(e["date"]) + entry_lag
        if pos + hold_days >= len(days):
            continue
        r = float((1 + rets.iloc[pos: pos + hold_days]).prod() - 1)
        trades.append({"date": e["date"].date().isoformat(),
                       "action": e["action"][:40], "ret": r})
    if not trades:
        return {"n": 0}
    df = pd.DataFrame(trades)
    return {"n": len(df), "mean_ret": float(df["ret"].mean()),
            "hit_rate": float((df["ret"] > 0).mean()),
            "worst": float(df["ret"].min()), "trades": df}


def a3_robustness() -> pd.DataFrame:
    """Perturb hold_days, entry_lag, min_severity +/-50%; does sign survive?"""
    rows = []
    for hold in (5, 10, 15):
        for lag in (1, 2):
            for sev in (2, 3, 4):
                r = a3_walk_forward(hold, lag, sev)
                if r["n"] >= 4:
                    rows.append({"hold_days": hold, "entry_lag": lag,
                                 "min_severity": sev, "n": r["n"],
                                 "mean_ret": r["mean_ret"],
                                 "hit_rate": r["hit_rate"]})
    return pd.DataFrame(rows)


def a6_walk_forward(hold_days: int = 10, threshold: float = 1.5) -> dict:
    """Rule: when the mean coded rhetoric score over the trailing 7d crosses
    above `threshold` (escalatory), long Brent for `hold_days`; below
    -threshold (conciliatory), short. Uses only scores dated <= entry day."""
    rets = _brent_returns()
    rh = read_latest("rhetoric").copy()
    rh["date"] = pd.to_datetime(rh["date"])
    rh["rhetoric_score"] = pd.to_numeric(rh["rhetoric_score"])
    daily = rh.groupby("date")["rhetoric_score"].mean()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    trail = daily.rolling(7, min_periods=1).mean()

    days = rets.index
    trades = []
    position_until = None
    for d, score in trail.dropna().items():
        if position_until is not None and d <= position_until:
            continue
        side = 1 if score >= threshold else (-1 if score <= -threshold else 0)
        if side == 0:
            continue
        pos = days.searchsorted(d) + 1
        if pos + hold_days >= len(days):
            continue
        r = side * float((1 + rets.iloc[pos: pos + hold_days]).prod() - 1)
        trades.append({"date": d.date().isoformat(), "side": side, "ret": r})
        position_until = d + pd.Timedelta(days=hold_days)
    if not trades:
        return {"n": 0}
    df = pd.DataFrame(trades)
    return {"n": len(df), "mean_ret": float(df["ret"].mean()),
            "hit_rate": float((df["ret"] > 0).mean()), "trades": df}


def a6_robustness() -> pd.DataFrame:
    rows = []
    for hold in (5, 10, 15):
        for thr in (0.75, 1.0, 1.5):
            r = a6_walk_forward(hold, thr)
            if r.get("n", 0) >= 3:
                rows.append({"hold_days": hold, "threshold": thr, "n": r["n"],
                             "mean_ret": r["mean_ret"], "hit_rate": r["hit_rate"]})
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 64)
    print(" BACKTEST AS FALSIFICATION — survival, not performance")
    print("=" * 64)

    base = a3_walk_forward()
    print(f"\n[A3 novelty-drift] base spec: n={base['n']} trades, "
          f"mean {base['mean_ret']:+.1%}, hit {base['hit_rate']:.0%}, "
          f"worst {base['worst']:+.1%}")
    grid = a3_robustness()
    if len(grid):
        pos = (grid["mean_ret"] > 0).mean()
        print(f"[A3] robustness grid ({len(grid)} specs): "
              f"{pos:.0%} positive mean, range "
              f"{grid['mean_ret'].min():+.1%}..{grid['mean_ret'].max():+.1%}")
        verdict = "SURVIVES" if pos >= 0.8 else "FRAGILE"
        print(f"[A3] verdict: {verdict}")
        grid["signal"] = "A3"

    a6 = a6_walk_forward()
    if a6.get("n", 0) > 0:
        print(f"\n[A6 rhetoric-tilt] base spec: n={a6['n']} trades, "
              f"mean {a6['mean_ret']:+.1%}, hit {a6['hit_rate']:.0%}")
        g6 = a6_robustness()
        if len(g6):
            pos6 = (g6["mean_ret"] > 0).mean()
            print(f"[A6] robustness grid ({len(g6)} specs): {pos6:.0%} positive, "
                  f"range {g6['mean_ret'].min():+.1%}..{g6['mean_ret'].max():+.1%}")
            print(f"[A6] verdict: {'SURVIVES' if pos6 >= 0.8 else 'FRAGILE'}")
            g6["signal"] = "A6"
            grid = pd.concat([grid, g6], ignore_index=True)
    else:
        print("\n[A6] too few rhetoric-triggered trades to test")

    if len(grid):
        out = write_partition(grid, "backtest_grid")
        print(f"\n[backtest] grid -> {out}")
    print("\nCAVEATS: n=1 war; overlapping event windows; rhetoric series is "
          "sparse backfill. Survival here earns a signal LIVE MONITORING, "
          "not capital. A1/A4 need accumulated daily vintages to test at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
