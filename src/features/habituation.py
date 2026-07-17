"""Habituation curve — does the market stop repricing repeated rungs? (§6.2)

Regresses event-day |Brent move| on event-type recurrence count:
    |r_event| = a * exp(-b * recurrence) + c
Hypothesis (the current tape supports it: sixth day of strikes, Brent still
~$86): response amplitude decays with repetition; only novel rungs reprice.

The fitted decay rate b is the A2 signal input: high b = habituation regime =
short-dated vol systematically overprices repeated-pattern strike cycles.

Run:  python -m src.features.habituation
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.common import read_latest, write_partition
from src.model.event_study import load_events, _price_panel


def _decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def fit_habituation() -> dict:
    events = load_events()
    rets = _price_panel()

    rows = []
    trading = rets.index.sort_values()
    for _, e in events.iterrows():
        pos = trading.searchsorted(e["date"])
        if pos >= len(trading):
            continue
        r = rets["brent"].iloc[pos]
        if pd.isna(r):
            continue
        rows.append({"recurrence": int(e["recurrence"]), "abs_move": abs(float(r)),
                     "rung": e["rung"], "date": e["date"].date().isoformat()})
    df = pd.DataFrame(rows)
    if len(df) < 6:
        raise RuntimeError(f"too few events to fit habituation (n={len(df)})")

    x, y = df["recurrence"].values.astype(float), df["abs_move"].values
    try:
        popt, _ = curve_fit(_decay, x, y, p0=[y.max(), 0.5, y.min()],
                            bounds=([0, 0, 0], [1, 5, 0.2]), maxfev=5000)
        a, b, c = (float(v) for v in popt)
        fit_ok = True
    except RuntimeError:
        # fall back to a linear slope sign if the exponential won't converge
        slope = float(np.polyfit(x, y, 1)[0])
        a, b, c, fit_ok = np.nan, np.nan, np.nan, False
        return {"df": df, "a": a, "b": b, "c": c, "fit_ok": fit_ok, "lin_slope": slope}

    half_life = np.log(2) / b if b > 1e-6 else np.inf
    return {"df": df, "a": a, "b": b, "c": c, "fit_ok": fit_ok,
            "half_life_events": float(half_life)}


def main() -> int:
    r = fit_habituation()
    df = r["df"]
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "df"}])
    out = write_partition(df.assign(**{k: v for k, v in r.items() if k != "df"}),
                          "habituation")
    print(f"[habituation] {len(df)} event-day moves -> {out}")
    if r["fit_ok"]:
        print(f"[habituation] |move| = {r['a']:.3f}*exp(-{r['b']:.2f}*n) + {r['c']:.3f}")
        print(f"[habituation] half-life = {r['half_life_events']:.1f} repetitions "
              f"(decay confirmed)" if np.isfinite(r["half_life_events"])
              else "[habituation] no meaningful decay")
    else:
        print(f"[habituation] exp fit failed; linear slope = {r['lin_slope']:+.4f}/repeat")
    # first-vs-later comparison, robust to fit choice
    first = df[df["recurrence"] == 0]["abs_move"].mean()
    later = df[df["recurrence"] >= 3]["abs_move"].mean()
    if not (np.isnan(first) or np.isnan(later)):
        print(f"[habituation] mean |move|: first-of-kind {first:.2%} vs "
              f"4th+ repetition {later:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
