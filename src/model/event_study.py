"""Event study engine — market reaction per coded event (design doc §6.1).

For every coded event: abnormal returns of Brent, OVX, tanker basket, XLE over
[+1d, +5d, +20d] windows vs. a no-event baseline (mean daily return on days with
no event within ±2 days). Tests the two tradable hypotheses:
  - post-event DRIFT after novel escalations (underreaction -> A3)
  - REVERSAL after repeated-pattern events (overreaction -> A2 fade)

Novelty is computed here, not by the LLM: an event is novel if its
(actor, target_type, rung) tuple has not appeared before in the coded stream.
Recurrence count feeds the habituation curve (features.habituation).

Run:  python -m src.model.event_study
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import read_latest, write_partition

WINDOWS = [1, 5, 20]
ASSETS = {
    "brent": ["BZ=F"],
    "ovx": ["^OVX"],
    "tankers": ["FRO", "INSW", "TNK", "TRMD", "NAT", "STNG"],
    "xle": ["XLE"],
}


def _price_panel() -> pd.DataFrame:
    px = read_latest("prices").copy()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["obs_date"] = pd.to_datetime(px["obs_date"])
    wide = px.pivot_table(index="obs_date", columns="ticker", values="close")
    out = pd.DataFrame(index=wide.index)
    for name, tks in ASSETS.items():
        cols = [t for t in tks if t in wide]
        if cols:
            out[name] = wide[cols].pct_change(fill_method=None).mean(axis=1)
    return out


def load_events() -> pd.DataFrame:
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # novelty + recurrence from the event stream itself
    key = ev["actor"].astype(str) + "|" + ev["target_type"].astype(str) + "|" + ev["rung"].astype(str)
    ev["recurrence"] = key.groupby(key).cumcount()  # 0 = first of its kind
    ev["novel"] = ev["recurrence"] == 0
    ev["event_key"] = key
    return ev


def abnormal_returns(events: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    # baseline: mean return on days with no event within +/-2 days
    event_days = set()
    for d in events["date"]:
        for off in range(-2, 3):
            event_days.add((d + pd.Timedelta(days=off)).normalize())
    quiet = rets[~rets.index.normalize().isin(event_days)]
    baseline = quiet.mean()

    rows = []
    trading_days = rets.index.sort_values()
    for _, e in events.iterrows():
        pos = trading_days.searchsorted(e["date"])
        if pos >= len(trading_days):
            continue
        for w in WINDOWS:
            if pos + w >= len(trading_days):
                continue
            win = rets.iloc[pos : pos + w]
            cum = (1 + win).prod() - 1
            abn = cum - baseline * w
            row = {
                "date": e["date"].date().isoformat(),
                "rung": e["rung"], "actor": e["actor"],
                "target_type": e["target_type"], "severity": e["severity"],
                "novel": bool(e["novel"]), "recurrence": int(e["recurrence"]),
                "window": w,
            }
            row.update({f"abn_{a}": float(abn[a]) for a in abn.index})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(ar: pd.DataFrame) -> None:
    print("[event_study] mean abnormal Brent return by novelty (n in parens):")
    for w in WINDOWS:
        sub = ar[ar["window"] == w]
        nov = sub[sub["novel"]]["abn_brent"]
        rep = sub[~sub["novel"]]["abn_brent"]
        print(f"  +{w:>2}d: novel {nov.mean():+.2%} (n={len(nov)})   "
              f"repeated {rep.mean():+.2%} (n={len(rep)})")
    print("[event_study] caveat: n is tiny and windows overlap during dense "
          "strike sequences — treat as descriptive, not inferential.")


def main() -> int:
    events = load_events()
    rets = _price_panel()
    ar = abnormal_returns(events, rets)
    if ar.empty:
        print("[event_study] no events with usable price windows", file=sys.stderr)
        return 1
    out = write_partition(ar, "event_study")
    print(f"[event_study] {len(ar)} event-windows -> {out}")
    summarize(ar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
