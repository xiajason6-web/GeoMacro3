"""Horizontal-escalation spread index (design doc v2, endurance layer 8c).

Mearsheimer's asymmetric-escalation thesis says the war WIDENS rather than
climbs: Iran degrades third-country Gulf infrastructure (Fujairah, Yanbu,
Kuwaiti desalination) faster and more cheaply than it can be stopped, and
horizontal spread prolongs wars. This module turns the coded event stream into
a weekly measure of that spread.

Columns per week:
  third_party_fronts : distinct THIRD-country targets struck (the core signal;
                       excludes the Iran/Israel/US belligerent homelands, where
                       strikes are S1 vertical exchange, not horizontal spread)
  proxy_active       : distinct proxy / GCC actors engaged (Houthis, militias, GCC)
  s3_events, s3_share
  broad_hit          : a "multiple"/region-wide S3 event fired that week
  spread_index       : third_party_fronts + proxy_active + broad_hit

Three jobs this one series does (per the plan):
  1. S3 ALPHA TRACKER — the axis no market instrument prices (alpha #2).
  2. First ENDURANCE COVARIATE — fronts multiply the depletion rate (couples
     into the future 8a munitions layer: burn ~ fronts x intensity).
  3. Scorecard input for the Mearsheimer PRIOR-STRENGTH slider — "is the war
     spreading horizontally right now?" is a testable Mearsheimer prediction.

Run:  python -m src.features.horizontal_spread
"""
from __future__ import annotations

import sys

import pandas as pd

from src.common import read_latest, write_partition

CORE_BELLIGERENTS = {"iran", "israel", "us", "united states", "usa"}
NON_COUNTRY = {"none", "nan", "", "international", "multiple"}
# Maritime/chokepoint locations belong to the S2 axis (vertical maritime war),
# NOT S3 horizontal spread to third countries — exclude them from front counts.
MARITIME = {"strait of hormuz", "gulf of oman", "persian gulf", "red sea",
            "arabian sea", "gulf", "indian ocean", "bab el-mandeb", "high seas"}


def _is_third_party(country: str) -> bool:
    c = str(country).strip().lower()
    return c not in NON_COUNTRY and c not in CORE_BELLIGERENTS and c not in MARITIME


def _load_events() -> pd.DataFrame:
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"])
    ev["week"] = ev["date"].dt.to_period("W").apply(lambda p: p.start_time.date())
    ev["severity"] = pd.to_numeric(ev["severity"], errors="coerce")
    return ev


def weekly_spread() -> pd.DataFrame:
    ev = _load_events()
    rows = []
    for wk, g in ev.groupby("week"):
        third = {str(c).strip() for c in g["target_country"] if _is_third_party(c)}
        proxies = set(g.loc[g["actor"].isin(["PROXY", "GCC"]), "actor"]) \
            | {str(a) for a in g["actor"] if str(a) == "PROXY"}
        s3 = g[g["rung"] == "S3"]
        broad = int(((g["rung"] == "S3") &
                     (g["target_country"].astype(str).str.lower().isin(["multiple", "international"]))
                     ).any())
        rows.append({
            "week": wk,
            "events": len(g),
            "third_party_fronts": len(third),
            "third_party_list": ", ".join(sorted(third)),
            "proxy_active": len(proxies),
            "s3_events": len(s3),
            "s3_share": len(s3) / len(g) if len(g) else 0.0,
            "broad_hit": broad,
            "max_severity": int(g["severity"].max()) if g["severity"].notna().any() else 0,
        })
    d = pd.DataFrame(rows).sort_values("week").reset_index(drop=True)
    d["spread_index"] = d["third_party_fronts"] + d["proxy_active"] + d["broad_hit"]
    return d


def s3_attractor_stat(d: pd.DataFrame) -> dict:
    """Tests the v2 'S3 is the attractor' claim directly: given an S3 event this
    week, does S3 recur next week (persistence) rather than resolve? Uses only
    consecutive coded weeks."""
    has_s3 = d["s3_events"] > 0
    persist, total = 0, 0
    for i in range(len(d) - 1):
        if has_s3.iloc[i]:
            total += 1
            if has_s3.iloc[i + 1]:
                persist += 1
    return {"n_s3_weeks_with_successor": total,
            "p_s3_persists_next_week": (persist / total) if total else None}


def spread_now(d: pd.DataFrame | None = None) -> dict:
    d = weekly_spread() if d is None else d
    cur = d.iloc[-1]
    trail4 = d.tail(4)["spread_index"].mean()
    war_avg = d["spread_index"].mean()
    peak = d.loc[d["spread_index"].idxmax()]
    return {
        "week": str(cur["week"]),
        "spread_index": int(cur["spread_index"]),
        "third_party_fronts": int(cur["third_party_fronts"]),
        "third_party_list": cur["third_party_list"],
        "s3_share": float(cur["s3_share"]),
        "trailing_4wk_index": float(trail4),
        "war_avg_index": float(war_avg),
        "peak_index": int(peak["spread_index"]),
        "peak_week": str(peak["week"]),
        "at_or_near_peak": bool(cur["spread_index"] >= peak["spread_index"] - 1),
        **s3_attractor_stat(d),
    }


def main() -> int:
    d = weekly_spread()
    out = write_partition(d, "horizontal_spread")
    cols = ["week", "events", "third_party_fronts", "proxy_active", "s3_events",
            "broad_hit", "spread_index"]
    print(d[cols].to_string(index=False))
    now = spread_now(d)
    print(f"\n[spread] current week {now['week']}: index={now['spread_index']} "
          f"({now['third_party_fronts']} third-party fronts: {now['third_party_list']})")
    print(f"[spread] trailing-4wk {now['trailing_4wk_index']:.1f} vs war-avg "
          f"{now['war_avg_index']:.1f}; peak {now['peak_index']} ({now['peak_week']})"
          + ("  <-- AT/NEAR PEAK" if now['at_or_near_peak'] else ""))
    if now["p_s3_persists_next_week"] is not None:
        print(f"[spread] S3-attractor test: P(S3 recurs next week | S3 this week) = "
              f"{now['p_s3_persists_next_week']:.0%} "
              f"(n={now['n_s3_weeks_with_successor']}) — v2 predicts this is high")
    print(f"[spread] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
