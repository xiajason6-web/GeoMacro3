"""Equities read-through — how the war thesis maps to tradeable equity buckets.

Buckets and their thesis linkage:
  tankers      war-risk premia + rerouting economics lift rates; but hulls are
               themselves S2 targets — long persistence, short direct-attack tail
  defense      the production-deficit story IS the revenue story: magazines spent
               at a 15:1 replacement gap must be rebuilt in EVERY scenario,
               including settlement — the least war-path-dependent bucket
  energy_eq    oil-beta with balance-sheet damping
  gulf_markets KSA/UAE/QAT ETFs — the S3 (Gulf-infrastructure) axis is direct
               risk to these markets; the free proxy for Gulf sovereign risk
               (true CDS is paid data)
  airlines     fuel-cost and airspace victim bucket

Computed per bucket, all from the existing price lake + coded events:
  - cumulative return since war outbreak (2026-02-27)
  - return over the June détente window (Jun 13-19) and July re-escalation
    (Jul 7-14) — the sign pattern identifies which side of the war each bucket
    actually trades on
  - S3-day sensitivity: mean daily return on days with a coded S3 event vs
    all other war days (the lateral-axis beta, measured not asserted)

Run:  python -m src.features.equities
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import read_latest, write_partition

BUCKETS = {
    "tankers": ["FRO", "INSW", "TNK", "TRMD", "NAT", "STNG"],
    "defense": ["ITA", "PPA"],
    "energy equity": ["XLE", "XOP", "OIH"],
    "gulf markets": ["KSA", "UAE", "QAT"],
    "airlines": ["JETS"],
}
WAR_START = "2026-02-26"
DETENTE = ("2026-06-13", "2026-06-19")
REESCALATION = ("2026-07-07", "2026-07-14")


def _panel() -> pd.DataFrame:
    px = read_latest("prices").copy()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["obs_date"] = pd.to_datetime(px["obs_date"])
    wide = px.pivot_table(index="obs_date", columns="ticker", values="close")
    out = pd.DataFrame(index=wide.index)
    for b, tks in BUCKETS.items():
        cols = [t for t in tks if t in wide]
        if cols:
            # equal-weight bucket index, normalized returns
            out[b] = wide[cols].pct_change(fill_method=None).mean(axis=1)
    return out


def readthrough() -> pd.DataFrame:
    rets = _panel()
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    s3_days = set(ev.loc[ev["rung"] == "S3", "date"].dt.normalize())

    war = rets.loc[WAR_START:]
    idx_norm = war.index.normalize()
    on_s3 = war[idx_norm.isin(s3_days)]
    off_s3 = war[~idx_norm.isin(s3_days)]

    rows = []
    for b in rets.columns:
        since_war = float((1 + war[b].dropna()).prod() - 1)
        det = rets.loc[DETENTE[0]:DETENTE[1], b]
        ree = rets.loc[REESCALATION[0]:REESCALATION[1], b]
        rows.append({
            "bucket": b,
            "since_war": since_war,
            "detente_jun": float((1 + det.dropna()).prod() - 1),
            "reescalation_jul": float((1 + ree.dropna()).prod() - 1),
            "s3_day_avg": float(on_s3[b].mean()) if len(on_s3) else np.nan,
            "other_day_avg": float(off_s3[b].mean()) if len(off_s3) else np.nan,
            "n_s3_days": int(on_s3[b].notna().sum()),
        })
    df = pd.DataFrame(rows)
    df["s3_sensitivity"] = df["s3_day_avg"] - df["other_day_avg"]
    return df


def main() -> int:
    df = readthrough()
    out = write_partition(df, "equities_readthrough")
    print(f"[equities] read-through -> {out}")
    for _, r in df.iterrows():
        print(f"  {r['bucket']:<14} since-war {r['since_war']:+7.1%}  "
              f"détente {r['detente_jun']:+6.1%}  re-esc {r['reescalation_jul']:+6.1%}  "
              f"S3-day edge {r['s3_sensitivity']:+.2%}/d (n={r['n_s3_days']})")
    print("[equities] read: buckets that rally on re-escalation and sell off in "
          "détente are long the war; Gulf markets' S3-day edge measures the "
          "lateral axis directly. Caveats: overlapping windows, n small.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
