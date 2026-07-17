"""Prediction-market panel — structured Q readout from the Polymarket vintage.

Organizes the raw polymarket lake into named families and extracts:
  - the Hormuz normalization CDF (reopening-time term structure)
  - deal/ceasefire odds (the A5 hedge trigger and S5 hazard covariate)
  - escalation families (fees, warships, strikes-on markets)

Every number carries volume so downstream weights thin quotes properly.

Run:  python -m src.market_implied.predmkt
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.common import read_latest, write_partition

FAMILIES = {
    "hormuz_normalize": r"traffic returns? to normal by",
    "hormuz_fees": r"charges hormuz fees",
    "deal": r"(?:ceasefire|deal|agreement|truce)",
    "warships": r"send warships",
    "transit_count": r"ships transit",
}


def _yes(row) -> float | None:
    try:
        outcomes = json.loads(row["outcomes"]) if row["outcomes"] else []
        prices = json.loads(row["outcome_prices"]) if row["outcome_prices"] else []
        for o, p in zip(outcomes, prices):
            if str(o).strip().lower() == "yes":
                return float(p)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        return float(row["last_trade_price"]) if row["last_trade_price"] is not None else None
    except (ValueError, TypeError):
        return None


def build_panel() -> pd.DataFrame:
    df = read_latest("polymarket").copy()
    df["yes_prob"] = df.apply(_yes, axis=1)
    df["end_dt"] = pd.to_datetime(df["end_date"], errors="coerce", utc=True)

    rows = []
    for fam, pattern in FAMILIES.items():
        sub = df[df["question"].str.contains(pattern, case=False, na=False, regex=True)]
        for _, r in sub.iterrows():
            if r["yes_prob"] is None:
                continue
            rows.append({
                "family": fam,
                "question": r["question"],
                "yes_prob": r["yes_prob"],
                "volume": r["volume"],
                "end_date": r["end_dt"].date().isoformat() if pd.notna(r["end_dt"]) else None,
                "closed": r["closed"],
                "market_id": r["market_id"],
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["market_id"])


def deal_odds(panel: pd.DataFrame) -> dict | None:
    """Best single read on near-term deal probability: highest-volume open,
    FORWARD-dated deal-family market (expired June markets pinned at 0% would
    otherwise pollute the read)."""
    from src.common import today_utc

    d = panel[(panel["family"] == "deal") & (panel["closed"] != True)]  # noqa: E712
    d = d[d["end_date"].fillna("") > today_utc().isoformat()]
    d = d[d["volume"].fillna(0) > 10_000].sort_values("volume", ascending=False)
    if not len(d):
        return None
    r = d.iloc[0]
    return {"yes_prob": float(r["yes_prob"]), "question": r["question"],
            "volume": float(r["volume"]), "end_date": r["end_date"]}


def main() -> int:
    panel = build_panel()
    if panel.empty:
        print("[predmkt] empty panel", file=sys.stderr)
        return 1
    out = write_partition(panel, "predmkt_panel")
    print(f"[predmkt] {len(panel)} rows across {panel['family'].nunique()} families -> {out}")
    for fam, grp in panel.groupby("family"):
        top = grp.sort_values("volume", ascending=False).head(3)
        print(f"  [{fam}]")
        for _, r in top.iterrows():
            print(f"    {r['yes_prob']:>6.1%}  vol ${r['volume'] or 0:>12,.0f}  {r['question'][:70]}")
    dl = deal_odds(panel)
    if dl is not None:
        print(f"[predmkt] headline deal odds: {dl['yes_prob']:.1%} "
              f"(\"{dl['question'][:60]}\" ends {dl['end_date']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
