"""Weekly S0–S5 state labels from observable criteria (taxonomy.yaml §2).

Two-layer design:
  1. Transit layer (always available): PortWatch 7dMA as fraction of baseline
     gives the maritime backbone — S2 vs degraded-S1 vs near-normal.
  2. Event layer (when coded events exist): S3/S4 evidence upgrades a week;
     S5 requires BOTH a coded deal/ceasefire event AND transits recovering
     (the "announced AND recovering" conjunction in the taxonomy).

Weeks where evidence is ambiguous carry soft probabilities, not hard labels
(carried as p_S* columns; the argmax is convenience only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import load_config, read_latest


def transit_weekly() -> pd.DataFrame:
    cfg = load_config("sources")["portwatch"]
    pw = read_latest("portwatch").copy()
    pw["n_total"] = pd.to_numeric(pw["n_total"], errors="coerce")
    pw["obs_date"] = pd.to_datetime(pw["obs_date"])
    pw = pw.set_index("obs_date").sort_index()
    ma7 = pw["n_total"].rolling(7, min_periods=4).mean()

    lo, hi = cfg["baseline_window"]
    baseline = pw.loc[lo:hi, "n_total"].mean()

    wk = pd.DataFrame({
        "ma7": ma7.resample("W-SUN").last(),
        "trend": ma7.diff(7).resample("W-SUN").last(),
    })
    wk["frac"] = wk["ma7"] / baseline
    wk["baseline"] = baseline
    return wk.dropna(subset=["frac"])


def _coded_events_weekly():
    """Weekly S3/S4/S5 evidence flags from coded events, if any are landed."""
    try:
        ev = read_latest("coded_events").copy()
    except FileNotFoundError:
        return None
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"]).set_index("date")
    flags = pd.DataFrame({
        "s3_evidence": ev["rung"].eq("S3").resample("W-SUN").sum(),
        "s4_evidence": ev["rung"].eq("S4").resample("W-SUN").sum(),
        "s5_evidence": ev["rung"].eq("S5").resample("W-SUN").sum(),
        "max_severity": pd.to_numeric(ev["severity"], errors="coerce").resample("W-SUN").max(),
    })
    return flags


def label_weeks(start: str = "2026-02-01") -> pd.DataFrame:
    wk = transit_weekly()
    wk = wk[wk.index >= pd.Timestamp(start)]
    flags = _coded_events_weekly()
    if flags is not None:
        wk = wk.join(flags)
    for c in ("s3_evidence", "s4_evidence", "s5_evidence"):
        if c not in wk:
            wk[c] = 0
    wk[["s3_evidence", "s4_evidence", "s5_evidence"]] = (
        wk[["s3_evidence", "s4_evidence", "s5_evidence"]].fillna(0)
    )

    probs = []
    for _, r in wk.iterrows():
        p = dict.fromkeys(["S0", "S1", "S2", "S3", "S4", "S5"], 0.0)
        frac, trend = r["frac"], r.get("trend", 0) or 0

        # transit backbone
        if frac >= 0.80:
            p["S0"] += 0.7; p["S1"] += 0.3
        elif frac >= 0.30:
            p["S1"] += 0.8; p["S2"] += 0.2
        else:
            p["S2"] += 1.0

        # event-layer upgrades (S3 shares mass with the maritime state rather
        # than replacing it — the "rung or transition" question stays open)
        if r["s4_evidence"] > 0:
            p = {k: v * 0.3 for k, v in p.items()}; p["S4"] += 0.7
        elif r["s3_evidence"] > 0:
            p = {k: v * 0.55 for k, v in p.items()}; p["S3"] += 0.45

        # S5 = deal evidence AND recovery underway
        if r["s5_evidence"] > 0 and (trend > 1.0 or frac > 0.5):
            p = {k: v * 0.35 for k, v in p.items()}; p["S5"] += 0.65

        tot = sum(p.values())
        probs.append({f"p_{k}": v / tot for k, v in p.items()})

    out = pd.concat([wk.reset_index(), pd.DataFrame(probs)], axis=1)
    pcols = [f"p_S{i}" for i in range(6)]
    out["state"] = out[pcols].idxmax(axis=1).str.replace("p_", "", regex=False)
    out = out.rename(columns={out.columns[0]: "week"})
    return out


if __name__ == "__main__":
    lab = label_weeks()
    cols = ["week", "ma7", "frac", "state"] + [f"p_S{i}" for i in range(6)]
    print(lab[cols].round(2).to_string(index=False))
