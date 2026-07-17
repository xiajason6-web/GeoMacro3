"""Cross-asset scenario fingerprint inversion (design doc §4.3).

Fills the payoff matrix A[asset, scenario] EMPIRICALLY: mean 5d forward return
of each asset bucket during weeks labeled in each state (Feb–Jul 2026 history,
via features.state_labels). Then solves for scenario weights q >= 0, sum q = 1
minimizing ||A q - r_now||^2 + lam ||q - q_anchor||^2.

The ridge term is MANDATORY, not cosmetic: the asset buckets are heavily
collinear (all long-oil-beta in disguise), so unregularized least squares is
effectively rank-deficient and the weights whipsaw on noise. We anchor toward
the prediction-market implied vector when available, else uniform.

Output is a SOFT cross-check on Polymarket — never an independent Q estimate.

Run:  python -m src.market_implied.fingerprint
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.common import read_latest, write_partition
from src.features.state_labels import label_weeks

ASSET_BUCKETS = {
    "crude": ["BZ=F", "CL=F"],
    "vol": ["^OVX"],
    "tankers": ["FRO", "INSW", "TNK", "TRMD", "NAT", "STNG"],
    "energy_eq": ["XLE", "XOP", "OIH"],
    "gold": ["GLD"],
    "defense": ["ITA", "PPA"],
}
STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]
LAMBDA = 0.5  # ridge strength toward anchor


def _bucket_returns(horizon: int = 5) -> pd.DataFrame:
    px = read_latest("prices").copy()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["obs_date"] = pd.to_datetime(px["obs_date"])
    wide = px.pivot_table(index="obs_date", columns="ticker", values="close")
    rets = wide.pct_change(horizon, fill_method=None)
    out = pd.DataFrame(index=rets.index)
    for b, tks in ASSET_BUCKETS.items():
        cols = [t for t in tks if t in rets]
        if cols:
            out[b] = rets[cols].mean(axis=1)
    return out


def build_payoff_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """A[bucket, state] = mean 5d bucket return in weeks labeled that state.
    Returns (A, counts) where counts = weeks of evidence per state."""
    rets = _bucket_returns()
    labels = label_weeks()
    labels["week"] = pd.to_datetime(labels["week"])

    # map each daily return date to its week label (soft: use argmax state)
    lab = labels.set_index("week")["state"]
    week_of = rets.index.to_period("W-SUN").to_timestamp("W-SUN")
    # older pandas: compute week-ending Sunday manually
    week_end = rets.index + pd.to_timedelta(6 - rets.index.dayofweek, unit="D")
    state_series = pd.Series(
        [lab.get(w) for w in week_end], index=rets.index, dtype="object"
    )

    A = pd.DataFrame(index=list(ASSET_BUCKETS), columns=STATES, dtype=float)
    counts = pd.Series(0, index=STATES, dtype=int)
    for s in STATES:
        mask = state_series == s
        counts[s] = int(mask.sum())
        if mask.sum() >= 3:
            A[s] = rets[mask].mean()
    return A, counts


def invert(A: pd.DataFrame, r_now: pd.Series, anchor: np.ndarray | None = None) -> dict:
    valid = [s for s in STATES if A[s].notna().all()]
    M = A[valid].values
    r = r_now.reindex(A.index).values
    n = len(valid)
    q_anchor = anchor if anchor is not None else np.full(n, 1.0 / n)

    def obj(q):
        return float(np.sum((M @ q - r) ** 2) + LAMBDA * np.sum((q - q_anchor) ** 2))

    cons = [{"type": "eq", "fun": lambda q: q.sum() - 1.0}]
    res = minimize(obj, q_anchor, bounds=[(0, 1)] * n, constraints=cons,
                   method="SLSQP")
    return {"states": valid, "weights": res.x, "resid": float(np.sum((M @ res.x - r) ** 2)),
            "success": bool(res.success)}


def main() -> int:
    A, counts = build_payoff_matrix()
    rets = _bucket_returns()
    r_now = rets.dropna(how="all").iloc[-1]

    print("[fingerprint] weeks of evidence per state:", dict(counts))
    est = invert(A, r_now)
    if not est["success"]:
        print("[fingerprint] optimizer failed", file=sys.stderr)
        return 1

    rows = [{"state": s, "weight": w, "evidence_weeks": int(counts[s])}
            for s, w in zip(est["states"], est["weights"])]
    df = pd.DataFrame(rows)
    df["resid"] = est["resid"]
    out = write_partition(df, "fingerprint")
    print(f"[fingerprint] -> {out}")
    for _, r in df.iterrows():
        print(f"  {r['state']}: {r['weight']:.1%}  ({r['evidence_weeks']}w evidence)")
    print("[fingerprint] NOTE: soft cross-check only — collinear buckets, "
          f"ridge-anchored (lambda={LAMBDA}); unlabeled states are excluded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
