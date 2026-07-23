"""Walk-forward hindcast — the honest backtest for a model built mid-war.

At each weekly cutoff t (mid-March onward), the model is refit using ONLY
information visible at t: transits lagged 5 days (PortWatch's real publication
lag), coded events dated <= t, transition counts from the truncated labels.
It then forecasts the state distribution 2 and 4 weeks ahead, which is graded
against the realized (full-history) label with the ranked probability score
(RPS — the proper score for ordered categories) and a Brier score on the
binary "still disrupted (S2+)" question, against two benchmarks:

  persistence  — next week's state = this week's state (the no-model forecast)
  climatology  — the average state distribution observed up to t

LEAKAGE DISCLOSED, not hidden (this is a hindcast, not out-of-sample):
  - the PRIOR's shape was written knowing this war's arc (v2 horizontal thesis)
  - the event CODING is retrospective (backfilled from chronologies)
  - covariates are OFF here because their effect sizes were calibrated on the
    full war; the analog corpus stays ON because it is genuinely pre-war
    information. What this exercise CAN show: whether the machinery converts
    point-in-time inputs into calibrated probabilities, and whether it beats
    naive benchmarks. What it CANNOT show: true out-of-sample skill — only the
    live ledger shows that.

Run:  python -m src.alpha.replay
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import write_partition
from src.features.state_labels import label_weeks
from src.model.regime_markov import posterior_matrix

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]
PCOLS = [f"p_{s}" for s in STATES]
HORIZONS = (2, 4)
START = "2026-03-22"   # first cutoff with enough truncated history to fit
MIN_WEEKS = 5


def _rps(forecast: np.ndarray, observed_idx: int) -> float:
    """Ranked probability score over the 6 ordered states (lower = better)."""
    cf = np.cumsum(forecast)
    co = np.cumsum(np.eye(6)[observed_idx])
    return float(np.sum((cf - co) ** 2) / (len(forecast) - 1))


def run_replay() -> pd.DataFrame:
    full = label_weeks()
    full["week"] = pd.to_datetime(full["week"])
    realized = full.set_index("week")

    cutoffs = pd.date_range(START, realized.index.max()
                            - pd.Timedelta(weeks=min(HORIZONS)), freq="W-SUN")
    rows = []
    for t in cutoffs:
        lab = label_weeks(as_of=t)
        if len(lab) < MIN_WEEKS:
            continue
        # point-in-time model: prior + analogs + truncated counts.
        # covariates OFF (their effect sizes were calibrated on the full war).
        T, _, _, _ = posterior_matrix(lab, prior_strength=1.0,
                                      use_covariates=False, use_analogs=True)
        p0 = lab[PCOLS].iloc[-1].values.astype(float)
        p0 = p0 / p0.sum()
        p0_idx = int(np.argmax(p0))

        # climatology benchmark: mean soft label observed up to t (no leakage)
        clim = lab[PCOLS].mean().values
        clim = clim / clim.sum()

        anchor = lab["week"].iloc[-1]  # last labeled week at time t
        for h in HORIZONS:
            target = pd.Timestamp(anchor) + pd.Timedelta(weeks=h)
            if target not in realized.index:
                continue
            obs = realized.loc[target, PCOLS].values.astype(float)
            obs_idx = int(np.argmax(obs))

            fc_model = p0 @ np.linalg.matrix_power(T, h)
            fc_persist = np.eye(6)[p0_idx]

            def s2p(v):
                return float(np.sum(v[2:5]))

            rows.append({
                "cutoff": t.date(), "horizon_w": h, "target": target.date(),
                "observed": STATES[obs_idx],
                "rps_model": _rps(fc_model, obs_idx),
                "rps_persist": _rps(fc_persist, obs_idx),
                "rps_clim": _rps(clim, obs_idx),
                "brier_s2p_model": (s2p(fc_model) - s2p(np.eye(6)[obs_idx])) ** 2,
                "brier_s2p_persist": (s2p(fc_persist) - s2p(np.eye(6)[obs_idx])) ** 2,
                "brier_s2p_clim": (s2p(clim) - s2p(np.eye(6)[obs_idx])) ** 2,
                "p_s2p_model": s2p(fc_model),
                "s2p_observed": s2p(np.eye(6)[obs_idx]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    df = run_replay()
    if df.empty:
        print("[replay] no gradeable forecasts", file=sys.stderr)
        return 1
    write_partition(df, "backtest_replay")

    print("WALK-FORWARD HINDCAST — point-in-time refits, graded vs realized")
    print(f"  {len(df)} graded forecasts across {df['cutoff'].nunique()} weekly "
          f"cutoffs ({df['cutoff'].min()} .. {df['cutoff'].max()})")
    for h in HORIZONS:
        sub = df[df["horizon_w"] == h]
        if not len(sub):
            continue
        print(f"\n  horizon {h}w (n={len(sub)}):")
        print(f"    RPS   — model {sub['rps_model'].mean():.3f}  "
              f"persistence {sub['rps_persist'].mean():.3f}  "
              f"climatology {sub['rps_clim'].mean():.3f}   (lower is better)")
        print(f"    Brier(S2+) — model {sub['brier_s2p_model'].mean():.3f}  "
              f"persistence {sub['brier_s2p_persist'].mean():.3f}  "
              f"climatology {sub['brier_s2p_clim'].mean():.3f}")
        wins = (sub["rps_model"] < sub["rps_persist"]).mean()
        print(f"    model beats persistence on RPS in {wins:.0%} of weeks")
    # calibration on the S2+ binary: mean forecast vs realized frequency
    d4 = df[df["horizon_w"] == 4]
    if len(d4) >= 6:
        print(f"\n  calibration (4w, S2+): mean forecast {d4['p_s2p_model'].mean():.0%} "
              f"vs realized frequency {d4['s2p_observed'].mean():.0%} "
              f"(gap {d4['p_s2p_model'].mean()-d4['s2p_observed'].mean():+.0%}; "
              "near zero = well calibrated on average)")
    print("\n  DISCLOSED LEAKAGE: prior shape and event coding are retrospective;"
          "\n  covariates off (calibrated on full war); analogs kept (pre-war info)."
          "\n  This grades the MACHINERY, not out-of-sample skill — the live"
          "\n  ledger is the only true out-of-sample test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
