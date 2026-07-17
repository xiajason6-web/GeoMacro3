"""Bayesian regime transition model — P, the model-implied distribution (§5).

Machinery: conjugate Dirichlet–multinomial on the weekly transition matrix.
With states ~observable (features.state_labels), the posterior over each row is
just prior_pseudocounts + observed_transition_counts — closed form, no MCMC.
That is a feature, not a shortcut: with ~20 weekly transitions the posterior is
prior-dominated by construction, and this makes the blend arithmetic auditable
(you can read exactly how many pseudo-counts the Mearsheimer file contributes
vs. how many real transitions the war has produced).

Soft labels are respected: a week that is 60/40 S2/S3 contributes fractional
transition counts, so classifier ambiguity propagates into the posterior.

Outputs (landed to lake as `regime_forecast`):
  - posterior mean transition matrix + prior/data weight per row
  - P(state at h) for h = 2w, 1M, 3M, 6M from the current state distribution
  - P(touch S4 before S5) and expected remaining time in S2+ (via simulation)

Run:  python -m src.model.regime_markov
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import load_config, write_partition
from src.features.state_labels import label_weeks

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]
HORIZONS_WEEKS = {"2w": 2, "1m": 4, "3m": 13, "6m": 26}
N_SIM = 20_000
RNG_SEED = 20260717  # fixed: reproducible runs, vintage discipline


def posterior_matrix(
    labels: pd.DataFrame, prior_strength: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (posterior mean T, prior pseudocount matrix, observed count matrix).
    prior_strength overrides config when given (the dashboard's Mearsheimer knob:
    0 = data only, 1 = priors as written, >1 = prior-dominated)."""
    priors_cfg = load_config("priors")
    strength = (float(priors_cfg.get("prior_strength", 1.0))
                if prior_strength is None else float(prior_strength))
    prior = np.array(
        [priors_cfg["transition_dirichlet"][s] for s in STATES], dtype=float
    ) * strength

    pcols = [f"p_{s}" for s in STATES]
    P = labels[pcols].values  # (weeks, 6) soft state probabilities
    # fractional transition counts: outer product of consecutive soft labels
    counts = np.zeros((6, 6))
    for t in range(len(P) - 1):
        counts += np.outer(P[t], P[t + 1])

    post = prior + counts
    T = post / post.sum(axis=1, keepdims=True)
    return T, prior, counts


def horizon_forecast(T: np.ndarray, p0: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, w in HORIZONS_WEEKS.items():
        out[name] = p0 @ np.linalg.matrix_power(T, w)
    return out


def touch_probabilities(T: np.ndarray, p0: np.ndarray, max_weeks: int = 52) -> dict:
    """Simulate: P(hit S4 before S5), expected weeks until first S5, time in S2+."""
    rng = np.random.default_rng(RNG_SEED)
    start = rng.choice(6, size=N_SIM, p=p0 / p0.sum())
    s4_first = 0
    s5_first = 0
    weeks_to_s5 = []
    weeks_in_s2plus = np.zeros(N_SIM)
    state = start.copy()
    resolved = np.zeros(N_SIM, dtype=bool)
    for wk in range(1, max_weeks + 1):
        # vectorized one-step transition
        u = rng.random(N_SIM)
        cum = T[state].cumsum(axis=1)
        state = (u[:, None] > cum).sum(axis=1)
        active = ~resolved
        weeks_in_s2plus[active] += np.isin(state[active], [2, 3, 4])
        hit4 = active & (state == 4)
        hit5 = active & (state == 5)
        s4_first += int(hit4.sum())
        s5_first += int(hit5.sum())
        weeks_to_s5.extend([wk] * int(hit5.sum()))
        resolved |= hit4 | hit5
        if resolved.all():
            break
    return {
        "p_touch_s4_before_s5": s4_first / N_SIM,
        "p_touch_s5_before_s4": s5_first / N_SIM,
        "p_neither_within_1y": 1 - (s4_first + s5_first) / N_SIM,
        "median_weeks_to_s5": float(np.median(weeks_to_s5)) if weeks_to_s5 else None,
        "mean_weeks_in_s2plus": float(weeks_in_s2plus.mean()),
    }


def run(prior_strength: float | None = None) -> dict:
    labels = label_weeks()
    T, prior, counts = posterior_matrix(labels, prior_strength)
    p0 = labels[[f"p_{s}" for s in STATES]].iloc[-1].values.astype(float)
    p0 = p0 / p0.sum()

    fc = horizon_forecast(T, p0)
    touch = touch_probabilities(T, p0)
    data_weight = counts.sum() / (counts.sum() + prior.sum())

    return {
        "labels": labels, "T": T, "prior": prior, "counts": counts,
        "p0": p0, "forecasts": fc, "touch": touch, "data_weight": data_weight,
    }


def main() -> int:
    r = run()
    rows = []
    for h, dist in r["forecasts"].items():
        for s, p in zip(STATES, dist):
            rows.append({"horizon": h, "state": s, "prob": float(p)})
    for k, v in r["touch"].items():
        rows.append({"horizon": "touch", "state": k, "prob": v})
    df = pd.DataFrame(rows)
    df["data_weight"] = r["data_weight"]
    out = write_partition(df, "regime_forecast")

    print(f"[regime] current state distribution: "
          + ", ".join(f"{s}={p:.0%}" for s, p in zip(STATES, r["p0"]) if p > 0.01))
    print(f"[regime] posterior is {r['data_weight']:.0%} data / "
          f"{1-r['data_weight']:.0%} prior (by pseudo-count mass)")
    for h in HORIZONS_WEEKS:
        dist = r["forecasts"][h]
        print(f"[regime] {h:>3}: " + ", ".join(
            f"{s}={p:.0%}" for s, p in zip(STATES, dist) if p >= 0.005))
    t = r["touch"]
    print(f"[regime] P(touch S4 before S5)={t['p_touch_s4_before_s5']:.0%}  "
          f"P(S5 first)={t['p_touch_s5_before_s4']:.0%}  "
          f"P(neither in 1y)={t['p_neither_within_1y']:.0%}")
    if t["median_weeks_to_s5"]:
        print(f"[regime] median weeks to S5 (when reached): {t['median_weeks_to_s5']:.0f}")
    print(f"[regime] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
