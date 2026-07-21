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
    labels: pd.DataFrame,
    prior_strength: float | None = None,
    use_covariates: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns (posterior mean T, prior pseudocount matrix, observed count matrix,
    covariate_info). prior_strength overrides config when given (the dashboard's
    Mearsheimer knob: 0 = data only, 1 = priors as written, >1 = prior-dominated).
    The prior is the horizontal-escalation thesis in priors.yaml.

    use_covariates=True applies the M9 endurance layer: the current 8a munitions
    and 8c spread readings modulate specific transition cells (S4 gate, S3 pump)
    before renormalization — the endurance measurements finally move P. Off by
    default recovers the static model exactly."""
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
    cov_info: dict = {}
    if use_covariates:
        from src.model.covariates import apply as apply_cov
        post, cov_info = apply_cov(post)
    T = post / post.sum(axis=1, keepdims=True)
    return T, prior, counts, cov_info


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
        # vectorized one-step transition; clip guards the float-rounding case
        # where u > cumsum[-1] (~1.0) would yield an out-of-range index 6.
        u = rng.random(N_SIM)
        cum = T[state].cumsum(axis=1)
        state = np.clip((u[:, None] > cum).sum(axis=1), 0, 5)
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
    # MARGINAL touch probabilities — simulated WITHOUT cross-absorption (the war
    # continues through S5 visits, as it really did in April and June). These are
    # far more knob-stable than the first-passage race above (which
    # ratio-compounds hazard perturbations over ~52 steps; see ASSUMPTIONS.md
    # sensitivity results) and answer the decision questions directly.
    rng2 = np.random.default_rng(RNG_SEED + 1)
    st2 = rng2.choice(6, size=N_SIM, p=p0 / p0.sum())
    seen4_13 = np.zeros(N_SIM, dtype=bool); seen5_13 = np.zeros(N_SIM, dtype=bool)
    seen4_26 = np.zeros(N_SIM, dtype=bool); seen5_26 = np.zeros(N_SIM, dtype=bool)
    for wk in range(1, 27):
        u = rng2.random(N_SIM)
        cum = T[st2].cumsum(axis=1)
        st2 = np.clip((u[:, None] > cum).sum(axis=1), 0, 5)
        if wk <= 13:
            seen4_13 |= st2 == 4; seen5_13 |= st2 == 5
        seen4_26 |= st2 == 4; seen5_26 |= st2 == 5

    return {
        "p_touch_s4_before_s5": s4_first / N_SIM,
        "p_touch_s5_before_s4": s5_first / N_SIM,
        "p_neither_within_1y": 1 - (s4_first + s5_first) / N_SIM,
        "median_weeks_to_s5": float(np.median(weeks_to_s5)) if weeks_to_s5 else None,
        "mean_weeks_in_s2plus": float(weeks_in_s2plus.mean()),
        "p_visit_s4_3m": float(seen4_13.mean()),
        "p_visit_s5_3m": float(seen5_13.mean()),
        "p_visit_s4_6m": float(seen4_26.mean()),
        "p_visit_s5_6m": float(seen5_26.mean()),
    }


def run(prior_strength: float | None = None, use_covariates: bool = True) -> dict:
    """use_covariates defaults True: the live model now runs with the M9
    endurance layer wired in. Pass False for the static-prior baseline."""
    labels = label_weeks()
    T, prior, counts, cov_info = posterior_matrix(labels, prior_strength, use_covariates)
    p0 = labels[[f"p_{s}" for s in STATES]].iloc[-1].values.astype(float)
    p0 = p0 / p0.sum()

    fc = horizon_forecast(T, p0)
    touch = touch_probabilities(T, p0)
    data_weight = counts.sum() / (counts.sum() + prior.sum())

    return {
        "labels": labels, "T": T, "prior": prior, "counts": counts,
        "p0": p0, "forecasts": fc, "touch": touch, "data_weight": data_weight,
        "covariates": cov_info, "use_covariates": use_covariates,
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
    ci = r.get("covariates") or {}
    if ci:
        print(f"[regime] M9 endurance covariates ON: munitions p_a={ci['p_a']:.2f} "
              f"(cost-exchange {ci['munitions'].get('cost_exchange_ratio', 0):.1f}:1), "
              f"spread p_c={ci['p_c']:.2f} -> S3 pump, S4 gate active")
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
