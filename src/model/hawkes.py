"""Hawkes process on the kinetic event stream — is this cycle burning out or
compounding? (design doc §6.3)

Univariate exponential Hawkes, hand-rolled MLE (no tick dependency):
    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta (t - t_i))
Branching ratio n = alpha/beta is THE live statistic:
    n < 1: cycle burns out geometrically; n -> 1: near-critical, escalations
    breed escalations. Feeds the transition-model covariates (priors.yaml).

With tens of events the MLE is noisy — we report the profile-likelihood range
alongside the point estimate and refuse false precision.

Run:  python -m src.model.hawkes
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.common import read_latest, write_partition


def _neg_loglik(params, t, T_end):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:  # subcritical constraint
        return 1e10
    # recursive intensity (Ogata): A_i = exp(-beta dt)(1 + A_{i-1})
    n = len(t)
    A = np.zeros(n)
    for i in range(1, n):
        A[i] = np.exp(-beta * (t[i] - t[i - 1])) * (1 + A[i - 1])
    ll = np.sum(np.log(mu + alpha * A))
    ll -= mu * T_end
    ll -= (alpha / beta) * np.sum(1 - np.exp(-beta * (T_end - t)))
    return -ll


def fit(event_dates: pd.Series) -> dict:
    days = pd.to_datetime(event_dates).sort_values()
    t0 = days.iloc[0]
    t = np.array([(d - t0).days + (i * 1e-4) for i, d in enumerate(days)])  # de-tie
    T_end = t[-1] + 1.0

    best = None
    for mu0, a0, b0 in [(0.1, 0.3, 0.5), (0.05, 0.5, 1.0), (0.2, 0.1, 0.3)]:
        res = minimize(_neg_loglik, [mu0, a0, b0], args=(t, T_end),
                       method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    mu, alpha, beta = best.x
    branching = alpha / beta

    # crude profile range on branching ratio: refit with branching pinned
    lo, hi = branching, branching
    base_ll = -best.fun
    for target in np.arange(0.05, 0.96, 0.05):
        def obj(p):
            m, b = p
            return _neg_loglik([m, target * b, b], t, T_end)
        r = minimize(obj, [mu, beta], method="Nelder-Mead",
                     options={"maxiter": 2000})
        if -r.fun > base_ll - 2.0:  # within ~2 log-lik units
            lo, hi = min(lo, target), max(hi, target)

    # current intensity forecast (events/day, next 7d)
    now = t[-1] + 1.0
    lam_now = mu + alpha * np.sum(np.exp(-beta * (now - t)))
    return {
        "mu": float(mu), "alpha": float(alpha), "beta": float(beta),
        "branching": float(branching), "branching_lo": float(lo),
        "branching_hi": float(hi), "n_events": len(t),
        "intensity_now": float(lam_now),
        "baseline_intensity": float(mu / (1 - branching)) if branching < 1 else None,
    }


def main() -> int:
    ev = read_latest("coded_events").copy()
    kinetic = ev[ev["rung"].isin(["S1", "S2", "S3", "S4"])]
    if len(kinetic) < 10:
        print(f"[hawkes] too few kinetic events (n={len(kinetic)})", file=sys.stderr)
        return 1
    r = fit(kinetic["date"])
    out = write_partition(pd.DataFrame([r]), "hawkes")
    print(f"[hawkes] n={r['n_events']} kinetic events")
    print(f"[hawkes] branching ratio = {r['branching']:.2f} "
          f"(profile range {r['branching_lo']:.2f}–{r['branching_hi']:.2f})")
    regime = ("NEAR-CRITICAL — escalations breeding escalations"
              if r["branching"] > 0.7 else
              "self-exciting but subcritical — clusters burn out"
              if r["branching"] > 0.3 else
              "weakly clustered — events mostly exogenous")
    print(f"[hawkes] read: {regime}")
    print(f"[hawkes] current intensity {r['intensity_now']:.2f} events/day "
          f"vs long-run {r['baseline_intensity']:.2f}")
    print(f"[hawkes] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
