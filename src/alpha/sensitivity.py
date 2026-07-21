"""Sensitivity sweep — which conclusions survive ±50% on the ARBITRARY knobs?

The design doc's falsification discipline, applied to the model's own judgment
knobs (ASSUMPTIONS.md tier-3): one-at-a-time ±50% perturbation of
  - covariate effect sizes (S4_SUPPRESS, S4_DECAY_BOOST, S3_PUMP, S5_DRIFT)
  - endurance pressures p_a/p_c/p_b (equivalent to perturbing every mapping
    denominator at once)
  - scorecard restriction weights (each ±50%)
  - state-classifier soft-label masses
plus two adversarial combos (all-hawkish, all-dovish).

Tracked conclusions:
  C1 P(S2+ at 3m)            "the war persists"
  C2 P(S3 at 3m)             "the war is widening (attractor)"
  C3 P(S4 at 3m)             "the all-out tail is small"
  C4 P(touch S4 before S5)   "S4-before-deal risk"
  C5 derived prior_strength  "how hard the scorecard leans Mearsheimer"

Verdict per conclusion: ROBUST if the direction/claim holds across the whole
sweep; FRAGILE if any perturbation flips it.

Run:  python -m src.alpha.sensitivity
"""
from __future__ import annotations

import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd


@contextmanager
def _patched(module, **attrs):
    old = {k: getattr(module, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def _metrics(prior_strength: float = 1.0) -> dict:
    from src.model.regime_markov import run
    r = run(prior_strength, use_covariates=True)
    f3 = r["forecasts"]["3m"]
    return {
        "P_S2plus_3m": float(sum(f3[2:5])),
        "P_S3_3m": float(f3[3]),
        "P_S4_3m": float(f3[4]),
        "P_touch_S4": float(r["touch"]["p_touch_s4_before_s5"]),
    }


def _scaled_pressures(cov, fa=1.0, fc=1.0, fb=1.0):
    """Return patched pressure functions scaling p_a/p_c/p_b by given factors."""
    base_a, base_c, base_b = (cov.munitions_pressure, cov.spread_pressure,
                              cov.economic_pressure)

    def mk(base, key, f):
        def wrapped():
            d = dict(base())
            d[key] = float(np.clip(d.get(key, 0.0) * f, 0, 1))
            return d
        return wrapped

    return mk(base_a, "p_a", fa), mk(base_c, "p_c", fc), mk(base_b, "p_b", fb)


def sweep() -> pd.DataFrame:
    import src.features.state_labels as sl
    import src.model.covariates as cov
    import src.model.scorecard as sc

    rows = [{"perturbation": "BASE", **_metrics(),
             "derived_strength": sc.compute()["derived_strength"]}]

    # 1) covariate effect sizes ±50%
    for knob in ("S4_SUPPRESS", "S4_DECAY_BOOST", "S3_PUMP", "S5_DRIFT"):
        for f in (0.5, 1.5):
            val = min(getattr(cov, knob) * f, 0.95)
            with _patched(cov, **{knob: val}):
                rows.append({"perturbation": f"{knob} x{f}", **_metrics()})

    # 2) pressures ±50% (covers all mapping denominators)
    for name, kw in (("p_a", "fa"), ("p_c", "fc"), ("p_b", "fb")):
        for f in (0.5, 1.5):
            a, c, b = _scaled_pressures(cov, **{kw: f})
            with _patched(cov, munitions_pressure=a, spread_pressure=c,
                          economic_pressure=b):
                rows.append({"perturbation": f"{name} x{f}", **_metrics()})

    # 3) adversarial combos
    for tag, f in (("ALL-HAWK (pumps x1.5, gates x0.5)", None),
                   ("ALL-DOVE (pumps x0.5, gates x1.5)", None)):
        hawk = tag.startswith("ALL-HAWK")
        a, c, b = _scaled_pressures(cov, fa=0.5 if hawk else 1.5,
                                    fc=1.5 if hawk else 0.5,
                                    fb=0.5 if hawk else 1.5)
        with _patched(cov, munitions_pressure=a, spread_pressure=c,
                      economic_pressure=b,
                      S3_PUMP=min(cov.S3_PUMP * (1.5 if hawk else 0.5), 0.95),
                      S4_SUPPRESS=min(cov.S4_SUPPRESS * (0.5 if hawk else 1.5), 0.95)):
            rows.append({"perturbation": tag, **_metrics()})

    # 4) soft-label masses ±(where meaningful)
    for knob, lo, hi in (("s3_upgrade", 0.25, 0.65), ("s4_upgrade", 0.5, 0.85),
                         ("s5_upgrade", 0.45, 0.85), ("s1_mid", 0.6, 0.95)):
        for v in (lo, hi):
            old = dict(sl.MASS)
            try:
                sl.MASS[knob] = v
                rows.append({"perturbation": f"MASS.{knob}={v}", **_metrics()})
            finally:
                sl.MASS.update(old)

    # 5) scorecard weights ±50% each -> derived strength range
    for k in list(sc.WEIGHTS):
        for f in (0.5, 1.5):
            old = dict(sc.WEIGHTS)
            try:
                sc.WEIGHTS[k] = old[k] * f
                rows.append({"perturbation": f"WEIGHT {k} x{f}",
                             "derived_strength": sc.compute()["derived_strength"]})
            finally:
                sc.WEIGHTS.clear(); sc.WEIGHTS.update(old)

    return pd.DataFrame(rows)


def verdicts(df: pd.DataFrame) -> list[str]:
    out = []
    base = df[df["perturbation"] == "BASE"].iloc[0]

    def rng(col):
        s = df[col].dropna()
        return s.min(), s.max()

    lo, hi = rng("P_S2plus_3m")
    out.append(f"C1 'war persists' P(S2+ @3m): base {base['P_S2plus_3m']:.0%}, "
               f"range {lo:.0%}–{hi:.0%} -> "
               + ("ROBUST (stays majority)" if lo > 0.5 else "FRAGILE (can drop below 50%)"))
    lo, hi = rng("P_S3_3m")
    out.append(f"C2 'widening/attractor' P(S3 @3m): base {base['P_S3_3m']:.0%}, "
               f"range {lo:.0%}–{hi:.0%} -> "
               + ("ROBUST (S3 stays elevated vs prior 15%)" if lo > 0.15 else "FRAGILE"))
    lo, hi = rng("P_S4_3m")
    out.append(f"C3 'small all-out tail' P(S4 @3m): base {base['P_S4_3m']:.0%}, "
               f"range {lo:.0%}–{hi:.0%} -> "
               + ("ROBUST (tail stays <10%)" if hi < 0.10 else "FRAGILE (tail can exceed 10%)"))
    lo, hi = rng("P_touch_S4")
    out.append(f"C4 P(touch S4 before S5): base {base['P_touch_S4']:.0%}, "
               f"range {lo:.0%}–{hi:.0%} -> "
               + ("ROBUST (material tail throughout)" if lo > 0.25 else "FRAGILE"))
    s = df["derived_strength"].dropna()
    out.append(f"C5 derived strength: base {base['derived_strength']:.2f}, "
               f"range {s.min():.2f}–{s.max():.2f} -> "
               + ("ROBUST (always prior-dominated, >1)" if s.min() > 1.0 else "FRAGILE"))
    return out


def touch_band() -> dict:
    """Knob-uncertainty band for the fragile first-passage race: evaluate under
    the two adversarial combos that bound the sensitivity sweep. Returns
    {lo, base, hi} for p_touch_s4_before_s5. Used by the brief/dashboard so the
    race stat is always quoted as a RANGE, never a point (ASSUMPTIONS.md)."""
    import src.model.covariates as cov
    from src.model.regime_markov import run

    base = run(use_covariates=True)["touch"]["p_touch_s4_before_s5"]
    vals = [base]
    for hawk in (True, False):
        a, c, b = _scaled_pressures(cov, fa=0.5 if hawk else 1.5,
                                    fc=1.5 if hawk else 0.5,
                                    fb=0.5 if hawk else 1.5)
        with _patched(cov, munitions_pressure=a, spread_pressure=c,
                      economic_pressure=b,
                      S3_PUMP=min(cov.S3_PUMP * (1.5 if hawk else 0.5), 0.95),
                      S4_SUPPRESS=min(cov.S4_SUPPRESS * (0.5 if hawk else 1.5), 0.95)):
            vals.append(run(use_covariates=True)["touch"]["p_touch_s4_before_s5"])
    return {"lo": float(min(vals)), "base": float(base), "hi": float(max(vals))}


def main() -> int:
    df = sweep()
    from src.common import write_partition
    write_partition(df, "sensitivity")
    pd.set_option("display.width", 140)
    cols = ["perturbation", "P_S2plus_3m", "P_S3_3m", "P_S4_3m", "P_touch_S4",
            "derived_strength"]
    print(df[cols].round(3).to_string(index=False))
    print("\nVERDICTS:")
    for v in verdicts(df):
        print("  " + v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
