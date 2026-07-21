"""Event-study elasticities — empirically calibrate the covariate effect sizes.

The M9 effect sizes (S4_SUPPRESS, S3_PUMP, S5_DRIFT) were declared conviction
knobs (ASSUMPTIONS.md #5). This module estimates two of them from the evidence
that now exists, and is honest that the third is unidentifiable:

1. S4_SUPPRESS <- the empirical S4 CONTINUATION RATE. Across the analog corpus
   and this war, how often does a week in all-out war lead to another week in
   all-out war? Episodes are uniformly short (Praying Mantis, Soleimani, the
   12-day war, this war's two excursion weeks), so continuation is low. We then
   numerically solve for the S4_SUPPRESS value that makes the model's posterior
   T[S4,S4] match the empirical rate at the CURRENT munitions pressure.

2. S3_PUMP <- the spread->S3 conditional transition in this war's own weekly
   series: P(S3-ish next week | spread above war-avg) vs (| below). The ratio,
   scaled by the mean pressure in high-spread weeks, implies the pump size.

3. S5_DRIFT: UNIDENTIFIABLE at current data — economic pressure barely varied
   in-sample (p_b was ~0 all war), and the analog case (Iran's 1988 exhaustion
   -> ceasefire) is a single qualitative episode. Stays a declared knob.

Caveats printed with every run: n is tiny everywhere; these are calibrations to
thin evidence, not estimates with standard errors. Their value is replacing
"0.6 because it felt right" with "0.6±wide because episodes last ~1-2 weeks."

Run:  python -m src.model.elasticities
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import load_config, write_partition

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]


def empirical_s4_continuation() -> dict:
    """S4 self-transition rate pooled across analog segments + live S4 weeks."""
    cfg = load_config("analogs")
    episodes, s4_weeks = 0, 0
    for spec in cfg["conflicts"].values():
        for seg in spec["segments"]:
            if seg["state"] == "S4":
                w = max(1, round((pd.Timestamp(str(seg["end"]))
                                  - pd.Timestamp(str(seg["start"]))).days / 7))
                episodes += 1
                s4_weeks += w
    # live war: consecutive-week runs where S4 has the argmax
    from src.features.state_labels import label_weeks
    lab = label_weeks()
    s4 = (lab["state"] == "S4").values
    i = 0
    while i < len(s4):
        if s4[i]:
            episodes += 1
            run_len = 1
            while i + run_len < len(s4) and s4[i + run_len]:
                run_len += 1
            s4_weeks += run_len
            i += run_len
        else:
            i += 1
    cont = (s4_weeks - episodes) / s4_weeks if s4_weeks else None
    return {"episodes": episodes, "s4_weeks": s4_weeks,
            "continuation_rate": float(cont) if cont is not None else None}


def calibrate_s4_suppress(target_cont: float) -> dict:
    """Numerically find S4_SUPPRESS s.t. posterior T[4,4] ~= empirical rate at
    the current munitions pressure (all other machinery as-is)."""
    import src.model.covariates as cov
    from src.model.regime_markov import posterior_matrix
    from src.features.state_labels import label_weeks

    lab = label_weeks()
    old = cov.S4_SUPPRESS
    best, best_err = old, 1e9
    try:
        for cand in np.arange(0.0, 0.96, 0.05):
            cov.S4_SUPPRESS = float(cand)
            T, *_ = posterior_matrix(lab, None, use_covariates=True)
            err = abs(float(T[4, 4]) - target_cont)
            if err < best_err:
                best, best_err = float(cand), err
        cov.S4_SUPPRESS = old
        T0, *_ = posterior_matrix(lab, None, use_covariates=True)
        return {"calibrated": best, "current": old,
                "model_T44_current": float(T0[4, 4]), "target": target_cont,
                "residual": best_err}
    finally:
        cov.S4_SUPPRESS = old


def empirical_s3_pump() -> dict:
    """P(S3-mass next week | spread high) vs low, from this war's series."""
    from src.common import read_latest
    from src.features.state_labels import label_weeks

    hs = read_latest("horizontal_spread").copy()
    # spread weeks are period-START (Monday); labels are W-SUN period END —
    # shift +6d to the same Sunday key or the merge comes back empty.
    hs["week"] = pd.to_datetime(hs["week"]) + pd.Timedelta(days=6)
    lab = label_weeks().copy()
    lab["week"] = pd.to_datetime(lab["week"])
    df = hs.merge(lab[["week", "p_S3"]], on="week", how="inner").sort_values("week")
    if len(df) < 6:
        return {"n": len(df), "note": "too few merged weeks"}
    avg = df["spread_index"].mean()
    df["next_p_s3"] = df["p_S3"].shift(-1)
    df = df.dropna(subset=["next_p_s3"])
    hi = df[df["spread_index"] > avg]
    lo = df[df["spread_index"] <= avg]
    if not len(hi) or not len(lo) or lo["next_p_s3"].mean() == 0:
        return {"n": len(df), "note": "no contrast available"}
    ratio = hi["next_p_s3"].mean() / max(lo["next_p_s3"].mean(), 1e-6)
    # implied pump at the mean pressure of high-spread weeks:
    # multiplier = 1 + PUMP * p_c  =>  PUMP = (ratio - 1) / mean_p_c_high
    from src.model.covariates import spread_pressure
    p_c_now = spread_pressure().get("p_c", 0.5) or 0.5
    p_c_ref = max(p_c_now, 0.3)  # guard tiny divisor
    implied = float(np.clip((ratio - 1.0) / p_c_ref, 0.0, 0.95))
    return {"n": len(df), "n_high": len(hi), "n_low": len(lo),
            "p_s3_next_high": float(hi["next_p_s3"].mean()),
            "p_s3_next_low": float(lo["next_p_s3"].mean()),
            "ratio": float(ratio), "implied_pump": implied}


def main() -> int:
    print("=" * 66)
    print(" EVENT-STUDY ELASTICITIES — calibrating the M9 effect sizes")
    print("=" * 66)

    e4 = empirical_s4_continuation()
    print(f"\n[S4] {e4['episodes']} all-out episodes across corpus+live, "
          f"{e4['s4_weeks']} total S4-weeks -> empirical continuation "
          f"{e4['continuation_rate']:.0%}")
    cal = calibrate_s4_suppress(e4["continuation_rate"])
    print(f"[S4] model T[S4,S4] at current knob ({cal['current']}): "
          f"{cal['model_T44_current']:.0%}; calibrated S4_SUPPRESS = "
          f"{cal['calibrated']:.2f} (residual {cal['residual']:.3f})")

    e3 = empirical_s3_pump()
    if "implied_pump" in e3:
        print(f"\n[S3] n={e3['n']} weeks ({e3['n_high']} high-spread / "
              f"{e3['n_low']} low): P(S3 next) {e3['p_s3_next_high']:.0%} vs "
              f"{e3['p_s3_next_low']:.0%} -> ratio {e3['ratio']:.2f}, "
              f"implied S3_PUMP = {e3['implied_pump']:.2f}")
    else:
        print(f"\n[S3] not estimable: {e3.get('note')} (n={e3.get('n')})")

    print("\n[S5] S5_DRIFT: UNIDENTIFIABLE — p_b ~0 all war (no in-sample "
          "variation); 1988 exhaustion->ceasefire is one qualitative episode. "
          "Stays a declared knob (0.4).")
    print("\nCAVEATS: n is tiny throughout; calibrations to thin evidence, not "
          "estimates with standard errors. Re-run as episodes accumulate.")

    row = {"s4_episodes": e4["episodes"], "s4_weeks": e4["s4_weeks"],
           "s4_continuation": e4["continuation_rate"],
           "s4_suppress_calibrated": cal["calibrated"],
           "s4_suppress_current": cal["current"],
           **{f"s3_{k}": v for k, v in e3.items() if not isinstance(v, str)}}
    write_partition(pd.DataFrame([row]), "elasticities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
