"""M9 — endurance covariates wire the orphan gauges (8a, 8c) into P.

Turns the STATIC transition matrix into T(covariates): the current munitions
(8a) and horizontal-spread (8c) readings modulate specific transition cells via
BOUNDED, prior-driven hazard multipliers. This is deliberately NOT a regression
fit — at ~23 weekly transitions any fitted covariate load would be noise. The
multipliers are conviction-tuned and bounded; they nudge the matrix in the
direction the endurance data points, they do not estimate a magnitude.

The thin slice (each gauge owns one region of the 6x6):
  8a munitions -> S4 GATE: depletion pressure p_a suppresses *->S4 and S4->S4
     (all-out war is unsustainable) and boosts S4->S2/S3 (decay to the grind).
  8c spread    -> S3 PUMP: horizontal-widening pressure p_c boosts S1/S2/S3->S3.

Applied to the posterior pseudo-counts (prior*strength + data counts), then rows
are renormalized — so covariates compose cleanly with both the prior and the
data, and turning them off recovers the static model exactly.

Pressures p_a, p_c are in [0,1]; the coefficients below are the conviction knobs.
"""
from __future__ import annotations

import numpy as np

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]

# Conviction knobs (bounded effect sizes; NOT fitted). At full pressure (p=1):
S4_SUPPRESS = 0.6      # *->S4 and S4->S4 multiplied by up to (1 - 0.6) = 0.40
S4_DECAY_BOOST = 0.8   # S4->S2/S3 multiplied by up to (1 + 0.8) = 1.80
S3_PUMP = 0.7          # S1/S2/S3->S3 multiplied by up to (1 + 0.7) = 1.70
S5_DRIFT = 0.4         # *->S5 multiplied by up to (1 + 0.4) = 1.40 — deliberately
#                        small: face-lock (prior) keeps deals unstable even under
#                        economic strain, so 8b only NUDGES the deal drift.


def munitions_pressure() -> dict:
    """p_a in [0,1]: how much munitions/interceptor depletion constrains vertical
    (S4) escalation.

    Weighting is now RESEARCH-GROUNDED (see config/munitions.yaml sources). The
    defense literature (MWI West Point) argues the cost-exchange RATIO is the
    WRONG primary metric; what binds is production rate + magazine depth. So:
      - production asymmetry (Iran out-produces interceptors ~15:1, ~35mo lead
        => no wartime replenishment) is the PRIMARY, structural driver (0.55).
      - magazine runway (depth / burn) is the dynamic secondary (0.30).
      - cost-exchange ratio is illustrative only (0.15); its "alarming" anchor is
        ~10:1 (the Apr-2024 Iran-Israel real exchange), not a made-up 3:1.
    """
    try:
        from src.features.munitions import build_ledger, sustainability, weekly
        led = build_ledger()
        s = sustainability(led, weekly(led))
    except Exception as exc:  # noqa: BLE001 — no munitions data => no pressure
        return {"p_a": 0.0, "note": f"unavailable: {str(exc)[:40]}"}

    gap = s.get("production_gap") or 1.0
    # structural: production gap >1 with no wartime replenishment => S4 is a
    # one-way depletion. gap 1->0, 15->~0.7, capped.
    struct_p = float(np.clip((gap - 1.0) / 20.0, 0, 1)) if not s.get(
        "wartime_replenishment_possible", False) else 0.0
    runway_hi = s.get("interceptor_runway_weeks_hi")
    runway_p = float(np.clip((30.0 - (runway_hi if runway_hi else 999)) / 30.0, 0, 1))
    ratio = s.get("cost_exchange_ratio") or 0.0
    ratio_p = float(np.clip((ratio - 5.0) / 25.0, 0, 1))      # 10:1 alarming -> 0.2
    p_a = float(np.clip(0.55 * struct_p + 0.30 * runway_p + 0.15 * ratio_p, 0, 1))
    return {"p_a": p_a, "cost_exchange_ratio": ratio, "production_gap": gap,
            "runway_hi_weeks": runway_hi, "struct_component": struct_p,
            "runway_component": runway_p, "ratio_component": ratio_p}


def economic_pressure() -> dict:
    """p_b in [0,1]: combined economic strain pushing toward de-escalation (US
    oil-price pain + Iran fiscal-runway pain). Low now (cheap-ish oil, long Iran
    runway) => near-zero S5 drift, which is why deals aren't imminent."""
    try:
        from src.features.economic import readings
        r = readings()
    except Exception as exc:  # noqa: BLE001
        return {"p_b": 0.0, "note": f"unavailable: {str(exc)[:40]}"}
    return {"p_b": r["economic_pressure"], "us_oil_pain": r["us_oil_pain"],
            "iran_pain": r["iran_pain"], "iran_runway_days": r["iran_runway_days"],
            "closer_to_cracking": r["closer_to_cracking"]}


def spread_pressure() -> dict:
    """p_c in [0,1]: how much the war is widening horizontally right now,
    from the 8c trailing-4wk spread index vs the war average."""
    try:
        from src.features.horizontal_spread import spread_now
        sp = spread_now()
    except Exception as exc:  # noqa: BLE001
        return {"p_c": 0.0, "note": f"unavailable: {str(exc)[:40]}"}
    war_avg = sp.get("war_avg_index") or 1.0
    ratio = (sp.get("trailing_4wk_index", 0.0) / war_avg) if war_avg else 1.0
    p_c = float(np.clip((ratio - 1.0) / 1.5, 0, 1))           # ratio 1->0, 2.5->1
    return {"p_c": p_c, "spread_ratio_vs_war_avg": ratio,
            "trailing_4wk": sp.get("trailing_4wk_index"), "war_avg": war_avg}


def multiplier_matrix() -> tuple[np.ndarray, dict]:
    """The 6x6 cell multipliers implied by the current 8a/8c readings."""
    a, c, e = munitions_pressure(), spread_pressure(), economic_pressure()
    pa, pc, pb = a["p_a"], c["p_c"], e["p_b"]
    M = np.ones((6, 6))

    # --- 8a: S4 gate (index 4) ---
    m_enter = 1.0 - S4_SUPPRESS * pa
    for r in (0, 1, 2, 3):
        M[r, 4] *= m_enter          # entry into all-out war suppressed
    M[4, 4] *= m_enter              # S4 persistence decays (unsustainable)
    M[4, 2] *= 1.0 + S4_DECAY_BOOST * pa   # S4 -> S2 (fall back to chokepoint war)
    M[4, 3] *= 1.0 + S4_DECAY_BOOST * pa   # S4 -> S3 (fall back to infra grind)

    # --- 8c: S3 pump (index 3) ---
    m_s3 = 1.0 + S3_PUMP * pc
    for r in (1, 2, 3):
        M[r, 3] *= m_s3             # widening -> more mass into/within S3

    # --- 8b: S5 drift (index 5) ---
    m_s5 = 1.0 + S5_DRIFT * pb
    for r in (0, 1, 2, 3):
        M[r, 5] *= m_s5            # economic strain nudges toward a deal

    return M, {"munitions": a, "spread": c, "economic": e,
               "p_a": pa, "p_c": pc, "p_b": pb}


def apply(post: np.ndarray) -> tuple[np.ndarray, dict]:
    """Modulate posterior pseudo-counts by the covariate multipliers.
    Caller renormalizes rows afterward. Returns (modulated_post, info)."""
    M, info = multiplier_matrix()
    info["multiplier_matrix"] = M
    return post * M, info


if __name__ == "__main__":
    M, info = multiplier_matrix()
    print(f"munitions p_a = {info['p_a']:.2f}  (8a: production gap "
          f"{info['munitions'].get('production_gap', 0):.0f}:1 [founded], "
          f"cost-exchange {info['munitions'].get('cost_exchange_ratio', 0):.1f}:1 "
          f"[illustrative]) -> S4 gate")
    print(f"spread    p_c = {info['p_c']:.2f}  (8c: trailing-4wk "
          f"{info['spread'].get('spread_ratio_vs_war_avg', 0):.1f}x war-avg) -> S3 pump")
    print(f"economic  p_b = {info['p_b']:.2f}  (8b: US-pain "
          f"{info['economic'].get('us_oil_pain', 0):.2f}, Iran-pain "
          f"{info['economic'].get('iran_pain', 0):.2f}) -> S5 drift")
    print("\nnon-trivial cell multipliers (row->col : x):")
    for i in range(6):
        for j in range(6):
            if abs(M[i, j] - 1.0) > 1e-6:
                print(f"  {STATES[i]}->{STATES[j]} : x{M[i, j]:.2f}")
