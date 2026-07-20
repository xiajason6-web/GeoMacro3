"""M10 — the Mearsheimer scorecard: DERIVE the prior-strength slider from live data.

Mearsheimer's thesis decomposes into testable restrictions. Each is a prediction
about what we should observe; each is graded against current data by one of the
endurance layers. The fraction confirmed -> a Mearsheimer-fit score M in [0,1] ->
the prior_strength slider.

  high M = the war is behaving as the thesis predicts -> trust the prior (slider up)
  low  M = the war is deviating -> let the sparse data speak (slider down)

Why this beats a static 1.0 or the empirical-Bayes fit:
  - DYNAMIC: recomputes as the war evolves.
  - GRACEFUL: the day a deal holds or the US gains dominance, the relevant
    sub-score falls, M drops, and the prior stops dominating. It cannot overstay.
  - LEGIBLE: you see WHICH restriction is holding or breaking, not an opaque dial.

The restriction weights are a JUDGMENT (n=1 war) — exposed and equal by default,
not estimated. Restriction #5 leans on 8a/8b; #4 on 8d (scorecard-only politics).

Run:  python -m src.model.scorecard
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import read_latest, write_partition

STRENGTH_MIN = 0.26   # ~50/50 data:prior crossover at current sample size
STRENGTH_MAX = 4.0    # empirical-Bayes ceiling (data prefers the prior shape)

WEIGHTS = {  # judgment, not fitted; equal by default
    "no_coercive_leverage": 1.0,
    "deals_decay": 1.0,
    "asymmetric_escalation": 1.0,
    "face_lock": 1.0,
    "endurance_asymmetry": 1.0,
}


def _clip(x, lo=0.05, hi=0.95):
    return float(np.clip(x, lo, hi))


def grade_no_coercive_leverage() -> dict:
    """US strikes should NOT be producing Iranian concessions."""
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"])
    recent = ev[ev["date"] >= ev["date"].max() - pd.Timedelta(weeks=4)]
    us_strikes = int(((recent["actor"] == "US") &
                      recent["rung"].isin(["S1", "S2", "S3", "S4"])).sum())
    concessions = int((recent["rung"] == "S5").sum())
    if us_strikes == 0:
        return {"score": 0.5, "detail": "no recent US strikes to judge"}
    score = _clip(0.9 - 0.6 * (concessions / max(1, us_strikes)))
    return {"score": score, "detail": f"{us_strikes} US strikes, {concessions} concessions (4wk)"}


def grade_deals_decay() -> dict:
    """Ceasefires/deals should collapse fast (short S5 half-life)."""
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"]).sort_values("date")
    s5 = ev[ev["rung"] == "S5"]
    lives = []
    for d in s5["date"]:
        nxt = ev[(ev["date"] > d) & ev["rung"].isin(["S1", "S2", "S3", "S4"])]
        if len(nxt):
            lives.append((nxt["date"].iloc[0] - d).days / 7.0)
    if not lives:
        return {"score": 0.6, "detail": "no completed deal episodes"}
    med = float(np.median(lives))
    score = _clip(1.0 - med / 12.0)   # <12wk half-life => decay confirmed
    return {"score": score, "detail": f"median deal half-life {med:.1f} wk (n={len(lives)})"}


def grade_asymmetric_escalation() -> dict:
    """War should widen horizontally (8c) and the exchange run against the US (8a)."""
    from src.features.horizontal_spread import spread_now
    from src.model.covariates import munitions_pressure
    try:
        sp = spread_now()
        spread_sig = 1.0 if sp["trailing_4wk_index"] > sp["war_avg_index"] else 0.35
        s3_persist = sp.get("p_s3_persists_next_week") or 0.5
    except Exception:  # noqa: BLE001
        spread_sig, s3_persist = 0.5, 0.5
    ratio = munitions_pressure().get("cost_exchange_ratio", 0) or 0
    exch_sig = _clip((ratio - 1) / 9.0)
    score = _clip(np.mean([spread_sig, s3_persist, exch_sig]))
    return {"score": score, "detail": f"spread {spread_sig:.1f}, S3-persist {s3_persist:.0%}, "
            f"cost-exchange {ratio:.1f}:1"}


def grade_face_lock() -> dict:
    """8d: neither side can absorb the domestic loss of de-escalating."""
    from src.features.will import readings
    r = readings()
    return {"score": _clip(r["face_lock_score"]),
            "detail": f"both striking={r['us_striking'] and r['iran_striking']}, "
            f"no durable deal={not r['recent_deal_held']}, "
            f"US will softening={r['us_will_softening']}"}


def grade_endurance_asymmetry() -> dict:
    """Iran should outlast the US: long Iran runway + US bearing disproportionate
    cost, but discounted while the US is not visibly strained (asymmetry untested)."""
    from src.features.economic import readings as econ
    from src.model.covariates import munitions_pressure
    e = econ()
    ratio = munitions_pressure().get("cost_exchange_ratio", 0) or 0
    iran_endures = 1.0 if (e.get("iran_runway_days") or 0) > 120 else 0.4
    us_burden = _clip((ratio - 1) / 9.0)
    us_untested = e.get("us_oil_pain", 0) < 0.10   # US not feeling price pain yet
    score = _clip(0.5 + 0.2 * iran_endures + 0.15 * us_burden - (0.1 if us_untested else 0))
    rw = e.get("iran_runway_days")
    return {"score": score, "detail": f"Iran runway {rw:.0f}d, "
            f"US-pain {e.get('us_oil_pain', 0):.2f}, cost-exchange {ratio:.1f}:1"
            if rw else f"US-pain {e.get('us_oil_pain', 0):.2f}"}


GRADERS = {
    "no_coercive_leverage": grade_no_coercive_leverage,
    "deals_decay": grade_deals_decay,
    "asymmetric_escalation": grade_asymmetric_escalation,
    "face_lock": grade_face_lock,
    "endurance_asymmetry": grade_endurance_asymmetry,
}


def compute() -> dict:
    subs = {}
    for name, fn in GRADERS.items():
        try:
            subs[name] = fn()
        except Exception as exc:  # noqa: BLE001 — a missing layer -> neutral
            subs[name] = {"score": 0.5, "detail": f"unavailable: {str(exc)[:40]}"}
    wsum = sum(WEIGHTS.values())
    M = sum(WEIGHTS[k] * subs[k]["score"] for k in subs) / wsum
    strength = STRENGTH_MIN + M * (STRENGTH_MAX - STRENGTH_MIN)
    return {"sub_scores": subs, "M": float(M), "derived_strength": float(strength)}


def derived_strength() -> float:
    return compute()["derived_strength"]


def main() -> int:
    r = compute()
    write_partition(pd.DataFrame([{**{k: v["score"] for k, v in r["sub_scores"].items()},
                                   "M": r["M"], "derived_strength": r["derived_strength"]}]),
                    "mearsheimer_scorecard")
    print("MEARSHEIMER SCORECARD — is the war behaving as the thesis predicts?")
    for name, s in r["sub_scores"].items():
        bar = "#" * int(round(s["score"] * 20))
        print(f"  {name:<24} {s['score']:.2f} |{bar:<20}| {s['detail']}")
    print(f"\n  Mearsheimer-fit M = {r['M']:.2f}  ->  derived prior_strength = "
          f"{r['derived_strength']:.2f}")
    dw = 23.0 / (23.0 + r["derived_strength"] * 88.0)
    print(f"  => posterior {dw:.0%} data / {1-dw:.0%} prior. The war is "
          f"~{r['M']:.0%} Mearsheimer-shaped right now.")
    print("  (weights are judgment at n=1; sub-scores degrade gracefully if the "
          "war deviates — a durable deal would drop deals_decay + face_lock.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
