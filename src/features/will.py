"""Will / casualties layer (endurance layer 8d) — SCORECARD-ONLY.

Mearsheimer's face-lock: de-escalation requires someone to absorb a domestic
political loss. A democratic US is casualty-averse; Iran's regime-survival stakes
give it high tolerance. This layer reads that political dimension.

Deliberately NOT a P covariate. Per Mearsheimer's own caveat ("models perform
better on material factors than pure politics"), a noisy will-index driving a
transition cell would inject political noise into the forecast. It feeds only the
M10 scorecard, where a coarse DIRECTIONAL read ("is face-lock holding?") is
robust to the softness. This is the softest layer in the system — treat as a
signal, not a measurement.

Outputs the face_lock_score in [0,1] and its components.

Run:  python -m src.features.will
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from src.common import read_latest

KIA = re.compile(r"(?:US|American|allied|Kuwaiti|service member|soldier|troops?|crew)"
                 r"[^.]{0,40}(?:kill|dead|died|casualt|lost)", re.I)
RECENT_WEEKS = 5


def readings() -> dict:
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"]).sort_values("date")
    cutoff = ev["date"].max() - pd.Timedelta(weeks=RECENT_WEEKS)
    recent = ev[ev["date"] >= cutoff]

    us_striking = bool(((recent["actor"] == "US") &
                        (recent["rung"].isin(["S1", "S2", "S3", "S4"]))).any())
    iran_striking = bool((recent["actor"].isin(["IRAN", "PROXY"]) &
                          (recent["rung"].isin(["S1", "S2", "S3", "S4"]))).any())
    # A deal "durably held" only if we are NOT currently at war. A ceasefire that
    # lasts a few weeks then collapses (April, June MOU) IS the decay pattern —
    # it confirms face-lock, it doesn't refute it. So key off current war status.
    last2 = ev[ev["date"] >= ev["date"].max() - pd.Timedelta(weeks=2)]
    currently_at_war = bool(last2["rung"].isin(["S1", "S2", "S3", "S4"]).any())
    recent_deal_held = not currently_at_war

    # allied/US casualty proxy (coarse text match — a floor)
    text = (ev["action"].astype(str) + " " + ev["summary"].astype(str))
    us_casualty_mentions = int(text.str.contains(KIA).sum())

    # US will softening? recent coded US rhetoric trending conciliatory
    us_will_softening = False
    try:
        rh = read_latest("rhetoric").copy()
        rh["date"] = pd.to_datetime(rh["date"], errors="coerce")
        rh["rhetoric_score"] = pd.to_numeric(rh["rhetoric_score"], errors="coerce")
        us = rh[rh["actor"] == "US"].dropna(subset=["date"]).sort_values("date")
        if len(us) >= 3:
            recent_r = us.tail(2)["rhetoric_score"].mean()
            prior_r = us.iloc[-4:-2]["rhetoric_score"].mean() if len(us) >= 4 else us.head(2)["rhetoric_score"].mean()
            us_will_softening = bool(recent_r < prior_r - 0.5)
    except FileNotFoundError:
        pass

    # face-lock: both sides fighting, no deal holding => neither can absorb the
    # loss => de-escalation is politically locked. Softened if US will is cracking.
    base = 0.8 if (us_striking and iran_striking and not recent_deal_held) else 0.35
    face_lock = max(0.0, base - (0.15 if us_will_softening else 0.0))

    return {
        "us_striking": us_striking, "iran_striking": iran_striking,
        "recent_deal_held": recent_deal_held,
        "us_casualty_mentions": us_casualty_mentions,
        "us_will_softening": us_will_softening,
        "face_lock_score": round(face_lock, 3),
    }


def main() -> int:
    try:
        r = readings()
    except FileNotFoundError as exc:
        print(f"[will] missing input: {exc}", file=sys.stderr)
        return 1
    print(f"[will] both sides striking: US={r['us_striking']} Iran={r['iran_striking']}; "
          f"any recent deal HELD: {r['recent_deal_held']}")
    print(f"[will] US casualty mentions (floor): {r['us_casualty_mentions']}; "
          f"US will softening: {r['us_will_softening']}")
    print(f"[will] FACE-LOCK score = {r['face_lock_score']:.2f} "
          "(high = de-escalation politically locked; neither side can absorb the loss)")
    print("[will] NOTE: softest layer — scorecard-only, coarse directional signal, "
          "NOT a P covariate (Mearsheimer: materiel > politics).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
