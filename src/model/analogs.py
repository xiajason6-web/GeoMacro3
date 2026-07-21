"""Analog-conflict transition fit (ASSUMPTIONS.md #1 — the empirical backbone).

Turns the coded analog corpus (config/analogs.yaml) into a 6x6 transition
pseudo-count matrix that joins the posterior as a third voice:

    posterior = strength * mearsheimer_prior  +  analog_counts  +  live_counts

Mechanics: a segment of state X lasting w weeks contributes (w-1) X->X
self-transitions plus one X->(next segment's state) transition; each conflict's
matrix is normalized to `mass_per_conflict` pseudo-counts then scaled by its
relevance weight. Normalization is essential — evidence per conflict is capped
at "one discounted war's worth," so the 7-year Tanker War informs the SHAPE of
its transitions without swamping the blend by sheer duration.

What the corpus contributes that neither the prior nor 23 live weeks can:
  - S2 self-row: the Tanker War's multi-year grind (persistence evidence)
  - S2->S3 and S3->S1: escalation to Gulf infra CAN decay without a deal (2019)
  - S4 exits: decapitation excursions resolved via S1->S0 (2020) and S5 (2025)
  - S5 rows where ceasefires HELD (1988, 2025) — direct counter-evidence to the
    deal-decay restriction, letting the data argue BOTH sides of face-lock.

Run:  python -m src.model.analogs
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import load_config, write_partition

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]
IDX = {s: i for i, s in enumerate(STATES)}


def conflict_counts(segments: list[dict]) -> np.ndarray:
    """Raw weekly transition counts implied by a conflict's dated segments."""
    C = np.zeros((6, 6))
    segs = sorted(segments, key=lambda s: str(s["start"]))
    for i, seg in enumerate(segs):
        weeks = max(1, round((pd.Timestamp(str(seg["end"]))
                              - pd.Timestamp(str(seg["start"]))).days / 7))
        j = IDX[seg["state"]]
        C[j, j] += max(0, weeks - 1)
        if i + 1 < len(segs):
            C[j, IDX[segs[i + 1]["state"]]] += 1
    return C


def build() -> tuple[np.ndarray, pd.DataFrame]:
    """Weighted, per-conflict-normalized analog pseudo-count matrix + summary."""
    cfg = load_config("analogs")
    budget = float(cfg.get("mass_per_conflict", 20))
    total = np.zeros((6, 6))
    rows = []
    for name, spec in cfg["conflicts"].items():
        C = conflict_counts(spec["segments"])
        mass = C.sum()
        if mass == 0:
            continue
        scaled = C / mass * budget * float(spec["weight"])
        total += scaled
        rows.append({"conflict": name, "weight": spec["weight"],
                     "raw_weeks": int(mass), "contributed_mass": float(scaled.sum()),
                     "states_visited": "->".join(dict.fromkeys(
                         s["state"] for s in sorted(spec["segments"],
                                                    key=lambda x: str(x["start"]))))})
    return total, pd.DataFrame(rows)


def main() -> int:
    A, summary = build()
    write_partition(summary, "analog_summary")
    print("[analogs] corpus:")
    for _, r in summary.iterrows():
        print(f"  {r['conflict']:<22} w={r['weight']:.1f}  {r['raw_weeks']:>4} raw wk "
              f"-> {r['contributed_mass']:.1f} pseudo-counts   {r['states_visited']}")
    print(f"[analogs] total analog mass: {A.sum():.1f} "
          "(vs ~23 live transitions, ~88 prior mass at strength 1)")
    print("\n[analogs] analog transition matrix (row-normalized %):")
    R = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-9)
    hdr = "      " + " ".join(f"{s:>5}" for s in STATES)
    print(hdr)
    for i, s in enumerate(STATES):
        print(f"  {s}  " + " ".join(f"{R[i, j]:>5.0%}" if A[i].sum() > 0 else "    -"
                                    for j in range(6)))
    print("\n[analogs] NOTE: S5 rows carry ceasefires that HELD (1988, 2025) — "
          "the corpus argues BOTH sides of the deal-decay restriction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
