"""Inter-coder reliability — does a second model agree with the primary coder?

Runs a SECOND, independent model (claude-haiku-4-5) over the same headline
window the primary coder (sonnet) used, with the same frozen prompt, then
measures agreement on the two fields that drive everything downstream:
  - rung distribution agreement (does the second coder see the same escalation
    mix?) — compared as total-variation distance between rung histograms
  - event-level matching: same-date same-actor pairs, % agreeing on rung and
    on target_type

Standard practice in event-coding research (ACLED-style datasets report
inter-coder checks); an interviewer from any data team will ask for this.
LLM phrasing varies run to run, so exact event alignment is fuzzy — date+actor
matching is the honest, coarse pairing.

Run:  python -m src.coding.reliability
"""
from __future__ import annotations

import sys

import pandas as pd

from src.common import read_latest, write_partition
from src.coding import llm_coder

SECOND_MODEL = "claude-haiku-4-5-20251001"


def main() -> int:
    art = read_latest("gdelt_articles")
    # code the same latest window with the second model
    old = llm_coder.API_MODEL
    llm_coder.API_MODEL = SECOND_MODEL
    try:
        events2 = []
        for (_, _), grp in art.groupby(["window_start", "window_end"]):
            ev, _ = llm_coder.code_window(grp, f"api-{SECOND_MODEL}")
            events2.extend(ev)
    finally:
        llm_coder.API_MODEL = old
    if not events2:
        print("[reliability] second coder returned nothing", file=sys.stderr)
        return 1
    b = pd.DataFrame(events2)
    write_partition(b, "coded_events_second_coder")

    # primary = live raw codings from the same window (sonnet)
    a = read_latest("coded_events_live_raw").copy()

    # 1) rung-mix agreement (total variation distance, 0=identical, 1=disjoint)
    ra = a["rung"].value_counts(normalize=True)
    rb = b["rung"].value_counts(normalize=True)
    rungs = sorted(set(ra.index) | set(rb.index))
    tv = 0.5 * sum(abs(ra.get(r, 0) - rb.get(r, 0)) for r in rungs)

    # 2) event-level: pair on (date, actor), compare rung + target_type
    a2 = a.assign(k=a["date"].astype(str) + "|" + a["actor"].astype(str))
    b2 = b.assign(k=b["date"].astype(str) + "|" + b["actor"].astype(str))
    merged = a2.merge(b2, on="k", suffixes=("_a", "_b"))
    n_pairs = len(merged)
    rung_agree = float((merged["rung_a"] == merged["rung_b"]).mean()) if n_pairs else None
    tt_agree = float((merged["target_type_a"] == merged["target_type_b"]).mean()) if n_pairs else None

    print(f"[reliability] primary (sonnet): {len(a)} events; "
          f"second ({SECOND_MODEL.split('-')[1]}): {len(b)} events")
    print(f"[reliability] rung-mix TV distance: {tv:.2f} "
          f"({'good' if tv < 0.20 else 'moderate' if tv < 0.35 else 'POOR'} — "
          "0=identical mix)")
    if n_pairs:
        print(f"[reliability] {n_pairs} date+actor-matched pairs: "
              f"rung agreement {rung_agree:.0%}, target-type agreement {tt_agree:.0%}")
    print("[reliability] primary rung mix:", dict(a["rung"].value_counts()))
    print("[reliability] second  rung mix:", dict(b["rung"].value_counts()))
    write_partition(pd.DataFrame([{
        "n_primary": len(a), "n_second": len(b), "tv_distance": tv,
        "n_pairs": n_pairs, "rung_agreement": rung_agree,
        "target_type_agreement": tt_agree, "second_model": SECOND_MODEL,
    }]), "coder_reliability")
    return 0


if __name__ == "__main__":
    sys.exit(main())
