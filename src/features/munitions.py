"""Munitions / interceptor sustainability layer (endurance layer 8a).

Mearsheimer's "no escalation dominance" claim is physical: the US cannot sustain
a high-intensity air campaign, and defending Gulf allies against Iran's cheap
missile/drone salvos costs far more than the salvos themselves. The initial
campaign wound down partly on munitions depletion. This layer turns coded event
TEXT into a munitions ledger and computes the two Mearsheimer-critical readings:

  1. COST-EXCHANGE RATIO — dollars the defender must spend to intercept vs.
     dollars Iran spends to attack. >>1 means the exchange runs structurally
     against the US/allies: his asymmetric-escalation thesis, quantified.
  2. INTERCEPTOR RUNWAY — order-of-magnitude weeks of interceptors left at the
     recent burn rate. Short runway => the US cannot sustain vertical defense
     => S4 breakout is constrained => the war caps at the horizontal S3 grind.
     This is the grind-vs-breakout DISCRIMINATOR the plan called for.

v1 is a rule-based extractor over existing coded-event text — transparent, works
on the frozen backfill with no re-coding, and HONEST that it is a floor (it only
counts munitions the summaries mention; real expenditure is higher). Upgrade
paths: an LLM munitions sub-coder, and think-tank depletion estimates.

Run:  python -m src.features.munitions
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from src.common import load_config, read_latest, write_partition


def _extract(text: str, keywords, catalog) -> list[tuple[str, int]]:
    """Return [(category, count)] found in text; mask matched phrases so specific
    keywords consume their span before generic fallbacks fire."""
    t = f" {str(text).lower()} "
    found = []
    for kw, cat in keywords:
        if kw in t:
            # count: a number within ~2 words before the keyword, else 1
            m = re.search(r"(\d[\d,]*)\s+(?:\w+\s+){0,2}" + re.escape(kw), t)
            n = int(m.group(1).replace(",", "")) if m else 1
            n = min(n, 2000)  # cap absurd parses
            found.append((cat, n))
            t = t.replace(kw, " ")  # mask
    return found


def build_ledger() -> pd.DataFrame:
    cfg = load_config("munitions")
    catalog, keywords = cfg["catalog"], [tuple(k) for k in cfg["keywords"]]
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"])
    ev["week"] = ev["date"].dt.to_period("W").apply(lambda p: p.start_time.date())

    rows = []
    for _, e in ev.iterrows():
        text = f"{e.get('action','')} {e.get('summary','')}"
        for cat, n in _extract(text, keywords, catalog):
            spec = catalog[cat]
            # side: catalog default, but honor the acting party for ANY-side cats
            side = spec["side"]
            if side == "ANY":
                side = "IRAN" if str(e["actor"]).upper() in ("IRAN", "PROXY") else "WEST"
            rows.append({
                "date": e["date"], "week": e["week"], "actor": e["actor"],
                "category": cat, "role": spec["role"], "side": side,
                "count": n, "cost_usd": n * spec["cost"],
            })
    return pd.DataFrame(rows)


def weekly(ledger: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config("munitions")
    catalog = cfg["catalog"]
    # implied defensive burden: every Iranian offensive projectile, if engaged,
    # costs ~one interceptor. Cost-exchange is a STRUCTURAL property, computable
    # from offensive volume even when interceptions aren't separately logged.
    interceptor_unit = np.mean([catalog[c]["cost"] for c in catalog
                                if catalog[c]["role"] == "defensive"])
    rows = []
    for wk, g in ledger.groupby("week"):
        iran_off = g[(g["side"] == "IRAN") & (g["role"] == "offensive")]
        west_off = g[(g["side"] == "WEST") & (g["role"] == "offensive")]
        west_def = g[g["role"] == "defensive"]
        iran_shots = int(iran_off["count"].sum())
        rows.append({
            "week": wk,
            "iran_offensive_usd": float(iran_off["cost_usd"].sum()),
            "west_offensive_usd": float(west_off["cost_usd"].sum()),
            "west_defensive_usd_logged": float(west_def["cost_usd"].sum()),
            "iran_shots": iran_shots,
            "implied_defensive_usd": iran_shots * interceptor_unit,
        })
    w = pd.DataFrame(rows).sort_values("week").reset_index(drop=True)
    return w


def sustainability(ledger: pd.DataFrame, w: pd.DataFrame) -> dict:
    cfg = load_config("munitions")
    inv_lo, inv_hi = cfg["interceptor_inventory"]["low"], cfg["interceptor_inventory"]["high"]

    iran_off_tot = w["iran_offensive_usd"].sum()
    implied_def_tot = w["implied_defensive_usd"].sum()
    ratio = implied_def_tot / iran_off_tot if iran_off_tot else None

    # interceptor burn = Iranian shots/week (each ~1 interceptor if engaged)
    recent = w.tail(4)["iran_shots"].mean()
    war_avg = w["iran_shots"].mean()
    runway_lo = inv_lo / recent if recent else None
    runway_hi = inv_hi / recent if recent else None

    accel = (recent / war_avg) if war_avg else None
    # grind-vs-breakout: short runway + high exchange ratio => vertical unsustainable
    short_runway = runway_hi is not None and runway_hi < 20  # <~5 months even optimistic
    constrained = bool(short_runway and (ratio or 0) > 3)
    return {
        "cost_exchange_ratio": ratio,
        "recent_iran_shots_per_wk": float(recent) if recent else 0.0,
        "burn_acceleration_vs_war_avg": float(accel) if accel else None,
        "interceptor_runway_weeks_lo": float(runway_lo) if runway_lo else None,
        "interceptor_runway_weeks_hi": float(runway_hi) if runway_hi else None,
        "s4_breakout_constrained": constrained,
    }


def main() -> int:
    ledger = build_ledger()
    if ledger.empty:
        print("[munitions] no munition mentions extracted", file=sys.stderr)
        return 1
    w = weekly(ledger)
    write_partition(ledger, "munitions_ledger")
    out = write_partition(w, "munitions_weekly")
    s = sustainability(ledger, w)

    print("[munitions] category totals (count | $M):")
    tot = ledger.groupby(["side", "category"]).agg(
        n=("count", "sum"), usd=("cost_usd", "sum")).reset_index()
    for _, r in tot.sort_values("usd", ascending=False).head(10).iterrows():
        print(f"    {r['side']:>5} {r['category']:<20} {int(r['n']):>5}  ${r['usd']/1e6:,.0f}M")
    print(f"\n[munitions] COST-EXCHANGE RATIO = {s['cost_exchange_ratio']:.1f}:1 "
          "(defender $ to intercept per $1 of Iranian offense)")
    print(f"[munitions]   -> Mearsheimer asymmetric escalation: the exchange runs "
          f"{s['cost_exchange_ratio']:.0f}x against the US/allies")
    print(f"[munitions] recent Iranian shots/wk: {s['recent_iran_shots_per_wk']:.0f} "
          f"(x{s['burn_acceleration_vs_war_avg']:.1f} vs war-avg)"
          if s["burn_acceleration_vs_war_avg"] else "")
    print(f"[munitions] interceptor runway (scenario, WIDE band): "
          f"{s['interceptor_runway_weeks_lo']:.0f}–{s['interceptor_runway_weeks_hi']:.0f} weeks")
    print(f"[munitions] S4-breakout constrained by depletion: {s['s4_breakout_constrained']} "
          "-> horizontal S3 grind favored" if s["s4_breakout_constrained"]
          else "[munitions] depletion not yet binding on the S4 tail")
    print(f"[munitions] -> {out}")
    print("[munitions] NOTE: rule-based FLOOR from event text; real expenditure "
          "is higher. Inventory band is a scenario, not intelligence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
