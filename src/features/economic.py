"""Economic-endurance layer (endurance layer 8b).

Mearsheimer: "wars end when one side faces economic collapse or unacceptable
costs," and Iran's cost tolerance exceeds the US's. This layer measures the two
sides' economic strain — which are DIFFERENT variables:

  US pain  = a PRICE story. High Brent → gasoline/inflation → a democratic
             electorate pressures the administration to deal. Gauged vs a $ band.
  Iran pain = a VOLUME story. Strait shut → Iran cannot export → revenue loss
             burns its FX buffer. Sanctions-hardened, so the runway is long —
             his high-tolerance claim, quantified.

Outputs:
  us_oil_pain [0,1], iran_export_loss_usd_day, iran_runway_days, iran_pain [0,1],
  economic_pressure p_b [0,1] (drives the S5-drift covariate), and the ASYMMETRY
  (who has the shorter runway) for the scorecard.

FREE-DATA PROXY: Brent (yfinance) + PortWatch transits. It does NOT yet separate
the war/persistence premium from soft demand — that is the fundamentals-control
job for alpha #1 and needs EIA balances + FRED. Flagged, not faked.

Run:  python -m src.features.economic
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import load_config, read_latest


def _brent() -> float:
    px = read_latest("prices")
    b = px[px["ticker"] == "BZ=F"].copy()
    b["obs_date"] = pd.to_datetime(b["obs_date"])
    return float(b.sort_values("obs_date")["close"].iloc[-1])


def _hormuz_frac() -> float:
    cfg = load_config("sources")["portwatch"]
    pw = read_latest("portwatch").copy()
    pw["n_total"] = pd.to_numeric(pw["n_total"], errors="coerce")
    pw["obs_date"] = pd.to_datetime(pw["obs_date"])
    pw = pw.set_index("obs_date").sort_index()
    ma7 = pw["n_total"].rolling(7, min_periods=4).mean().iloc[-1]
    lo, hi = cfg["baseline_window"]
    baseline = pw.loc[lo:hi, "n_total"].mean()
    return float(ma7 / baseline)


def readings() -> dict:
    cfg = load_config("economic")
    brent = _brent()
    frac = _hormuz_frac()

    lo, hi = cfg["us_oil_pain"]["low"], cfg["us_oil_pain"]["high"]
    # SPR adjustment (Shapiro): a drained reserve removes the buffer that lets
    # Washington ride out price spikes, so the pain band shifts DOWN as SPR
    # falls below its ~650 Mbbl historical full level. Declared mapping: up to
    # -$15 on both band edges at a 300 Mbbl floor (tagged in ASSUMPTIONS.md).
    spr_shift = 0.0
    spr_mbbl = None
    try:
        eia = read_latest("eia_stocks")
        if "spr_kbbl" in eia.columns and eia["spr_kbbl"].notna().any():
            spr_mbbl = float(eia["spr_kbbl"].dropna().iloc[-1]) / 1000.0
            spr_shift = 15.0 * float(np.clip((650.0 - spr_mbbl) / 350.0, 0, 1))
    except FileNotFoundError:
        pass
    lo, hi = lo - spr_shift, hi - spr_shift
    us_pain = float(np.clip((brent - lo) / (hi - lo), 0, 1))

    ie = cfg["iran_export"]
    loss_bpd = ie["baseline_bpd"] * ie["via_hormuz_frac"] * max(0.0, 1.0 - frac)
    loss_usd_day = loss_bpd * brent
    buffer = float(ie["fx_reserve_buffer_usd"])
    runway_days = (buffer / loss_usd_day) if loss_usd_day > 0 else float("inf")
    horizon = ie["pain_horizon_days"]
    iran_pain = float(np.clip((horizon - runway_days) / horizon, 0, 1)) \
        if np.isfinite(runway_days) else 0.0

    p_b = float(np.clip(0.5 * us_pain + 0.5 * iran_pain, 0, 1))
    shorter = "US (price)" if us_pain > iran_pain else "Iran (volume)"
    return {
        "brent": brent, "hormuz_frac": frac,
        "us_oil_pain": us_pain,
        "iran_export_loss_usd_day": loss_usd_day,
        "iran_runway_days": runway_days if np.isfinite(runway_days) else None,
        "iran_pain": iran_pain,
        "economic_pressure": p_b,
        "closer_to_cracking": shorter,
        "spr_mbbl": spr_mbbl,
        "us_pain_band": (lo, hi),
    }


def main() -> int:
    try:
        r = readings()
    except FileNotFoundError as exc:
        print(f"[economic] missing input: {exc}", file=sys.stderr)
        return 1
    rw = r["iran_runway_days"]
    print(f"[economic] Brent ${r['brent']:.0f}, Hormuz {r['hormuz_frac']:.0%} of baseline")
    print(f"[economic] US oil-pain (price story): {r['us_oil_pain']:.2f} "
          "(0 = no political pain; rises past ~$100)")
    print(f"[economic] Iran export loss ~${r['iran_export_loss_usd_day']/1e6:.0f}M/day "
          f"-> fiscal runway {rw:.0f} days" if rw else "[economic] Iran export loss ~0")
    print(f"[economic] Iran pain (volume story): {r['iran_pain']:.2f}  "
          f"-> Mearsheimer's high-tolerance: {'LONG' if (rw or 0) > 120 else 'shortening'} runway")
    print(f"[economic] combined economic pressure p_b = {r['economic_pressure']:.2f} "
          f"-> S5-drift covariate; closer to cracking: {r['closer_to_cracking']}")
    print("[economic] NOTE: free-data proxy; does NOT yet separate war premium "
          "from soft demand (needs EIA/FRED — the fundamentals-control upgrade).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
