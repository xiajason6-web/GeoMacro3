"""War-premium decomposition BY MECHANISM (extends the fundamentals control).

The fundamentals control says how much of Brent is war premium ($26 of $82).
This module says WHICH KIND of war is being priced, via three cross-checks:

  1. MARITIME LOCALIZATION — Brent−WTI spread vs its pre-war mean. Brent is the
     seaborne global barrel (Hormuz-exposed); WTI is the landlocked US barrel.
     Spread elevation = the premium is specifically maritime/chokepoint (S2).
  2. GAS LEG (the S3 instrument) — TTF (European LNG, Qatar-exposed: Iran hit
     the world's largest LNG facility in March) vs US Henry Hub (war-insulated).
     TTF elevation vs pre-war while HH stays flat = the market pricing Gulf-
     infrastructure (horizontal, S3) risk — the axis oil alone can't see.
  3. GOLD REGIME FLAG — gold fell ~28% since the war began: the market treats
     this as an oil-supply event, NOT a systemic event. If gold starts RISING
     with escalation, the market is re-classifying the war as systemic —
     an S4-adjacent warning. Watched, never modeled as a textbook safe haven.

All comparisons vs the same PRE-WAR window as fundamentals.py (2024-01..2026-01).
Honest limits: TTF/HH have their own non-war drivers (weather, storage, LNG
capacity cycles) and get no demand control here — % elevation is a CRUDE proxy,
useful for direction and change, not level.

Run:  python -m src.features.premia
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import write_partition

PREWAR_START = "2024-01-01"
PREWAR_END = "2026-01-31"
WAR_START = "2026-02-27"


def _yf_series(ticker: str) -> pd.Series:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=PREWAR_START, auto_adjust=True)
    s = h["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()


def decompose() -> dict:
    brent, wti = _yf_series("BZ=F"), _yf_series("CL=F")
    ttf, hh = _yf_series("TTF=F"), _yf_series("NG=F")
    gold = _yf_series("GC=F")

    out: dict = {"as_of": str(max(brent.index[-1], ttf.index[-1]).date())}

    # 1. maritime localization: Brent-WTI spread vs pre-war mean
    spread = (brent - wti).dropna()
    pre = spread.loc[PREWAR_START:PREWAR_END].mean()
    now = float(spread.iloc[-5:].mean())  # 5d mean, robust to one print
    out.update({"brent_wti_now": now, "brent_wti_prewar": float(pre),
                "maritime_localization": now - float(pre)})

    # 2. gas leg: TTF vs pre-war mean, against Henry Hub as the insulated twin
    ttf_pre, hh_pre = ttf.loc[PREWAR_START:PREWAR_END].mean(), hh.loc[PREWAR_START:PREWAR_END].mean()
    ttf_elev = float(ttf.iloc[-5:].mean() / ttf_pre - 1)
    hh_elev = float(hh.iloc[-5:].mean() / hh_pre - 1)
    out.update({"ttf_elevation": ttf_elev, "hh_elevation": hh_elev,
                "gas_war_premium_proxy": ttf_elev - hh_elev})

    # 3. gold regime flag
    g_war0 = float(gold.loc[:WAR_START].iloc[-1])
    g_now = float(gold.iloc[-1])
    g_30d = float(gold.iloc[-22:].pct_change(fill_method=None).sum())
    since_war = g_now / g_war0 - 1
    regime = ("SYSTEMIC-REPRICING WARNING: gold rising with escalation"
              if (since_war > -0.10 and g_30d > 0.05) else
              "supply-local: market prices an oil event, not a systemic one")
    out.update({"gold_since_war": since_war, "gold_30d": g_30d,
                "gold_regime": regime})
    return out


def main() -> int:
    try:
        d = decompose()
    except Exception as exc:  # noqa: BLE001
        print(f"[premia] failed: {exc}", file=sys.stderr)
        return 1
    write_partition(pd.DataFrame([d]), "premia")
    print(f"[premia] as of {d['as_of']}")
    print(f"[premia] 1. MARITIME: Brent-WTI ${d['brent_wti_now']:+.1f} vs pre-war "
          f"${d['brent_wti_prewar']:+.1f} -> localization ${d['maritime_localization']:+.1f} "
          f"({'chokepoint-specific premium' if d['maritime_localization'] > 1.5 else 'premium not maritime-specific'})")
    print(f"[premia] 2. GAS (S3 instrument): TTF {d['ttf_elevation']:+.0%} vs pre-war, "
          f"Henry Hub {d['hh_elevation']:+.0%} -> gas war-premium proxy "
          f"{d['gas_war_premium_proxy']:+.0%} "
          f"({'market IS pricing Gulf-infrastructure risk' if d['gas_war_premium_proxy'] > 0.15 else 'little horizontal risk priced'})")
    print(f"[premia] 3. GOLD: {d['gold_since_war']:+.0%} since war, {d['gold_30d']:+.1%} last 30d "
          f"-> {d['gold_regime']}")
    print("[premia] NOTE: gas legs are crude proxies (no demand control; weather/"
          "storage uncontrolled). Direction and changes over levels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
