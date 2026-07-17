"""Peace-shock stress test (§7 portfolio rules) — the trade's true risk.

Precedent is direct: on 2026-04-07 a ceasefire was announced hours after peak
escalation rhetoric; on 2026-06-14 the MOU dropped and transits tripled within
two weeks. A book long escalation convexity bleeds hard on those tapes.

We stress a stylized A1+A5 book against three overnight scenarios, using moves
measured from THIS war's own deal windows (empirical, not textbook):
  deal_shock   : the Jun 14-18 window (MOU signed, blockade lifted)
  partial_deal : the Apr 7-11 window (ceasefire announced, quickly eroded)
  escalation   : the Jul 8-13 window (MOU dead, strait re-closed) — the upside
                 case, for symmetry.

Run:  python -m src.alpha.stress
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import read_latest

# Stylized book: notional weights per sleeve (research units, not dollars).
BOOK = {
    "brent_call_spread": 0.50,   # A1: long-dated OTM call spread (delta ~0.3, capped)
    "tanker_basket": 0.20,       # A3/A4 expression
    "ovx_proxy": 0.15,           # long vol sleeve
    "deal_hedge_puts": 0.15,     # A5: downside puts / deal-market YES
}

WINDOWS = {
    "deal_shock (Jun 14-18 MOU)": ("2026-06-13", "2026-06-19"),
    "partial_deal (Apr 7-11)": ("2026-04-06", "2026-04-12"),
    "escalation (Jul 8-13)": ("2026-07-07", "2026-07-14"),
}


def _window_moves(lo: str, hi: str) -> dict:
    px = read_latest("prices").copy()
    px["obs_date"] = pd.to_datetime(px["obs_date"])
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    wide = px.pivot_table(index="obs_date", columns="ticker", values="close")
    win = wide.loc[lo:hi]
    if len(win) < 2:
        return {}
    move = win.iloc[-1] / win.iloc[0] - 1

    def basket(tks):
        vals = [move[t] for t in tks if t in move and pd.notna(move[t])]
        return float(np.mean(vals)) if vals else np.nan

    return {
        "brent": basket(["BZ=F"]),
        "tankers": basket(["FRO", "INSW", "TNK", "TRMD", "NAT", "STNG"]),
        "ovx": basket(["^OVX"]),
    }


def book_pnl(m: dict) -> dict:
    """Translate underlying moves into stylized sleeve P&L.
    Call spread: capped-delta long (0.35x brent move, floored at premium loss -1.5%
    of book). Puts: convex on the downside (gamma kicker), bleed otherwise."""
    if not m:
        return {}
    brent, tankers, ovx = m["brent"], m["tankers"], m["ovx"]
    call_spread = np.clip(0.35 * brent, -0.015, 0.10)
    tanker_pnl = tankers
    ovx_pnl = 0.25 * ovx if pd.notna(ovx) else 0.0
    put_pnl = (0.9 * abs(brent) if brent < -0.03 else -0.004)  # convex hedge vs theta
    sleeves = {
        "brent_call_spread": BOOK["brent_call_spread"] * call_spread,
        "tanker_basket": BOOK["tanker_basket"] * tanker_pnl,
        "ovx_proxy": BOOK["ovx_proxy"] * ovx_pnl,
        "deal_hedge_puts": BOOK["deal_hedge_puts"] * put_pnl,
    }
    sleeves["TOTAL"] = sum(sleeves.values())
    return sleeves


def main() -> int:
    print("=" * 64)
    print(" PEACE-SHOCK STRESS — stylized A1+A5 book vs this war's own tapes")
    print("=" * 64)
    print(f" book: {BOOK}")
    for name, (lo, hi) in WINDOWS.items():
        m = _window_moves(lo, hi)
        if not m:
            print(f"\n {name}: insufficient price data")
            continue
        pnl = book_pnl(m)
        print(f"\n {name}")
        print(f"   underlying: brent {m['brent']:+.1%}, tankers {m['tankers']:+.1%}, "
              f"ovx {m['ovx']:+.1%}" if pd.notna(m["ovx"]) else
              f"   underlying: brent {m['brent']:+.1%}, tankers {m['tankers']:+.1%}")
        for k, v in pnl.items():
            tag = "  <-- TOTAL" if k == "TOTAL" else ""
            print(f"   {k:<20} {v:+.2%}{tag}")
    print("\n RULE: if TOTAL in the deal_shock row is worse than -2% of book, "
          "the escalation sleeves are oversized relative to the hedge. "
          "Re-run after every book change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
