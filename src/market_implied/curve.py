"""Futures curve — the market's revealed duration estimate for the disruption.

Downloads the Brent (BZ) and WTI (CL) monthly contract ladder from yfinance
(verified available free, e.g. BZU26.NYM), lands it in the lake, and reports:
  - front-to-6M and front-to-12M spreads (backwardation = disruption priced as
    temporary; flattening after events = market repricing persistence)
  - the full ladder for the daily brief

Run:  python -m src.market_implied.curve
"""
from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
import yfinance as yf

from src.common import today_utc, write_partition

MONTH_CODES = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]


def _ladder_tickers(root: str, start: dt.date, n_months: int = 15) -> list[tuple[str, str]]:
    """(ticker, contract_month) for the next n monthly contracts from start."""
    out = []
    y, m = start.year, start.month
    for _ in range(n_months):
        m += 1
        if m > 12:
            m, y = 1, y + 1
        code = MONTH_CODES[m - 1]
        out.append((f"{root}{code}{str(y)[2:]}.NYM", f"{y}-{m:02d}"))
    return out


def fetch() -> pd.DataFrame:
    rows = []
    for root in ("BZ", "CL"):
        for ticker, cmonth in _ladder_tickers(root, today_utc()):
            try:
                h = yf.Ticker(ticker).history(period="5d")
            except Exception:
                continue
            if h is None or h.empty:
                continue
            rows.append({
                "obs_date": h.index[-1].date().isoformat(),
                "root": root,
                "ticker": ticker,
                "contract_month": cmonth,
                "close": float(h["Close"].iloc[-1]),
            })
    if not rows:
        raise RuntimeError("no futures ladder data returned")
    return pd.DataFrame(rows)


def curve_metrics(ladder: pd.DataFrame) -> dict:
    out = {}
    for root, grp in ladder.groupby("root"):
        g = grp.sort_values("contract_month").reset_index(drop=True)
        front = g.iloc[0]
        def spread(months):
            if len(g) > months:
                return float(front["close"] - g.iloc[months]["close"])
            return None
        out[root] = {
            "front_month": front["contract_month"],
            "front": float(front["close"]),
            "spread_6m": spread(6),
            "spread_12m": spread(12),
            "n_contracts": len(g),
            "ladder": list(zip(g["contract_month"], g["close"].round(2))),
        }
    return out


def main() -> int:
    ladder = fetch()
    out = write_partition(ladder, "futures_curve")
    print(f"[curve] {len(ladder)} contracts -> {out}")
    m = curve_metrics(ladder)
    for root, d in m.items():
        shape = "backwardation" if (d["spread_6m"] or 0) > 0 else "contango"
        print(f"[curve] {root}: front {d['front']:.2f} ({d['front_month']}), "
              f"6m spread {d['spread_6m']:+.2f}, 12m {d['spread_12m'] or float('nan'):+.2f} "
              f"-> {shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
