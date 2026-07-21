"""Risk-neutral density from USO/BNO option chains (Breeden–Litzenberger).

Pipeline: pull chains for the nearest liquid expiries -> clean (OI, bid>0) ->
fit a smooth IV curve in log-moneyness (weighted cubic polynomial) -> price a
dense call grid with Black–Scholes -> second difference in strike = density.

Honesty flags (design-doc §4.1 and weakness #2): ETF options embed fund roll
mechanics and are delayed — the density's SHAPE and day-over-day CHANGES are
informative; absolute levels are approximate. Thresholds are reported in ETF
return space with approximate Brent equivalents (proportional mapping), clearly
labeled as such.

Run:  python -m src.market_implied.rnd
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from src.common import today_utc, write_partition

MIN_OI = 5
MIN_EXPIRY_DAYS = 10
N_EXPIRIES = 3
# ETF-return thresholds with approximate Brent equivalents (Brent ~86 today;
# proportional move mapping — approximate by construction).
UP_THRESHOLDS = [0.16, 0.40]    # ~ Brent > 100, > 120
DOWN_THRESHOLD = -0.13          # ~ Brent < 75


def _bs_call(F, K, T, iv):
    d1 = (np.log(F / K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    return F * norm.cdf(d1) - K * norm.cdf(d2)  # zero-rate approx, fine for shape


def density_from_calls(calls: pd.DataFrame, spot: float, expiry: str) -> dict | None:
    """Core Breeden-Litzenberger given a calls frame with columns
    strike / impliedVolatility / openInterest / bid."""
    T = (pd.Timestamp(expiry).date() - today_utc()).days / 365.0
    if T <= 0:
        return None
    calls = calls.copy()
    calls = calls[(calls["openInterest"].fillna(0) >= MIN_OI) & (calls["bid"] > 0)]
    calls = calls.dropna(subset=["impliedVolatility"])
    calls = calls[(calls["impliedVolatility"] > 0.05) & (calls["impliedVolatility"] < 3.0)]
    if len(calls) < 8:
        return None

    k = np.log(calls["strike"].values / spot)
    iv = calls["impliedVolatility"].values
    w = np.sqrt(calls["openInterest"].fillna(1).values)  # weight liquid strikes
    coef = np.polyfit(k, iv, deg=min(3, len(calls) // 5), w=w)

    # dense strike grid, clipped to a sane moneyness band
    kg = np.linspace(max(k.min(), -0.6), min(k.max(), 0.9), 400)
    ivg = np.clip(np.polyval(coef, kg), 0.05, 3.0)
    Kg = spot * np.exp(kg)
    C = _bs_call(spot, Kg, T, ivg)

    dens = np.gradient(np.gradient(C, Kg), Kg)  # d2C/dK2
    dens = np.clip(dens, 0, None)
    total = np.trapezoid(dens, Kg)
    if total <= 0:
        return None
    dens = dens / total  # renormalize (truncated support)

    def prob_above(ret):
        thresh = spot * (1 + ret)
        mask = Kg >= thresh
        return float(np.trapezoid(dens[mask], Kg[mask])) if mask.any() else 0.0

    atm_iv = float(np.polyval(coef, 0.0))
    # 25-delta-ish risk reversal proxy: iv at +/-15% moneyness
    rr = float(np.polyval(coef, np.log(1.15)) - np.polyval(coef, np.log(0.87)))
    return {
        "expiry": expiry,
        "days": int(T * 365),
        "atm_iv": atm_iv,
        "risk_reversal_15pct": rr,
        "p_up16": prob_above(UP_THRESHOLDS[0]),
        "p_up40": prob_above(UP_THRESHOLDS[1]),
        "p_dn13": 1.0 - prob_above(DOWN_THRESHOLD),
        "n_strikes_used": int(len(calls)),
    }


def _cboe_chain(symbol: str) -> tuple[float, dict]:
    """CBOE's official public delayed-quotes JSON — cleaner than yfinance
    (real bid/ask, exchange-computed IV, OI, all strikes). Returns
    (spot, {expiry: calls_df}). Raises on any failure -> caller falls back."""
    import re

    import requests

    r = requests.get(
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json",
        timeout=30, headers={"User-Agent": "iran-escalation-model/1.0 (research)"})
    r.raise_for_status()
    data = r.json()["data"]
    spot = float(data["current_price"])
    pat = re.compile(rf"^{symbol}(\d{{6}})([CP])(\d{{8}})$")
    by_exp: dict[str, list] = {}
    for o in data["options"]:
        m = pat.match(o.get("option", ""))
        if not m or m.group(2) != "C":
            continue
        yymmdd, strike = m.group(1), int(m.group(3)) / 1000.0
        expiry = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
        by_exp.setdefault(expiry, []).append({
            "strike": strike, "bid": o.get("bid") or 0.0,
            "impliedVolatility": o.get("iv"),
            "openInterest": o.get("open_interest") or 0.0,
        })
    return spot, {e: pd.DataFrame(rows) for e, rows in by_exp.items()}


def fetch(symbol: str = "USO") -> pd.DataFrame:
    # preferred: CBOE official delayed feed; fallback: yfinance chains
    rows = []
    try:
        spot, chains = _cboe_chain(symbol)
        expiries = sorted(e for e in chains
                          if (pd.Timestamp(e).date() - today_utc()).days >= MIN_EXPIRY_DAYS)
        for e in expiries[:N_EXPIRIES]:
            d = density_from_calls(chains[e], spot, e)
            if d:
                d.update({"symbol": symbol, "spot": spot, "source": "cboe"})
                rows.append(d)
        if rows:
            return pd.DataFrame(rows)
    except Exception:  # noqa: BLE001 — fall through to yfinance
        pass

    tk = yf.Ticker(symbol)
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"no spot for {symbol}")
    spot = float(hist["Close"].iloc[-1])
    expiries = [e for e in (tk.options or ())
                if (pd.Timestamp(e).date() - today_utc()).days >= MIN_EXPIRY_DAYS]
    for e in expiries[:N_EXPIRIES]:
        try:
            calls = tk.option_chain(e).calls
        except Exception:  # noqa: BLE001
            continue
        d = density_from_calls(calls, spot, e)
        if d:
            d.update({"symbol": symbol, "spot": spot, "source": "yfinance"})
            rows.append(d)
    return pd.DataFrame(rows)


def main() -> int:
    frames = [f for s in ("USO", "BNO") if len(f := fetch(s))]
    if not frames:
        print("[rnd] no usable chains", file=sys.stderr)
        return 1
    df = pd.concat(frames, ignore_index=True)
    out = write_partition(df, "rnd")
    print(f"[rnd] {len(df)} expiry-densities -> {out}")
    for _, r in df.iterrows():
        print(f"[rnd] {r['symbol']} {r['expiry']} ({r['days']}d): atm_iv={r['atm_iv']:.0%} "
              f"P(+16%~Brent>100)={r['p_up16']:.1%} P(+40%~Brent>120)={r['p_up40']:.1%} "
              f"P(-13%~Brent<75)={r['p_dn13']:.1%} rr={r['risk_reversal_15pct']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
