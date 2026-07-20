"""Fundamentals control — separate the war/persistence premium from soft demand.

This is the confounder-cleaner for alpha #1 (the curve back-end edge) and a
refinement of 8b's US-oil-pain: a high Brent only pressures the US politically if
it is WAR-driven, not if it is a macro/fundamentals move.

Method (keyless — FRED CSV + yfinance, no API key):
  1. Pull Brent + the drivers that move oil ABSENT geopolitics: broad USD, 10y
     breakeven, 10y real yield (FRED), and — the factor that actually matters —
     copper and the S&P (yfinance) as cyclical GLOBAL-DEMAND proxies. Macro rates
     alone explain ~7% of oil; adding copper lifts the pre-war fit to ~72%.
  2. Estimate Brent ~ drivers by OLS on a PRE-WAR window (2024-01 .. 2026-01), so
     the fitted relationship is uncontaminated by the war.
  3. fundamentals_fair = the pre-war model applied to today's drivers. The residual
     (actual − fair) is the GEOPOLITICAL / WAR PREMIUM.

Alpha #1 read: compare the war premium to the futures backwardation (front − 12M).
  - backwardation ≈ war premium  -> market prices the ENTIRE premium fading by the
    back end (full normalization priced).
  - backwardation < war premium   -> market prices only PARTIAL fade -> some
    persistence already in the curve.
  - premium priced at the back end ≈ 0 while the model says persistence is likely
    -> the divergence that IS alpha #1, now clean of a demand explanation.

Honest limits: a 3-factor macro proxy, not a full supply/demand balance (that
still wants EIA). FRED Brent (spot) differs from the yfinance front future used
elsewhere — this module stays internally consistent on FRED for the decomposition.

Run:  python -m src.features.fundamentals
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd
import requests

from src.common import read_latest, write_partition

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES = {"brent": "DCOILBRENTEU", "usd": "DTWEXBGS",
          "breakeven": "T10YIE", "real_yield": "DFII10"}
YF_SERIES = {"copper": "HG=F", "spx": "^GSPC"}   # cyclical global-demand proxies
PREWAR_START = "2024-01-01"
PREWAR_END = "2026-01-31"      # war opens 2026-02-28; leave a clean margin
DRIVERS = ["usd", "breakeven", "real_yield", "copper", "spx"]


def _fred(series_id: str, start: str = PREWAR_START) -> pd.Series:
    r = requests.get(FRED, params={"id": series_id, "cosd": start}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
    return df.dropna().set_index("date")["value"]


def _panel() -> pd.DataFrame:
    import yfinance as yf
    cols = {name: _fred(sid) for name, sid in SERIES.items()}
    df = pd.DataFrame(cols)
    yfd = yf.download(list(YF_SERIES.values()), start=PREWAR_START,
                      progress=False)["Close"]
    yfd.columns = list(YF_SERIES.keys())
    yfd.index = pd.to_datetime(yfd.index).tz_localize(None)
    df = df.join(yfd).sort_index().ffill().dropna()
    return df


def decompose() -> dict:
    df = _panel()
    pre = df.loc[PREWAR_START:PREWAR_END]
    if len(pre) < 60:
        raise RuntimeError(f"insufficient pre-war macro data (n={len(pre)})")

    X = np.column_stack([np.ones(len(pre))] + [pre[d].values for d in DRIVERS])
    y = pre["brent"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)

    latest = df.iloc[-1]
    x_now = np.array([1.0] + [latest[d] for d in DRIVERS])
    fair = float(x_now @ coef)
    actual = float(latest["brent"])
    premium = actual - fair

    out = {
        "as_of": str(df.index[-1].date()),
        "brent_fred": actual, "fundamentals_fair": fair,
        "war_premium": premium, "prewar_r2": float(r2),
        "coef": {"intercept": float(coef[0]),
                 **{d: float(c) for d, c in zip(DRIVERS, coef[1:])}},
        "macro_now": {d: float(latest[d]) for d in DRIVERS},
    }

    # alpha #1: how much of the war premium does the curve keep at the 12M horizon?
    #   12M Brent = spot - backwardation; premium still priced at back = that minus
    #   fundamentals-fair. frac_persisting = share of today's premium the market
    #   still prices at 12M. Low => market prices normalization => alpha #1 (if the
    #   model says persistence). High => persistence already in the curve.
    try:
        from src.market_implied.curve import curve_metrics
        bz = curve_metrics(read_latest("futures_curve"))["BZ"]
        backwardation = bz["spread_12m"]  # front - 12M (positive = backwardation)
        premium_at_back = premium - (backwardation or 0.0)
        frac = (premium_at_back / premium) if premium > 1 else None
        if frac is None:
            read = "no meaningful war premium to decompose"
        elif frac < 0.25:
            read = ("market prices NEAR-FULL normalization by 12M — persistence "
                    "NOT in the curve => alpha #1 strong if the model holds")
        elif frac < 0.60:
            read = (f"market prices PARTIAL persistence (~{frac:.0%} of the "
                    "premium remains at 12M) => alpha #1 mild — edge only if your "
                    "persistence view exceeds this")
        elif premium_at_back >= 0:
            read = (f"market already prices STRONG persistence (~{frac:.0%} of "
                    "premium remains at 12M) => alpha #1 largely arbitraged")
        else:
            read = ("back end prices the premium fully gone AND fundamentals "
                    "softening further")
        out.update({"backwardation_12m": backwardation,
                    "premium_priced_at_back": premium_at_back,
                    "frac_premium_persisting": frac, "read": read})
    except Exception:  # noqa: BLE001
        pass
    return out


def main() -> int:
    try:
        d = decompose()
    except Exception as exc:  # noqa: BLE001
        print(f"[fundamentals] failed: {exc}", file=sys.stderr)
        return 1
    write_partition(pd.DataFrame([{k: v for k, v in d.items()
                                   if not isinstance(v, dict)}]), "fundamentals")
    print(f"[fundamentals] as of {d['as_of']} (FRED spot Brent)")
    print(f"[fundamentals] pre-war model R^2 = {d['prewar_r2']:.2f} "
          "(Brent ~ USD + breakeven + real yield + copper + S&P, 2024-01..2026-01)")
    print(f"[fundamentals] Brent ${d['brent_fred']:.1f} = fundamentals-fair "
          f"${d['fundamentals_fair']:.1f} + WAR PREMIUM ${d['war_premium']:+.1f}")
    if "backwardation_12m" in d:
        print(f"[fundamentals] 12M backwardation ${d['backwardation_12m']:+.1f}; "
              f"premium the curve keeps at 12M ${d['premium_priced_at_back']:+.1f} "
              f"({d['frac_premium_persisting']:.0%} of it)"
              if d.get("frac_premium_persisting") is not None else "")
        print(f"[fundamentals] ALPHA #1: {d['read']}")
    print("[fundamentals] NOTE: macro+demand proxy (copper is the key factor), not "
          "a supply/demand balance (full version wants EIA). FRED spot differs "
          "from the front future used elsewhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
