"""Market prices via yfinance — crude, vol, energy/tanker/cross-asset equities.

M1 lands front-month continuous crude + ETFs + vol + equity baskets. The
individual futures-month ladder (front-to-6M / front-to-12M curve spreads, §4.2)
is NOT reliably available on free yfinance and is deferred to M3; do not infer
curve shape from this module.

Stored long/tidy: one row per (obs_date, ticker) with close + volume, tagged
with the `group` it belongs to so the basis monitor can pull the tanker basket
directly.

Run:  python -m src.ingest.prices
"""
from __future__ import annotations

import sys

import pandas as pd
import yfinance as yf

from src.common import load_config, write_partition

SOURCE = "prices"


def fetch() -> pd.DataFrame:
    cfg = load_config("sources")["prices"]
    lookback = int(cfg.get("lookback_days", 400))

    groups = {
        "crude_front": cfg["crude_front"],
        "crude_etf": cfg["crude_etf"],
        "vol": cfg["vol"],
        "energy_equity": cfg["energy_equity"],
        "tanker_equity": cfg["tanker_equity"],
        "cross_asset": cfg["cross_asset"],
    }
    ticker_group = {t: g for g, ts in groups.items() for t in ts}
    all_tickers = list(ticker_group.keys())

    # threads=False: yfinance's shared SQLite tz-cache deadlocks ("database is
    # locked") under concurrent access, silently dropping tickers. Serial is
    # slightly slower but reliable — reliability wins for a daily pull.
    raw = yf.download(
        all_tickers,
        period=f"{lookback}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )

    def tidy(sub, t):
        if sub is None or sub.empty or "Close" not in sub:
            return None
        f = sub.reset_index()[["Date", "Close", "Volume"]].copy()
        f.columns = ["obs_date", "close", "volume"]
        f = f.dropna(subset=["close"])
        if f.empty:
            return None
        f["obs_date"] = pd.to_datetime(f["obs_date"]).dt.date.astype(str)
        f["ticker"] = t
        f["group"] = ticker_group[t]
        return f

    frames, missing = [], []
    for t in all_tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            sub = None
        f = tidy(sub, t)
        (frames.append(f) if f is not None else missing.append(t))

    # retry any dropped ticker individually (transient cache/network failures)
    for t in missing:
        try:
            sub = yf.download(t, period=f"{lookback}d", interval="1d",
                              auto_adjust=False, progress=False, threads=False)
            f = tidy(sub, t)
            if f is not None:
                frames.append(f)
                print(f"[prices] recovered {t} on retry", file=sys.stderr)
            else:
                print(f"[prices] warn: {t} still empty after retry", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - report and move on
            print(f"[prices] warn: {t} retry failed: {exc}", file=sys.stderr)

    if not frames:
        raise RuntimeError("yfinance returned no usable data for any ticker")
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "obs_date"])


def main() -> int:
    df = fetch()
    out = write_partition(df, SOURCE)
    got = sorted(df["ticker"].unique())
    print(f"[prices] {len(df)} rows across {len(got)} tickers -> {out}")
    # quick sanity: latest Brent
    bz = df[df["ticker"] == "BZ=F"]
    if len(bz):
        last = bz.sort_values("obs_date").iloc[-1]
        print(f"[prices] Brent (BZ=F) latest {last['obs_date']}: {last['close']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
