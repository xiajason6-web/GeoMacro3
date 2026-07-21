"""EIA weekly petroleum balances — the real fundamentals control input.

KEY-GATED: needs EIA_API_KEY in .env (free, instant: eia.gov/opendata — the
one registration step only the user can do). Keyless fallbacks are all dead:
FRED discontinued its EIA weekly mirrors (WCESTUS1 etc. return errors) and the
WPSR public CSVs are gone. Without a key this module no-ops with a message and
the fundamentals control runs on its 5 keyless drivers.

Pulls weekly U.S. ending stocks of crude oil (ex-SPR) — the single series that
does the most confounder-cleaning work: high stocks = loose fundamentals = the
Tanker-War failure mode where the war premium bleeds while the ladder holds.

Run:  python -m src.ingest.eia
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import requests

from src.common import write_partition

API = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
SERIES = "WCESTUS1"  # weekly U.S. ending stocks of crude oil excl. SPR, kbbl


def fetch() -> pd.DataFrame:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError(
            "EIA_API_KEY not set — register free at eia.gov/opendata and add "
            "to .env; the fundamentals control runs without inventories until then"
        )
    resp = requests.get(API, params={
        "api_key": key, "frequency": "weekly",
        "data[0]": "value", "facets[series][]": SERIES,
        "start": "2024-01-01", "sort[0][column]": "period",
        "sort[0][direction]": "asc", "length": 5000,
    }, timeout=45)
    resp.raise_for_status()
    rows = resp.json()["response"]["data"]
    df = pd.DataFrame([{"obs_date": r["period"], "crude_stocks_kbbl": float(r["value"])}
                       for r in rows])
    return df


def main() -> int:
    try:
        df = fetch()
    except RuntimeError as exc:
        print(f"[eia] skipped: {exc}", file=sys.stderr)
        return 0  # graceful: absence is expected until the key exists
    out = write_partition(df, "eia_stocks")
    print(f"[eia] {len(df)} weekly crude-stock observations -> {out}")
    print(f"[eia] latest: {df.iloc[-1]['obs_date']} = "
          f"{df.iloc[-1]['crude_stocks_kbbl']/1000:.0f} Mbbl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
