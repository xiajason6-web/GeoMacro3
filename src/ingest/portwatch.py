"""IMF PortWatch — daily transit calls at the Strait of Hormuz.

The keystone series. It is simultaneously the S2 threshold input to the regime
model (P) and Polymarket's resolution source (Q), so it is stored ONCE here and
referenced from both sides — the shared dependency is made explicit, not hidden,
so that a P-Q "divergence" that is really just this series' ~5-day publication
lag does not masquerade as genuine belief disagreement (see basis_monitor).

Run:  python -m src.ingest.portwatch
"""
from __future__ import annotations

import sys

import pandas as pd
import requests

from src.common import epoch_ms_to_date, load_config, write_partition

SOURCE = "portwatch"


def fetch() -> pd.DataFrame:
    cfg = load_config("sources")["portwatch"]
    server = cfg["feature_server"]
    chokepoint = cfg["chokepoint"]

    # ArcGIS caps each response at maxRecordCount (1000 here), so paginate via
    # resultOffset until the server stops flagging exceededTransferLimit. Without
    # this we'd silently get only the oldest 1000 rows and miss all recent data.
    features = []
    offset = 0
    page_size = 1000
    while True:
        params = {
            "where": f"portname='{chokepoint}'",
            "outFields": "date,portname,n_total,n_tanker,n_cargo,n_container,"
            "n_dry_bulk,n_general_cargo,n_roro",
            "orderByFields": "date ASC",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "json",
        }
        resp = requests.get(f"{server}/query", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if "features" not in payload:
            raise RuntimeError(f"unexpected PortWatch response: {str(payload)[:200]}")
        batch = payload["features"]
        features.extend(batch)
        if not payload.get("exceededTransferLimit") or len(batch) < page_size:
            break
        offset += page_size

    rows = []
    for feat in features:
        a = feat["attributes"]
        obs = epoch_ms_to_date(a.get("date"))
        rows.append(
            {
                "obs_date": obs.isoformat() if obs else None,
                "chokepoint": a.get("portname"),
                "n_total": a.get("n_total"),
                "n_tanker": a.get("n_tanker"),
                "n_cargo": a.get("n_cargo"),
                "n_container": a.get("n_container"),
                "n_dry_bulk": a.get("n_dry_bulk"),
                "n_general_cargo": a.get("n_general_cargo"),
                "n_roro": a.get("n_roro"),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["obs_date"]).sort_values("obs_date")
    return df.reset_index(drop=True)


def main() -> int:
    df = fetch()
    out = write_partition(df, SOURCE)
    latest = df.iloc[-1]
    lag_note = f"latest obs {latest['obs_date']} (n_total={latest['n_total']})"
    print(f"[portwatch] {len(df)} rows -> {out}")
    print(f"[portwatch] {lag_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
