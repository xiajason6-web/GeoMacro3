"""Polymarket (Gamma API) — Iran + Hormuz prediction-market families.

Direct readout of Q for specific scenarios. Two hazards handled here:
  - Thinness: we store `volume` and `liquidity` so downstream can weight/ignore
    thin quotes. These are INFORMATION, never executable Q.
  - Criteria drift: we store the full `description` (resolution-criteria text)
    with every price, so a later resolution-rule change is auditable.

Run:  python -m src.ingest.polymarket
"""
from __future__ import annotations

import json
import sys

import pandas as pd
import requests

from src.common import load_config, utcnow, write_partition

SOURCE = "polymarket"


def _parse_json_field(val):
    """Gamma returns outcomes / outcomePrices as JSON-encoded strings."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


def fetch() -> pd.DataFrame:
    cfg = load_config("sources")["polymarket"]
    search_url = cfg["gamma_search"]
    terms = cfg["search_terms"]

    seen: dict[str, dict] = {}
    for term in terms:
        try:
            resp = requests.get(
                search_url,
                params={"q": term, "limit_per_type": 20, "events_status": "active"},
                headers={"Accept": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:  # keep going on a single-term failure
            print(f"[polymarket] warn: term {term!r} failed: {exc}", file=sys.stderr)
            continue

        events = resp.json().get("events", []) or []
        for ev in events:
            ev_desc = ev.get("description")
            for mkt in ev.get("markets", []) or []:
                mid = str(mkt.get("id"))
                if mid in seen:
                    continue
                outcomes = _parse_json_field(mkt.get("outcomes"))
                prices = _parse_json_field(mkt.get("outcomePrices"))
                seen[mid] = {
                    "market_id": mid,
                    "event_id": str(ev.get("id")),
                    "event_title": ev.get("title"),
                    "question": mkt.get("question"),
                    "slug": mkt.get("slug") or ev.get("slug"),
                    "outcomes": json.dumps(outcomes) if outcomes is not None else None,
                    "outcome_prices": json.dumps(prices) if prices is not None else None,
                    "last_trade_price": mkt.get("lastTradePrice"),
                    "volume": _to_float(mkt.get("volume") or ev.get("volume")),
                    "liquidity": _to_float(mkt.get("liquidity") or ev.get("liquidity")),
                    "end_date": mkt.get("endDate") or ev.get("endDate"),
                    "closed": mkt.get("closed"),
                    # resolution-criteria text — stored verbatim for audit
                    "resolution_criteria": mkt.get("description") or ev_desc,
                    "matched_term": term,
                    "pulled_at": utcnow().isoformat(),
                }

    df = pd.DataFrame(list(seen.values()))
    return df


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    df = fetch()
    if len(df) == 0:
        print("[polymarket] WARNING: 0 markets matched — nothing written", file=sys.stderr)
        return 1
    out = write_partition(df, SOURCE)
    print(f"[polymarket] {len(df)} markets -> {out}")
    # surface the Hormuz family specifically
    hz = df[df["question"].str.contains("Hormuz", case=False, na=False)]
    for _, r in hz.iterrows():
        print(f"  [Hormuz] {r['question']}  price={r['outcome_prices']}  vol={r['volume']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
