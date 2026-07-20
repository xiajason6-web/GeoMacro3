"""Google News RSS — keyless headline fallback for the LLM coder.

GDELT's DOC API enforces aggressive per-IP rate limits and its penalties can
outlast a whole session. This module lands the same schema into the SAME lake
source (`gdelt_articles`, provider column distinguishes) so the coder is
agnostic to where headlines came from. RSS gives ~100 recent headlines per
query — plenty for coding the last few days; it cannot backfill history.

Run:  python -m src.ingest.newsrss
"""
from __future__ import annotations

import datetime as dt
import sys
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from src.common import today_utc, write_partition

RSS_URL = "https://news.google.com/rss/search"

QUERIES = {
    "kinetic": "iran strike OR missile OR airstrike",
    "maritime": "iran tanker OR hormuz OR shipping",
    "diplomacy": "iran ceasefire OR deal OR negotiation",
}


def fetch(lookback_days: int = 14) -> pd.DataFrame:
    start = today_utc() - dt.timedelta(days=lookback_days)
    rows = []
    for name, q in QUERIES.items():
        try:
            resp = requests.get(
                RSS_URL,
                params={"q": f"{q} when:{lookback_days}d", "hl": "en-US",
                        "gl": "US", "ceid": "US:en"},
                timeout=30,
                headers={"User-Agent": "iran-escalation-model/1.0 (research)"},
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"[newsrss] warn: {name} failed: {exc}", file=sys.stderr)
            continue
        for item in root.iter("item"):
            title = item.findtext("title")
            pub = item.findtext("pubDate")
            try:
                seen = pd.to_datetime(pub).strftime("%Y%m%d%H%M%S")
            except (ValueError, TypeError):
                seen = None
            src = item.find("source")
            rows.append({
                "window_start": start.isoformat(),
                "window_end": today_utc().isoformat(),
                "query": name,
                "seendate": seen,
                "title": title,
                "url": item.findtext("link"),
                "domain": src.text if src is not None else None,
                "sourcecountry": None,
                "provider": "google_news_rss",
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["title"])
    return df


def main() -> int:
    df = fetch()
    if df.empty:
        print("[newsrss] no headlines returned", file=sys.stderr)
        return 1
    out = write_partition(df, "gdelt_articles")
    print(f"[newsrss] {len(df)} headlines -> {out}")
    for q, g in df.groupby("query"):
        print(f"  [{q}] {len(g)} headlines, e.g. {g.iloc[0]['title'][:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
