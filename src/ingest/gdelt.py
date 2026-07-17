"""GDELT 2.0 DOC API — conflict-intensity timelines + headline article lists.

Two products land in the lake:
  gdelt_timeline : daily article-volume series per tracked query (media-intensity
                   proxy for the escalation index; recall over precision).
  gdelt_articles : headline lists per query/window, the raw text the LLM coder
                   (src/coding) turns into rung-coded events.

GDELT enforces ~1 request / 5 seconds — every call goes through _throttled_get.
Violating it returns an HTML scold instead of JSON, which we detect and retry.

Run:  python -m src.ingest.gdelt            (default: last 7 days of articles)
      python -m src.ingest.gdelt backfill   (Feb 1 -> today, weekly windows)
"""
from __future__ import annotations

import datetime as dt
import sys
import time

import pandas as pd
import requests

from src.common import today_utc, write_partition

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
WAR_START = dt.date(2026, 2, 1)

# Tracked queries. Kept few and broad: GDELT gives recall, the coder gives precision.
QUERIES = {
    "kinetic": '(iran AND (strike OR strikes OR missile OR airstrike)) sourcelang:english',
    "maritime": '(iran AND (tanker OR hormuz OR shipping)) sourcelang:english',
    "diplomacy": '(iran AND (ceasefire OR deal OR negotiation OR talks)) sourcelang:english',
}

_last_call = [0.0]


def _throttled_get(params: dict) -> dict | None:
    """GET with 5.5s spacing and 429 backoff; returns JSON or None on failure.

    GDELT signals overuse two ways: an HTTP 429, or a 200 whose body is a plain-
    text scold instead of JSON. Both mean back off — the penalty window is about
    a minute, so retry up to 3 times with growing sleeps rather than dying.
    """
    for attempt in range(4):
        wait = 5.5 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(DOC_API, params=params, timeout=45)
        _last_call[0] = time.time()
        if resp.status_code == 429:
            backoff = 30 * (attempt + 1)
            print(f"[gdelt] 429, backing off {backoff}s", file=sys.stderr)
            time.sleep(backoff)
            continue
        resp.raise_for_status()
        text = resp.text.strip()
        if text.startswith("{"):
            return resp.json()
        backoff = 15 * (attempt + 1)
        print(f"[gdelt] scold-text response, backing off {backoff}s", file=sys.stderr)
        time.sleep(backoff)
    print(f"[gdelt] giving up on {params.get('query')!r}", file=sys.stderr)
    return None


def _fmt(d: dt.date, end: bool = False) -> str:
    return d.strftime("%Y%m%d") + ("235959" if end else "000000")


def fetch_timelines(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily raw article counts per query over [start, end] — one call per query."""
    frames = []
    for name, q in QUERIES.items():
        payload = _throttled_get({
            "query": q, "mode": "timelinevolraw", "format": "json",
            "startdatetime": _fmt(start), "enddatetime": _fmt(end, end=True),
        })
        if not payload:
            continue
        series = payload.get("timeline", [{}])[0].get("data", [])
        f = pd.DataFrame(series)
        if f.empty:
            continue
        f["obs_date"] = pd.to_datetime(f["date"]).dt.date.astype(str)
        f = f.groupby("obs_date", as_index=False)["value"].sum()
        f["query"] = name
        frames.append(f[["obs_date", "query", "value"]])
    if not frames:
        raise RuntimeError("GDELT returned no timeline data")
    return pd.concat(frames, ignore_index=True)


def fetch_articles(start: dt.date, end: dt.date, max_records: int = 75) -> pd.DataFrame:
    """Top headlines per query for one window (relevance-ranked by GDELT)."""
    rows = []
    for name, q in QUERIES.items():
        payload = _throttled_get({
            "query": q, "mode": "artlist", "format": "json",
            "maxrecords": max_records, "sort": "hybridrel",
            "startdatetime": _fmt(start), "enddatetime": _fmt(end, end=True),
        })
        if not payload:
            continue
        for a in payload.get("articles", []):
            rows.append({
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "query": name,
                "seendate": a.get("seendate"),
                "title": a.get("title"),
                "url": a.get("url"),
                "domain": a.get("domain"),
                "sourcecountry": a.get("sourcecountry"),
            })
    return pd.DataFrame(rows)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "recent"
    end = today_utc()

    if mode == "backfill":
        start = WAR_START
        tl = fetch_timelines(start, end)
        out = write_partition(tl, "gdelt_timeline")
        print(f"[gdelt] timeline {len(tl)} rows -> {out}")
        # weekly article windows
        frames = []
        w = start
        while w <= end:
            w_end = min(w + dt.timedelta(days=6), end)
            f = fetch_articles(w, w_end)
            if len(f):
                frames.append(f)
            print(f"[gdelt] articles {w} .. {w_end}: {len(f)} rows", file=sys.stderr)
            w = w_end + dt.timedelta(days=1)
        art = pd.concat(frames, ignore_index=True)
        out = write_partition(art, "gdelt_articles")
        print(f"[gdelt] articles total {len(art)} rows -> {out}")
    else:
        start = end - dt.timedelta(days=7)
        tl = fetch_timelines(WAR_START, end)  # timeline is cheap: refresh full history
        out = write_partition(tl, "gdelt_timeline")
        print(f"[gdelt] timeline {len(tl)} rows -> {out}")
        art = fetch_articles(start, end)
        if len(art):
            out = write_partition(art, "gdelt_articles")
            print(f"[gdelt] articles {len(art)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
