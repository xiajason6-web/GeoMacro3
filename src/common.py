"""Shared plumbing: config loading, the parquet lake, and timestamp discipline.

The one rule that everything else depends on: every row we land carries an
`ingested_at` UTC timestamp, and partitions are keyed by the date WE pulled the
data (`ingest_date=`), not by the observation date inside the data. That is the
point-in-time / vintage discipline the backtest (M6) will rely on — so that a
walk-forward can never see a value before the day we could actually have pulled
it. PortWatch, for example, lags ~5 days; storing it by ingest date makes that
lag visible and non-cheatable instead of silently backfilled.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"


def utcnow() -> _dt.datetime:
    """Timezone-aware UTC now. Single source of 'now' for the whole codebase."""
    return _dt.datetime.now(_dt.timezone.utc)


def today_utc() -> _dt.date:
    return utcnow().date()


def load_config(name: str) -> dict[str, Any]:
    """Load config/<name>.yaml (name may include or omit the .yaml suffix)."""
    if not name.endswith((".yaml", ".yml")):
        name = name + ".yaml"
    with open(CONFIG_DIR / name) as fh:
        return yaml.safe_load(fh)


def write_partition(
    df: pd.DataFrame,
    source: str,
    ingest_date: _dt.date | None = None,
) -> Path:
    """Write a DataFrame to data/<source>/ingest_date=YYYY-MM-DD/data.parquet.

    Adds `ingested_at` (UTC, ISO string) to every row if not already present.
    Overwrites that day's partition — one clean vintage per source per day.
    """
    if df is None or len(df) == 0:
        raise ValueError(f"refusing to write empty frame for source={source!r}")

    ingest_date = ingest_date or today_utc()
    df = df.copy()
    if "ingested_at" not in df.columns:
        df["ingested_at"] = utcnow().isoformat()

    part_dir = DATA_DIR / source / f"ingest_date={ingest_date.isoformat()}"
    part_dir.mkdir(parents=True, exist_ok=True)
    out = part_dir / "data.parquet"
    df.to_parquet(out, index=False)
    return out


def read_latest(source: str) -> pd.DataFrame:
    """Read the most recent vintage we have for a source (by ingest_date)."""
    root = DATA_DIR / source
    parts = sorted(root.glob("ingest_date=*/data.parquet"))
    if not parts:
        raise FileNotFoundError(f"no vintages landed yet for source={source!r}")
    return pd.read_parquet(parts[-1])


def epoch_ms_to_date(ms: Any) -> _dt.date | None:
    """Convert an ArcGIS date to a UTC date.

    ArcGIS is inconsistent: depending on the query it returns dates as
    epoch-milliseconds (numeric), an epoch-ms string, or a formatted date
    string. Handle all three.
    """
    if ms is None or (isinstance(ms, float) and pd.isna(ms)):
        return None
    if isinstance(ms, str):
        s = ms.strip()
        if s.isdigit():  # epoch-ms encoded as a string
            return _dt.datetime.utcfromtimestamp(int(s) / 1000).date()
        parsed = pd.to_datetime(s, errors="coerce", utc=True)  # formatted string
        return None if pd.isna(parsed) else parsed.date()
    return _dt.datetime.utcfromtimestamp(ms / 1000).date()
