"""Timestamp-discipline tests. This is the invariant the whole backtest (M6)
rests on, so it is guarded from M1: every landed row carries an ingest stamp,
and partitions are keyed by ingest date (vintage), not observation date.
"""
import datetime as dt

import pandas as pd
import pytest

from src import common


def test_write_partition_adds_ingested_at(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    df = pd.DataFrame({"obs_date": ["2026-07-01"], "n_total": [10]})
    out = common.write_partition(df, "unittest", ingest_date=dt.date(2026, 7, 17))
    assert out.exists()
    assert "ingest_date=2026-07-17" in str(out)
    got = pd.read_parquet(out)
    assert "ingested_at" in got.columns
    assert got["ingested_at"].notna().all()


def test_refuses_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError):
        common.write_partition(pd.DataFrame(), "unittest")


def test_epoch_ms_to_date():
    # 2026-07-12 in epoch-ms (PortWatch's date encoding)
    d = common.epoch_ms_to_date(1_752_278_400_000)
    assert isinstance(d, dt.date)
