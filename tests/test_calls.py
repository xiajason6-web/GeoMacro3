"""Calls-ledger discipline tests: grading logic + Brier arithmetic."""
import pandas as pd
import pytest

from src.report import calls


def test_brier_math():
    doc = {"calls": [
        {"made": "2026-07-01", "p": 0.8, "status": "resolved", "outcome": "YES"},
        {"made": "2026-07-01", "p": 0.8, "status": "resolved", "outcome": "NO"},
        {"made": "2026-07-01", "p": 0.5, "status": "open"},
    ]}
    s = calls.summary(doc)
    assert s["n_resolved"] == 2 and s["n_open"] == 1
    # (0.8-1)^2 = 0.04 ; (0.8-0)^2 = 0.64 ; mean = 0.34
    assert abs(s["brier"] - 0.34) < 1e-9


def test_portwatch_grading_yes_and_pending(monkeypatch):
    idx = pd.date_range("2026-07-01", periods=14, freq="D")
    frame = pd.DataFrame({"obs_date": idx, "n_total": [70] * 14})

    monkeypatch.setattr(calls, "read_latest", lambda *_: frame)
    call_yes = {"made": "2026-07-01", "p": 0.5,
                "criteria": {"type": "portwatch_ma7_gte", "threshold": 60,
                             "by": "2026-07-20"}}
    res = calls._grade_portwatch(call_yes)
    assert res is not None and res[0] == "YES"

    # threshold never met and data doesn't yet cover `by` -> stays open (None)
    frame2 = frame.assign(n_total=10)
    monkeypatch.setattr(calls, "read_latest", lambda *_: frame2)
    call_open = {"made": "2026-07-01", "p": 0.5,
                 "criteria": {"type": "portwatch_ma7_gte", "threshold": 60,
                              "by": "2099-01-01"}}
    assert calls._grade_portwatch(call_open) is None


def test_ledger_file_parses_and_calls_are_wellformed():
    doc = calls.load()
    assert len(doc["calls"]) >= 6
    for c in doc["calls"]:
        assert 0.0 <= float(c["p"]) <= 1.0
        assert c["criteria"]["type"] in calls.GRADERS
        assert ("by" in c["criteria"]) or ("on" in c["criteria"])
