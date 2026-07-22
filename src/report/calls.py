"""Calls ledger — auto-grading + Brier scoring for the public track record.

The ledger (calls/ledger.yaml) is append-only and git-timestamped; this module
grades open calls against the repo's own data lake and reports the running
Brier score. Runs inside every `make refresh` and the daily CI workflow, and
the daily brief prints the summary — so the track record accumulates and
self-grades without anyone remembering to do it.

Grading rules (criteria types):
  portwatch_ma7_gte — resolves YES the first day the Hormuz 7d-mean transit
      count >= threshold between `made` and `by`; NO once `by` passes.
  coded_event — resolves YES if any coded event with the required rung (and
      target_type, if specified) is dated in (made, by]; NO once `by` passes
      (with a 3-day grace window for coding lag before a NO is finalized).

Run:  python -m src.report.calls          # grade + print ledger
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.common import REPO_ROOT, read_latest, today_utc

LEDGER = REPO_ROOT / "calls" / "ledger.yaml"
GRACE_DAYS = 3  # coding/publication lag before a NO is finalized


def _grade_portwatch(c: dict) -> tuple[str, str] | None:
    pw = read_latest("portwatch").copy()
    pw["n_total"] = pd.to_numeric(pw["n_total"], errors="coerce")
    pw["obs_date"] = pd.to_datetime(pw["obs_date"])
    pw = pw.set_index("obs_date").sort_index()
    ma7 = pw["n_total"].rolling(7, min_periods=4).mean()
    made, by = pd.Timestamp(str(c["made"])), pd.Timestamp(str(c["criteria"]["by"]))
    window = ma7.loc[made:by]
    hits = window[window >= float(c["criteria"]["threshold"])]
    if len(hits):
        return "YES", f"7dMA {hits.iloc[0]:.0f} on {hits.index[0].date()}"
    # data lag: PortWatch publishes ~5d late — only finalize NO once data covers `by`
    if ma7.index.max() >= by:
        return "NO", f"never reached {c['criteria']['threshold']} through {by.date()}"
    if pd.Timestamp(today_utc()) > by + pd.Timedelta(days=GRACE_DAYS + 7):
        return "NO", "deadline passed (data lag grace exhausted)"
    return None


def _grade_coded_event(c: dict) -> tuple[str, str] | None:
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    made, by = pd.Timestamp(str(c["made"])), pd.Timestamp(str(c["criteria"]["by"]))
    sel = ev[(ev["date"] > made) & (ev["date"] <= by)
             & (ev["rung"] == c["criteria"]["rung"])]
    tts = c["criteria"].get("target_types")
    if tts:
        sel = sel[sel["target_type"].isin(tts)]
    if len(sel):
        r = sel.sort_values("date").iloc[0]
        return "YES", f"{r['date'].date()}: {str(r['action'])[:60]}"
    if pd.Timestamp(today_utc()) > by + pd.Timedelta(days=GRACE_DAYS):
        return "NO", f"no qualifying event through {by.date()}"
    return None


def _ma7(c: dict):
    pw = read_latest("portwatch").copy()
    pw["n_total"] = pd.to_numeric(pw["n_total"], errors="coerce")
    pw["obs_date"] = pd.to_datetime(pw["obs_date"])
    return pw.set_index("obs_date").sort_index()["n_total"].rolling(
        7, min_periods=4).mean()


def _grade_portwatch_consecutive(c: dict) -> tuple[str, str] | None:
    """YES if the 7dMA holds >= threshold for `days` consecutive calendar days
    at any point between made and by — the DURABLE-normalization variant."""
    ma7 = _ma7(c)
    made, by = pd.Timestamp(str(c["made"])), pd.Timestamp(str(c["criteria"]["by"]))
    need = int(c["criteria"]["days"])
    win = (ma7.loc[made:by] >= float(c["criteria"]["threshold"]))
    run = (win.groupby((~win).cumsum()).cumsum())
    if (run >= need).any():
        d = run[run >= need].index[0]
        return "YES", f"{need}d sustained >= {c['criteria']['threshold']} by {d.date()}"
    if ma7.index.max() >= by:
        return "NO", f"no {need}-day sustained recovery through {by.date()}"
    if pd.Timestamp(today_utc()) > by + pd.Timedelta(days=GRACE_DAYS + 7):
        return "NO", "deadline passed (data lag grace exhausted)"
    return None


def _grade_portwatch_on_date(c: dict) -> tuple[str, str] | None:
    """YES if the 7dMA >= threshold on the LAST observation at/before `on` —
    the occupancy ('is the war over on that date') variant."""
    ma7 = _ma7(c)
    on = pd.Timestamp(str(c["criteria"]["on"]))
    upto = ma7.loc[:on].dropna()
    # only grade once data reaches (or grace passes) the target date
    if ma7.index.max() < on:
        if pd.Timestamp(today_utc()) > on + pd.Timedelta(days=GRACE_DAYS + 7):
            pass  # fall through and grade on the last available obs
        else:
            return None
    if not len(upto):
        return None
    val = float(upto.iloc[-1])
    ok = val >= float(c["criteria"]["threshold"])
    return ("YES" if ok else "NO",
            f"7dMA {val:.0f} on {upto.index[-1].date()} (last obs before {on.date()})")


def _grade_next_s5_brent(c: dict) -> tuple[str, str] | None:
    """Framework discriminator: does the NEXT deal episode arrive in a week
    with average Brent >= threshold (price-thermostat) or below it (fiscal
    clock)? Resolves NO if no qualifying deal arrives by the deadline."""
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    made, by = pd.Timestamp(str(c["made"])), pd.Timestamp(str(c["criteria"]["by"]))
    s5 = ev[(ev["rung"] == "S5") & (ev["date"] > made) & (ev["date"] <= by)]
    if len(s5):
        d0 = s5.sort_values("date").iloc[0]["date"]
        px = read_latest("prices")
        bz = px[px["ticker"] == "BZ=F"].copy()
        bz["obs_date"] = pd.to_datetime(bz["obs_date"])
        wk = bz[(bz["obs_date"] >= d0 - pd.Timedelta(days=3))
                & (bz["obs_date"] <= d0 + pd.Timedelta(days=3))]
        if not len(wk):
            return None  # wait for price data to cover the deal week
        avg = float(pd.to_numeric(wk["close"], errors="coerce").mean())
        ok = avg >= float(c["criteria"]["threshold"])
        return ("YES" if ok else "NO",
                f"deal {d0.date()}, Brent week-avg ${avg:.0f} "
                f"({'thermostat' if ok else 'fiscal-clock'} regime)")
    if pd.Timestamp(today_utc()) > by + pd.Timedelta(days=GRACE_DAYS):
        return "NO", f"no deal episode arrived by {by.date()}"
    return None


GRADERS = {"portwatch_ma7_gte": _grade_portwatch,
           "next_s5_brent": _grade_next_s5_brent,
           "portwatch_ma7_gte_consecutive": _grade_portwatch_consecutive,
           "portwatch_ma7_gte_on_date": _grade_portwatch_on_date,
           "coded_event": _grade_coded_event}


def load() -> dict:
    with open(LEDGER) as fh:
        return yaml.safe_load(fh)


def grade(write: bool = True) -> dict:
    doc = load()
    changed = False
    for c in doc["calls"]:
        if c.get("status") != "open":
            continue
        fn = GRADERS.get(c["criteria"]["type"])
        if not fn:
            continue
        try:
            res = fn(c)
        except FileNotFoundError:
            res = None
        if res:
            outcome, evidence = res
            c["status"] = "resolved"
            c["outcome"] = outcome
            c["evidence"] = evidence
            c["resolved_at"] = today_utc().isoformat()
            changed = True
    if changed and write:
        with open(LEDGER, "w") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, width=100)
    return doc


def summary(doc: dict | None = None) -> dict:
    doc = doc or load()
    calls = doc["calls"]
    resolved = [c for c in calls if c.get("status") == "resolved"]
    briers = [(float(c["p"]) - (1.0 if c["outcome"] == "YES" else 0.0)) ** 2
              for c in resolved]
    return {
        "n_calls": len(calls),
        "n_open": sum(1 for c in calls if c.get("status") == "open"),
        "n_resolved": len(resolved),
        "brier": (sum(briers) / len(briers)) if briers else None,
        "first_call": min(str(c["made"]) for c in calls) if calls else None,
    }


def main() -> int:
    doc = grade(write=True)
    s = summary(doc)
    print(f"CALLS LEDGER — {s['n_calls']} calls since {s['first_call']}: "
          f"{s['n_open']} open, {s['n_resolved']} resolved"
          + (f", Brier {s['brier']:.3f}" if s["brier"] is not None else ""))
    for c in doc["calls"]:
        flag = {"open": "○", "resolved": "●"}.get(c.get("status"), "?")
        line = f" {flag} [{c['made']}] p={c['p']:.0%}  {c['claim'][:74]}"
        if c.get("status") == "resolved":
            line += f"\n     -> {c['outcome']} ({c.get('evidence','')})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
