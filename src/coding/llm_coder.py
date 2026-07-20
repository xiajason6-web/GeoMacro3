"""LLM coder — turns headline batches into rung-coded events + rhetoric scores.

Backend resolution, in order:
  1. ANTHROPIC_API_KEY set  -> Anthropic API (claude-sonnet-4-6)
  2. `claude` CLI on PATH   -> `claude -p` headless
  3. neither                -> RuntimeError with instructions

Coder-drift discipline (design-doc weakness #6):
  - Prompts live in src/coding/prompts/, versioned in the filename (…_v1.md) and
    frozen once used. A prompt change = new version file + explicit re-code.
  - Every coded row is stamped with `coder_version` and `prompt_version`.
  - The Feb–Jul 2026 backfill was coded in-session by Claude Opus 4.8 on
    2026-07-17 (coder_version=manual-opus-4.8-20260717) from GDELT headline
    windows, and is FROZEN in config/coded_events_backfill.yaml — the live coder
    appends, it never re-codes history.

Run:  python -m src.coding.llm_coder            (code latest gdelt_articles vintage)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.common import read_latest, write_partition

PROMPTS = Path(__file__).parent / "prompts"
RUNG_PROMPT_VERSION = "rung_mapper_v2"  # v2 migration 2026-07-17: S4 strictness
#                                         + escalatory-diplomacy-is-not-S5 (QA vs backfill)
RHETORIC_PROMPT_VERSION = "rhetoric_v1"
API_MODEL = "claude-sonnet-4-6"


def _backend() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    if shutil.which("claude"):
        return "cli"
    raise RuntimeError(
        "No LLM backend: set ANTHROPIC_API_KEY or install the claude CLI. "
        "(The Feb–Jul 2026 backfill is already frozen in "
        "config/coded_events_backfill.yaml and does not need a backend.)"
    )


def _call_llm(system_prompt: str, user_text: str) -> str:
    be = _backend()
    if be == "api":
        import requests

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": API_MODEL,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_text}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    # cli
    out = subprocess.run(
        ["claude", "-p", "--append-system-prompt", system_prompt],
        input=user_text, capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {out.stderr[:200]}")
    return out.stdout


def _parse_json_array(text: str) -> list[dict]:
    """Extract the first JSON array from an LLM response."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in response: {text[:200]}")
    return json.loads(text[start : end + 1])


def code_window(headlines: pd.DataFrame, coder_version: str) -> tuple[list[dict], list[dict]]:
    """Code one window of headlines -> (events, rhetoric_scores)."""
    lines = [
        f"- [{r['seendate']}] ({r['query']}) {r['title']}"
        for _, r in headlines.iterrows()
        if r.get("title")
    ]
    body = "\n".join(lines)
    window = f"{headlines['window_start'].iloc[0]} .. {headlines['window_end'].iloc[0]}"

    rung_sys = (PROMPTS / f"{RUNG_PROMPT_VERSION}.md").read_text()
    events = _parse_json_array(_call_llm(rung_sys, f"Window: {window}\n\nHeadlines:\n{body}"))
    for e in events:
        e["coder_version"] = coder_version
        e["prompt_version"] = RUNG_PROMPT_VERSION

    rh_sys = (PROMPTS / f"{RHETORIC_PROMPT_VERSION}.md").read_text()
    scores = _parse_json_array(_call_llm(rh_sys, f"Window: {window}\n\nHeadlines:\n{body}"))
    for s in scores:
        s["coder_version"] = coder_version
        s["prompt_version"] = RHETORIC_PROMPT_VERSION

    return events, scores


def _frozen_backfill() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the canonical frozen history from the YAML (never from the lake —
    a prior bad vintage must not poison the merge)."""
    import yaml

    from src.common import CONFIG_DIR

    with open(CONFIG_DIR / "coded_events_backfill.yaml") as fh:
        raw = yaml.safe_load(fh)
    ev = pd.DataFrame(raw["events"])
    ev["coder_version"] = "manual-opus-4.8-20260717"
    ev["prompt_version"] = "rung_mapper_v1"
    ev["source"] = "wikipedia:2026_Iran_war (fetched 2026-07-17)"
    rh = pd.DataFrame(raw["rhetoric"])
    rh["coder_version"] = "manual-opus-4.8-20260717"
    rh["prompt_version"] = "rhetoric_v1"
    rh["source"] = "wikipedia:2026_Iran_war (fetched 2026-07-17)"
    for df in (ev, rh):
        df["date"] = df["date"].astype(str)
    return ev, rh


def main() -> int:
    art = read_latest("gdelt_articles")
    coder_version = f"{_backend()}-{API_MODEL}"
    all_events, all_scores = [], []
    for (ws, we), grp in art.groupby(["window_start", "window_end"]):
        ev, sc = code_window(grp, coder_version)
        all_events.extend(ev)
        all_scores.extend(sc)
        print(f"[coder] {ws}..{we}: {len(ev)} events, {len(sc)} rhetoric rows",
              file=sys.stderr)

    # Raw live codings land separately for audit/QA — never merged blindly.
    if all_events:
        out = write_partition(pd.DataFrame(all_events), "coded_events_live_raw")
        print(f"[coder] raw live codings -> {out}")

    # Merge policy: the frozen backfill is the AUTHORITATIVE spine (hand-verified,
    # never re-coded). Live codings ENRICH the trailing OVERLAP_DAYS: they add
    # net-new events the sparse backfill missed in recent weeks (so the current
    # week doesn't go dark just because the freeze predates it), but the frozen
    # row always wins a key collision, and live can never rewrite deep history.
    # This replaced a strict "> cutoff" append that silently dropped every live
    # event inside the backfill's final week — blinding the 8c spread index
    # exactly when the war was most active.
    OVERLAP_DAYS = 21
    ev_frozen, rh_frozen = _frozen_backfill()

    def merge(frozen: pd.DataFrame, live_new: list[dict], source: str,
              dedupe_keys: list[str]) -> pd.DataFrame:
        cutoff = pd.Timestamp(frozen["date"].max())
        window_start = (cutoff - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
        today = pd.DataFrame(live_new)
        # carry forward live rows appended by PRIOR runs (headlines age out of the
        # sliding window). Prior coded_events = frozen + live, so recover the live
        # part by coder_version (frozen rows are tagged manual-*).
        try:
            prior = read_latest(source)
            prior_live = prior[~prior["coder_version"].astype(str).str.startswith("manual-")]
        except (FileNotFoundError, KeyError):
            prior_live = pd.DataFrame()
        live = pd.concat([prior_live, today], ignore_index=True)
        if len(live):
            live = live[live["date"].astype(str) >= window_start]
            # frozen precedence: drop any live row colliding with a frozen key
            fk = set(map(tuple, frozen[dedupe_keys].astype(str).values))
            keep = ~live[dedupe_keys].astype(str).apply(tuple, axis=1).isin(fk)
            live = live[keep].drop_duplicates(subset=dedupe_keys, keep="first")
        merged = pd.concat([frozen, live], ignore_index=True).sort_values("date")
        print(f"[coder] {source} = {len(frozen)} frozen + {len(live)} live "
              f"(net-new, trailing {OVERLAP_DAYS}d from {window_start}, frozen-precedence)")
        return merged

    merged_ev = merge(ev_frozen, all_events, "coded_events",
                      ["date", "actor", "rung", "target_type"])
    out = write_partition(merged_ev, "coded_events")
    print(f"[coder] -> {out}")

    merged_rh = merge(rh_frozen, all_scores, "rhetoric", ["date", "actor"])
    out = write_partition(merged_rh, "rhetoric")
    print(f"[coder] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
