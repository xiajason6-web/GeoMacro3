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
RUNG_PROMPT_VERSION = "rung_mapper_v1"
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
    if all_events:
        out = write_partition(pd.DataFrame(all_events), "coded_events")
        print(f"[coder] {len(all_events)} events -> {out}")
    if all_scores:
        out = write_partition(pd.DataFrame(all_scores), "rhetoric")
        print(f"[coder] {len(all_scores)} rhetoric rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
