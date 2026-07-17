"""Land the frozen coded backfill (config/coded_events_backfill.yaml) into the
lake as `coded_events` and `rhetoric` vintages.

The YAML is the canonical frozen history (versioned in git); this loader is
idempotent — rerunning just refreshes today's vintage from the same file. The
live coder (llm_coder.py) APPENDS new windows; it never rewrites these rows.

Run:  python -m src.coding.load_backfill
"""
from __future__ import annotations

import sys

import pandas as pd

from src.common import CONFIG_DIR, write_partition
import yaml

CODER_VERSION = "manual-opus-4.8-20260717"
SOURCE_NOTE = "wikipedia:2026_Iran_war (fetched 2026-07-17)"


def main() -> int:
    with open(CONFIG_DIR / "coded_events_backfill.yaml") as fh:
        raw = yaml.safe_load(fh)

    ev = pd.DataFrame(raw["events"])
    ev["date"] = ev["date"].astype(str)
    ev["coder_version"] = CODER_VERSION
    ev["prompt_version"] = "rung_mapper_v1"
    ev["source"] = SOURCE_NOTE
    out = write_partition(ev, "coded_events")
    print(f"[backfill] {len(ev)} coded events -> {out}")
    print(f"[backfill] rung counts: {ev['rung'].value_counts().to_dict()}")

    rh = pd.DataFrame(raw["rhetoric"])
    rh["date"] = rh["date"].astype(str)
    rh["coder_version"] = CODER_VERSION
    rh["prompt_version"] = "rhetoric_v1"
    rh["source"] = SOURCE_NOTE
    out = write_partition(rh, "rhetoric")
    print(f"[backfill] {len(rh)} rhetoric rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
