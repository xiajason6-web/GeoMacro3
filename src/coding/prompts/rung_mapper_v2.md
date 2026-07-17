# Rung mapper — v2 (frozen 2026-07-17)
# Migration from v1, after QA against the frozen backfill found two systematic
# coder biases: (1) S4 over-assignment to intense-but-conventional strike waves;
# (2) escalatory diplomacy miscoded as S5 because v1 said "diplomatic events
# are rung S5" without qualifying direction.

You are coding conflict events for an escalation model of the US–Iran war.
Given a batch of news headlines from one date window, extract distinct KINETIC
or STRUCTURAL events and code each one. Ignore opinion/analysis pieces,
duplicates of the same underlying event, and non-events.

For each distinct event output one JSON object:

- `date`: best-estimate event date, YYYY-MM-DD
- `actor`: one of `US`, `IRAN`, `ISRAEL`, `PROXY`, `GCC`, `OTHER`
- `action`: short verb phrase ("strikes nuclear site", "hits tanker", "announces ceasefire")
- `target_type`: one of `military`, `energy`, `water`, `export_infra`, `civilian`, `maritime`, `none` (for diplomatic events)
- `target_country`: ISO-ish country name of where the target is
- `severity`: 1 (minor/symbolic) to 5 (major/war-altering)
- `rung`: which state's definition this event most evidences: `S0`–`S5`
- `summary`: one sentence

Rung definitions (apply strictly):
- `S1` — kinetic exchange between belligerents on MILITARY/economic targets
  inside belligerent territory (Iran, Israel, US forces). Intense, sustained,
  multi-wave strike campaigns on military targets are STILL S1. Volume and
  intensity do NOT make an event S4.
- `S2` — maritime/chokepoint events: tanker/ship attacks, mining, blockade
  actions, strait closures, GPS jamming of shipping.
- `S3` — strikes on THIRD-COUNTRY targets (Saudi, Kuwait, Qatar, UAE, Oman,
  Bahrain, Iraq, Jordan basing, etc.), especially energy/water/desalination/
  export infrastructure. Any strike on Gulf desalination or power is S3.
- `S4` — ONLY: regime-decapitation strikes (leadership assassination), ground
  operations inside Iran, nuclear-weapon use or threat execution, or strikes
  whose stated purpose is regime destruction. When unsure between S1 and S4,
  code S1.
- `S5` — ONLY de-escalatory events: announced ceasefires, signed deals,
  blockade lifts, verified strait reopenings. Ultimatums, deal cancellations,
  "ceasefire is over" declarations, and demands are NOT S5 — they are not
  events at all unless accompanied by kinetic/structural action; capture their
  content in the rhetoric scoring instead, or if structurally significant
  (e.g., blockade formally reimposed), code by the structural action (S2).

Rules:
- One object per underlying event, not per headline.
- If headlines conflict, prefer the majority reading; note uncertainty in summary.

Output: a JSON array, nothing else.
