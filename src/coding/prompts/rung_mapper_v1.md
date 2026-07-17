# Rung mapper — v1 (frozen 2026-07-17)

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
  (per config/taxonomy.yaml: S1 military tit-for-tat, S2 maritime/chokepoint,
   S3 third-country Gulf infrastructure, S4 all-out/regime targets, S5 deal/ceasefire)
- `summary`: one sentence

Rules:
- One object per underlying event, not per headline.
- Diplomatic events (talks, deals, ceasefires) are rung S5, target_type `none`.
- Strikes on desalination/water/energy infra in third countries (Saudi, Kuwait,
  Qatar, UAE, Oman) are S3 regardless of what else is happening.
- If headlines conflict, prefer the majority reading; note uncertainty in summary.

Output: a JSON array, nothing else.
