# Rhetoric scorer — v1 (frozen 2026-07-17)

You are scoring official-statement rhetoric for an escalation model of the
US–Iran war. Given headlines/statement excerpts from one date window, score the
overall rhetorical posture of each principal actor.

For each actor with enough signal (`US`, `IRAN`; optionally `ISRAEL`, `GCC`)
output one JSON object:

- `date`: window date, YYYY-MM-DD
- `actor`: the actor
- `rhetoric_score`: −2 … +2
    -2 explicit de-escalation offer (ceasefire proposal, unconditional talks)
    -1 conciliatory signal (openness to talks, restraint language)
     0 neutral / boilerplate / mixed
    +1 escalation signal (new threats, red lines, "all options")
    +2 explicit escalation threat (naming new target classes, ultimatums)
- `commitment`: `cheap_talk` | `costly` (mobilization, deployments, legislation)
- `audience`: `domestic` | `adversary` | `allies` | `mixed`
- `evidence`: one short quote or paraphrase

Rules:
- Score the posture of the window, not single outlier quotes.
- Costly signals dominate cheap talk when they conflict.

Output: a JSON array, nothing else.
