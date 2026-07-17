"""Daily brief — one command, four sections (M7):
  1. Current state estimate (regime + sub-state flags + Hawkes intensity)
  2. P vs Q table (model horizons vs market CDF, curve, RND, deal odds)
  3. Live signals A1-A6
  4. What changed since the previous brief (diffed against last snapshot)

Writes a markdown snapshot to data/briefs/brief_<date>.md and prints to stdout.

Run:  python -m src.report.daily_brief
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import DATA_DIR, load_config, read_latest, today_utc
from src.model.regime_markov import run as run_regime, STATES

BRIEF_DIR = DATA_DIR / "briefs"


def _fmt_pct(x, nd=0):
    return f"{x:.{nd}%}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n/a"


def section_state() -> list[str]:
    reg = run_regime()
    labels = reg["labels"]
    cur = labels.iloc[-1]
    p0 = reg["p0"]
    lines = ["## 1. State estimate", ""]
    dist = ", ".join(f"{s} {p:.0%}" for s, p in zip(STATES, p0) if p >= 0.01)
    lines.append(f"- **Current regime:** {dist} (week of {cur['week'].date()})")
    lines.append(f"- Hormuz 7dMA **{cur['ma7']:.1f}** calls/day = "
                 f"**{cur['frac']:.0%} of baseline** ({cur['baseline']:.0f})")

    # sub-state: S3 events in the last 10 days?
    try:
        ev = read_latest("coded_events").copy()
        ev["date"] = pd.to_datetime(ev["date"])
        recent = ev[ev["date"] >= ev["date"].max() - pd.Timedelta(days=10)]
        s3 = recent[recent["rung"] == "S3"]
        if len(s3):
            lines.append(f"- **Sub-state flag:** {len(s3)} S3 event(s) in last 10d "
                         f"(latest: {s3.iloc[-1]['action']}) — S2-with-S3-events, "
                         "the rung-or-transition question is live")
    except FileNotFoundError:
        pass

    try:
        hk = read_latest("hawkes").iloc[0]
        lines.append(f"- Hawkes branching **{hk['branching']:.2f}** "
                     f"(range {hk['branching_lo']:.2f}–{hk['branching_hi']:.2f}), "
                     f"intensity {hk['intensity_now']:.2f}/day vs {hk['baseline_intensity']:.2f} long-run")
    except FileNotFoundError:
        pass
    lines.append(f"- Posterior weight: {reg['data_weight']:.0%} data / "
                 f"{1 - reg['data_weight']:.0%} prior")
    return lines


def section_pq() -> list[str]:
    reg = run_regime()
    lines = ["", "## 2. P vs Q", ""]
    # P: model horizons
    lines.append("**P (model):**")
    for h in ("1m", "3m", "6m"):
        d = reg["forecasts"][h]
        lines.append(f"- {h}: " + ", ".join(f"{s} {p:.0%}" for s, p in zip(STATES, d) if p >= 0.05))
    t = reg["touch"]
    lines.append(f"- P(touch S4 before S5) **{t['p_touch_s4_before_s5']:.0%}**; "
                 f"median weeks to S5 when reached: {t['median_weeks_to_s5']:.0f}")

    lines.append("")
    lines.append("**Q (market):**")
    try:
        pm = read_latest("predmkt_panel")
        hz = pm[pm["family"] == "hormuz_normalize"].copy()
        hz = hz[hz["end_date"].notna()].sort_values("end_date")
        fwd = hz[hz["end_date"] > today_utc().isoformat()]
        cdf = "  ->  ".join(f"{r['end_date'][:7]}: {r['yes_prob']:.0%}"
                            for _, r in fwd.iterrows() if (r["volume"] or 0) > 100_000)
        lines.append(f"- Normalization CDF: {cdf}")
    except FileNotFoundError:
        lines.append("- predmkt panel not landed")
    try:
        from src.market_implied.curve import curve_metrics
        cm = curve_metrics(read_latest("futures_curve"))
        bz = cm.get("BZ", {})
        lines.append(f"- Brent {bz.get('front'):.2f}, 6m spread {bz.get('spread_6m'):+.2f}, "
                     f"12m {bz.get('spread_12m'):+.2f} (backwardation = disruption "
                     "priced as temporary)")
    except FileNotFoundError:
        pass
    try:
        rnd = read_latest("rnd")
        r = rnd[rnd["symbol"] == "USO"].iloc[-1]
        lines.append(f"- RND (USO {r['expiry']}): P(~Brent>100) {r['p_up16']:.0%}, "
                     f"P(~Brent<75) {r['p_dn13']:.0%}, ATM IV {r['atm_iv']:.0%} "
                     "(levels approximate; trust changes)")
    except (FileNotFoundError, IndexError):
        pass
    try:
        from src.market_implied.predmkt import deal_odds, build_panel
        dl = deal_odds(build_panel())
        if dl:
            lines.append(f"- Deal odds: **{dl['yes_prob']:.0%}** "
                         f"(\"{dl['question'][:55]}\")")
    except FileNotFoundError:
        pass
    return lines


def section_signals() -> list[str]:
    from src.alpha.signals import compute_all
    dirmap = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    lines = ["", "## 3. Signals", ""]
    lines.append("| # | direction | conf | key reading |")
    lines.append("|---|-----------|------|-------------|")
    for s in compute_all():
        name = s["signal"].split()[0]
        k, v = next(iter(s["value"].items())) if s["value"] else ("", "")
        if isinstance(v, dict):  # e.g. A5's deal-market blob — show the scalar
            v = v.get("yes_prob", "...")
        if isinstance(v, float):
            v = round(v, 3)
        lines.append(f"| {name} | {dirmap[s['direction']]} | {s['confidence']} "
                     f"| {k}={v} |")
    lines.append("")
    lines.append("_Full rationale/caveats: `make signals`. A5 is mandatory "
                 "whenever A1 is on._")
    return lines


def section_changes(body: str) -> list[str]:
    """Diff headline numbers against the most recent prior brief."""
    lines = ["", "## 4. What changed", ""]
    prior = sorted(BRIEF_DIR.glob("brief_*.md"))
    if not prior:
        lines.append("- First brief — no prior snapshot.")
        return lines
    prev = prior[-1].read_text()

    def grab(text, tag):
        for ln in text.splitlines():
            if tag in ln:
                return ln.strip()
        return None

    tags = ["Current regime", "Hormuz 7dMA", "Deal odds", "Brent ", "branching"]
    changed = False
    for tag in tags:
        old, new = grab(prev, tag), grab(body, tag)
        if old and new and old != new:
            lines.append(f"- was: {old}")
            lines.append(f"- now: {new}")
            changed = True
    if not changed:
        lines.append("- No headline changes vs last brief.")
    return lines


def main() -> int:
    today = today_utc().isoformat()
    parts = [f"# Iran escalation brief — {today}", ""]
    parts += section_state()
    parts += section_pq()
    parts += section_signals()
    body = "\n".join(parts)
    parts += section_changes(body)
    parts += ["", "---", "_Research framework, not financial advice. Q levels "
              "approximate (free/delayed data); P is prior-dominated (n=1 war). "
              "See CLAUDE.md standing weaknesses._"]
    out = "\n".join(parts)

    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    path = BRIEF_DIR / f"brief_{today}.md"
    path.write_text(out)
    print(out)
    print(f"\n[brief] saved -> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
