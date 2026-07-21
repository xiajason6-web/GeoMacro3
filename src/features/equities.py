"""Equities read-through — how the war thesis maps to tradeable equity buckets.

Buckets and their thesis linkage:
  tankers      war-risk premia + rerouting economics lift rates; but hulls are
               themselves S2 targets — long persistence, short direct-attack tail
  defense      the production-deficit story IS the revenue story: magazines spent
               at a 15:1 replacement gap must be rebuilt in EVERY scenario,
               including settlement — the least war-path-dependent bucket
  energy_eq    oil-beta with balance-sheet damping
  gulf_markets KSA/UAE/QAT ETFs — the S3 (Gulf-infrastructure) axis is direct
               risk to these markets; the free proxy for Gulf sovereign risk
               (true CDS is paid data)
  airlines     fuel-cost and airspace victim bucket

Computed per bucket, all from the existing price lake + coded events:
  - cumulative return since war outbreak (2026-02-27)
  - return over the June détente window (Jun 13-19) and July re-escalation
    (Jul 7-14) — the sign pattern identifies which side of the war each bucket
    actually trades on
  - S3-day sensitivity: mean daily return on days with a coded S3 event vs
    all other war days (the lateral-axis beta, measured not asserted)

Run:  python -m src.features.equities
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import read_latest, write_partition

BUCKETS = {
    "tankers": ["FRO", "INSW", "TNK", "TRMD", "NAT", "STNG"],
    "defense": ["ITA", "PPA"],
    "energy equity": ["XLE", "XOP", "OIH"],
    "gulf markets": ["KSA", "UAE", "QAT"],
    "airlines": ["JETS"],
}
WAR_START = "2026-02-26"
DETENTE = ("2026-06-13", "2026-06-19")
REESCALATION = ("2026-07-07", "2026-07-14")


def _panel() -> pd.DataFrame:
    px = read_latest("prices").copy()
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px["obs_date"] = pd.to_datetime(px["obs_date"])
    wide = px.pivot_table(index="obs_date", columns="ticker", values="close")
    out = pd.DataFrame(index=wide.index)
    for b, tks in BUCKETS.items():
        cols = [t for t in tks if t in wide]
        if cols:
            # equal-weight bucket index, normalized returns
            out[b] = wide[cols].pct_change(fill_method=None).mean(axis=1)
    return out


def readthrough() -> pd.DataFrame:
    rets = _panel()
    ev = read_latest("coded_events").copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    s3_days = set(ev.loc[ev["rung"] == "S3", "date"].dt.normalize())

    war = rets.loc[WAR_START:]
    idx_norm = war.index.normalize()
    on_s3 = war[idx_norm.isin(s3_days)]
    off_s3 = war[~idx_norm.isin(s3_days)]

    rows = []
    for b in rets.columns:
        since_war = float((1 + war[b].dropna()).prod() - 1)
        det = rets.loc[DETENTE[0]:DETENTE[1], b]
        ree = rets.loc[REESCALATION[0]:REESCALATION[1], b]
        rows.append({
            "bucket": b,
            "since_war": since_war,
            "detente_jun": float((1 + det.dropna()).prod() - 1),
            "reescalation_jul": float((1 + ree.dropna()).prod() - 1),
            "s3_day_avg": float(on_s3[b].mean()) if len(on_s3) else np.nan,
            "other_day_avg": float(off_s3[b].mean()) if len(off_s3) else np.nan,
            "n_s3_days": int(on_s3[b].notna().sum()),
        })
    df = pd.DataFrame(rows)
    df["s3_sensitivity"] = df["s3_day_avg"] - df["other_day_avg"]
    return df


def variant_tilt(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the P−Q divergence to the buckets: expected 3m return differential
    under OUR scenario probabilities vs the MARKET's.

    Decomposition: E[3m] ≈ grind drift + p(resolution)·(resolution episode
    payoff) + p(escalation)·(escalation episode payoff). The grind term is
    common to both measures, so the DIFFERENTIAL is exactly
        tilt = Δp_res · payoff_res + Δp_esc · payoff_esc
    where the payoffs are each bucket's MEASURED June-détente and
    July-re-escalation window returns.

    Probability pairings (object-matching is approximate; documented):
      resolution: model first-passage P(Hormuz normal by Sep 30) vs the
        Polymarket September contract — identical resolution criteria.
      escalation: model P(visit S4 within 3m) vs the options-implied
        P(≈Brent>100) — the best available market proxy for the escalation
        tail; an imperfect object match, flagged as such.
    """
    import numpy as np

    from src.model.scorecard import derived_strength
    from src.model.regime_markov import run as _run

    strength = derived_strength()
    r = _run(strength, use_covariates=True)
    T, p0 = r["T"], r["p0"]
    rng = np.random.default_rng(20260721)
    N = 20000
    stv = rng.choice(6, size=N, p=p0 / p0.sum())
    first = np.full(N, 999)
    for wk in range(1, 30):
        u = rng.random(N)
        cum = T[stv].cumsum(axis=1)
        stv = np.clip((u[:, None] > cum).sum(axis=1), 0, 5)
        hit = (stv == 0) & (first == 999)
        first[hit] = wk
    import datetime as _dt
    wks_sep = max(1, round((_dt.date(2026, 9, 30) - _dt.date.today()).days / 7))
    p_res_model = float((first <= wks_sep).mean())
    p_esc_model = float(r["touch"].get("p_visit_s4_3m", 0.3))

    pm = read_latest("predmkt_panel")
    hz = pm[(pm["family"] == "hormuz_normalize")
            & (pm["end_date"].astype(str).str.startswith("2026-09"))]
    p_res_mkt = float(hz.sort_values("volume", ascending=False).iloc[0]["yes_prob"]) \
        if len(hz) else np.nan
    try:
        rnd = read_latest("rnd")
        p_esc_mkt = float(rnd[rnd["symbol"] == "USO"].iloc[-1]["p_up16"])
    except Exception:  # noqa: BLE001
        p_esc_mkt = np.nan

    d_res = p_res_model - p_res_mkt
    d_esc = p_esc_model - p_esc_mkt
    df = df.copy()
    df["tilt_3m"] = d_res * df["detente_jun"] + d_esc * df["reescalation_jul"]
    df["p_res_model"], df["p_res_mkt"] = p_res_model, p_res_mkt
    df["p_esc_model"], df["p_esc_mkt"] = p_esc_model, p_esc_mkt
    return df


def main() -> int:
    df = readthrough()
    try:
        df = variant_tilt(df)
    except Exception as exc:  # noqa: BLE001 — descriptive table still lands
        print(f"[equities] variant tilt unavailable: {exc}", file=sys.stderr)
    out = write_partition(df, "equities_readthrough")
    print(f"[equities] read-through -> {out}")
    has_tilt = "tilt_3m" in df.columns
    for _, r in df.iterrows():
        line = (f"  {r['bucket']:<14} since-war {r['since_war']:+7.1%}  "
                f"détente {r['detente_jun']:+6.1%}  re-esc {r['reescalation_jul']:+6.1%}  "
                f"S3-day edge {r['s3_sensitivity']:+.2%}/d (n={r['n_s3_days']})")
        if has_tilt:
            line += f"  VARIANT TILT {r['tilt_3m']:+.2%}"
        print(line)
    if has_tilt:
        r0 = df.iloc[0]
        print(f"[equities] tilt inputs: resolution P {r0['p_res_model']:.0%} vs "
              f"Q {r0['p_res_mkt']:.0%} (Δ{r0['p_res_model']-r0['p_res_mkt']:+.0%}); "
              f"escalation P {r0['p_esc_model']:.0%} vs Q {r0['p_esc_mkt']:.0%} "
              f"(Δ{r0['p_esc_model']-r0['p_esc_mkt']:+.0%})")
        print("[equities] tilt = Δp_res·(détente payoff) + Δp_esc·(re-esc payoff): "
              "the expected 3m return differential IF our probabilities are right "
              "and the market's are wrong. Positive = own more than the market "
              "would; grind drift cancels in the differential.")
    print("[equities] caveats: episode payoffs from single windows; escalation "
          "market proxy is options-implied P(Brent>100) — an imperfect object "
          "match; n small throughout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
