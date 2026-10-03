"""Indicative check on Official-like windows: 15 days starting Oct 8-16, 2021-2025.

Oct 12-30 is peak Q3 earnings season. Ideas rejected on the full history
(earnings trims, gross 0.7, VIX overlay) are re-scored on these windows only.
There are only 5 Octobers, and the windows overlap, so this is a tiebreaker for
the post-Validation decisions, not an adoption test.
Writes reports/october_check.json.
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eyris import sentiment  # noqa: E402
from eyris.backtest import evaluate_windows, rank_in_field, target_policy  # noqa: E402
from eyris.config import Params  # noqa: E402
from eyris.execution import sanitize  # noqa: E402
from event_experiment import EventStudy  # noqa: E402
from sentiment_experiment import load_vix  # noqa: E402

START_DAYS = (8, 16)   # window start day-of-month range in October


def candidates(cur):
    trim = replace(cur, event_mode="trim", event_cut=1.0, event_min_move=0.03)
    return {
        "current": cur,
        "gross_0.7": replace(cur, gross=0.7),
        "earnings_trim_full": trim,
        "earnings_trim_half": replace(trim, event_cut=0.5),
        "vix_level25_x1.4": replace(cur, vix_mode="level", vix_threshold=25.0, vix_boost=1.4),
    }


def evaluate(es, q, vix, keep):
    st, p, r = es.st, es.st.p, es.st.r
    targets = st.targets(q)
    if q.vix_mode != "off":
        m = sentiment.multipliers(vix, p.days, q)
        targets = np.array([sanitize(t * m[d]) for t, d in zip(targets, r.day[st.ks])])
    flags = None
    if q.event_mode != "off":
        flags = lambda k: es.sched.flags(p.days[r.day[k]], int(r.number[k]), q)  # noqa: E731
    pol = target_policy(targets, st.k_first, q, flags)
    wins = [w for w, k in zip(st.wins, keep) if k]
    fields = {v: [f for f, k in zip(st.field_for(variant=v), keep) if k] for v in ("full", "active", "llm")}
    df = evaluate_windows(r, wins, pol, fields["full"])
    recs = df.to_dict("records")
    for v in ("active", "llm"):
        df[f"rank_{v}"] = [rank_in_field(row, f) for row, f in zip(recs, fields[v])]
    df["year"] = [p.days[r.day[k0]].year for k0, _ in wins]
    return df


def main():
    t0 = time.time()
    cur = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    vix = load_vix()
    frames = []
    for period in ("train", "holdout"):
        es = EventStudy(period, stride=1)
        p, r = es.st.p, es.st.r
        starts = [p.days[r.day[k0]] for k0, _ in es.st.wins]
        keep = [s.month == 10 and START_DAYS[0] <= s.day <= START_DAYS[1] for s in starts]
        for name, q in candidates(cur).items():
            df = evaluate(es, q, vix, keep)
            df["name"] = name
            frames.append(df)
        print(period, sum(keep), "windows", f"{time.time() - t0:.0f}s", flush=True)
    all_df = pd.concat(frames, ignore_index=True)
    cols = ["rank_score", "rank_active", "rank_llm", "cumulative_return", "maximum_drawdown", "turnover"]
    overall = all_df.groupby("name")[cols].mean()
    overall["windows"] = all_df.groupby("name").size()
    by_year = all_df.pivot_table(index="year", columns="name", values="rank_score", aggfunc="mean")
    print(overall.round(4).to_string())
    print("mean full-field rank by year:")
    print(by_year.round(3).to_string())
    out = {"overall": overall.reset_index().to_dict("records"),
           "by_year": by_year.reset_index().to_dict("records"),
           "rank_llm_by_year": all_df.pivot_table(index="year", columns="name", values="rank_llm",
                                                  aggfunc="mean").reset_index().to_dict("records")}
    (ROOT / "reports" / "october_check.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
