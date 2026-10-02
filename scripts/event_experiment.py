"""Backtest the earnings-event risk control.

The Official window (Oct 12-30) is Q3 earnings season, so the primary test uses
"earnings-season windows" (>= MIN_EVENTS announcements inside the 15 days).
Selection: mean median-rank percentile over the full / active / LLM fields on
those windows (validation 2025-01..09), tie-break worst window. Adopt only if it
beats the current model there, does not lose on train earnings-season windows,
and does not lose on all validation windows.
Writes reports/event_experiment.{json,md}.
"""
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris import events  # noqa: E402
from eyris.backtest import evaluate_windows, rank_in_field, target_policy  # noqa: E402
from eyris.config import Params  # noqa: E402
from eyris.experiment import Study  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from agentic_research import select_score  # noqa: E402

MIN_EVENTS = 15


class EventStudy:
    def __init__(self, period, stride=1):
        self.st = Study(period, stride=stride)
        p, r = self.st.p, self.st.r
        cal = events.load_calendar()
        moves = events.historical_moves(p, cal)
        self.sched = events.EventSchedule(cal, events.trading_dates(p.days), events.trailing_move_fn(moves))
        jumps = self.sched.events["jump"].values
        self.n_events = []
        for k0, k1 in self.st.wins:
            d0, d1 = p.days[r.day[k0]], p.days[r.day[k1 - 1]]
            self.n_events.append(int(((jumps >= np.datetime64(d0)) & (jumps <= np.datetime64(d1))).sum()))
        self.n_events = np.array(self.n_events)
        self.sizes = {"score": len(self.st.field_for()[0]) + 1,
                      "active": len(self.st.field_for(variant="active")[0]) + 1,
                      "llm": len(self.st.field_for(variant="llm")[0]) + 1}

    def evaluate(self, params):
        st, p, r = self.st, self.st.p, self.st.r
        flags = None
        if params.event_mode != "off":
            flags = lambda k: self.sched.flags(p.days[r.day[k]], int(r.number[k]), params)  # noqa: E731
        pol = target_policy(st.targets(params), st.k_first, params, flags)
        df = evaluate_windows(r, st.wins, pol, st.field_for())
        recs = df.to_dict("records")
        for v in ("active", "llm"):
            df[f"rank_{v}"] = [rank_in_field(row, f) for row, f in zip(recs, st.field_for(variant=v))]
        df["n_events"] = self.n_events
        return df


def summary(df):
    row = {}
    for col, name in (("rank_score", "score"), ("rank_active", "active"), ("rank_llm", "llm")):
        row[f"rank_{name}_median"] = df[col].median()
        row[f"rank_{name}_worst"] = df[col].max()
    if "rank_score_median" in row:
        row["rank_score_median"], row["rank_score_worst"] = row["rank_score_median"], row["rank_score_worst"]
    row.update(ret_median=df.cumulative_return.median(), mdd_median=df.maximum_drawdown.median(),
               mdd_worst=df.maximum_drawdown.max(), turnover_median=df.turnover.median(),
               sharpe_median=df.sharpe_ratio.median(), windows=len(df))
    return row


def scored(es, cands, season_only):
    rows = []
    for name, q in cands:
        df = es.evaluate(q)
        if season_only:
            df = df[df.n_events >= MIN_EVENTS]
        rows.append({"name": name, **q.to_dict(), **summary(df)})
    return select_score(pd.DataFrame(rows), es.sizes)


def main():
    t0 = time.time()
    cur = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    cands = [("current", cur)]
    for mode, cut, mm, rr in itertools.product(("trim", "trim_restore"), (0.5, 1.0), (0.0, 0.03, 0.05),
                                               (1, 2)):
        if mode == "trim" and rr == 2:
            continue
        q = replace(cur, event_mode=mode, event_cut=cut, event_min_move=mm, event_restore_round=rr)
        cands.append((f"{mode} cut={cut} min_move={mm} restore_r={rr}", q))
    val = EventStudy("validation")
    print(f"validation windows {len(val.n_events)}, earnings-season windows {(val.n_events >= MIN_EVENTS).sum()}")
    season = scored(val, cands, True)
    cols = ["name", "selection", "pct_full", "pct_active", "pct_llm", "worst_pct", "ret_median", "mdd_median",
            "mdd_worst", "turnover_median", "windows"]
    print(season[cols].round(4).to_string())
    best_name = season.iloc[0]["name"]
    best = dict(cands)[best_name] if best_name != "current" else None
    allwin = scored(val, cands, False)
    print("all validation windows:")
    print(allwin[cols].head(8).round(4).to_string())

    picks = [("current", cur)] + ([(best_name, best)] if best is not None else [])
    checks = {}
    for period, stride in (("train", 2), ("holdout", 1)):
        es = EventStudy(period, stride=stride)
        checks[period] = {"season": scored(es, picks, True), "all": scored(es, picks, False)}
        for k, v in checks[period].items():
            print(period, k)
            print(v[cols].round(4).to_string())

    def sel(df, name):
        return float(df.set_index("name").loc[name, "selection"])
    adopt = False
    if best is not None:
        adopt = (sel(season, best_name) < sel(season, "current") - 1e-9
                 and sel(checks["train"]["season"], best_name) <= sel(checks["train"]["season"], "current") + 1e-9
                 and sel(allwin, best_name) <= sel(allwin, "current") + 1e-9)
    out = {"min_events": MIN_EVENTS, "best": best_name, "adopt": adopt,
           "best_params": best.to_dict() if best is not None else None,
           "validation_season": season.to_dict("records"), "validation_all": allwin.to_dict("records"),
           **{f"{p}_{k}": v.to_dict("records") for p, d in checks.items() for k, v in d.items()}}
    (ROOT / "reports" / "event_experiment.json").write_text(json.dumps(out, indent=1, default=float))
    print("best:", best_name, "adopt:", adopt, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
