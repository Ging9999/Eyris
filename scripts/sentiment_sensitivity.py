"""Robustness of the contrarian VIX overlay (scripts/sentiment_experiment.py).

1. Neighbourhood: thresholds (VIX level 20-30, z 1.5-3) x boost (1.2, 1.4, 1.6).
   A real effect should improve on current across the neighbourhood, not at one point.
2. Paired per-window test vs current for the pre-registered winners: mean rank
   difference (full / active / llm field) over non-overlapping 15-day windows, with t-stat.
Writes reports/sentiment_sensitivity.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.config import Params, WINDOW_DAYS  # noqa: E402
from eyris.experiment import Study, summary_row  # noqa: E402
from agentic_research import select_score  # noqa: E402
from sentiment_experiment import load_vix, overlay_policy, signals  # noqa: E402
from structural_experiment import evaluate_policy  # noqa: E402

LEVELS = (20, 22.5, 25, 27.5, 30)
ZS = (1.5, 2.0, 2.5, 3.0)
BOOSTS = (1.2, 1.4, 1.6)


def mult(sig, kind, thr, boost):
    fear = (sig.vix > thr) if kind == "level" else (sig.z > thr)
    return np.where(fear.fillna(False).to_numpy(), boost, 1.0)


def paired(cur_df, new_df, step):
    out = {}
    for col in ("rank_score", "rank_active", "rank_llm"):
        d = (new_df[col] - cur_df[col]).to_numpy()[::step]
        sd = d.std(ddof=1)
        out[col] = {"mean_diff": float(d.mean()), "t": float(d.mean() / sd * np.sqrt(len(d))) if sd > 0 else 0.0,
                    "n": int(len(d)), "better": int((d < -1e-9).sum()), "worse": int((d > 1e-9).sum())}
    return out


def main():
    t0 = time.time()
    cur = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    vix = load_vix()
    res, sizes = {"grid": {}, "paired": {}}, None
    for period, stride in (("validation", 1), ("train", 5), ("holdout", 1)):
        st = Study(period, stride=stride)
        sizes = sizes or {"score": len(st.field_for()[0]) + 1, "active": len(st.field_for(variant="active")[0]) + 1,
                          "llm": len(st.field_for(variant="llm")[0]) + 1}
        sig = signals(vix, st.p.days)
        cur_df = st.evaluate_both(cur)
        rows = [{"kind": "current", "thr": 0, "boost": 1.0, **summary_row(cur_df)}]
        dfs = {}
        for kind, thrs in (("level", LEVELS), ("z", ZS)):
            for thr in thrs:
                for boost in BOOSTS:
                    df = evaluate_policy(st, overlay_policy(st, cur, mult(sig, kind, thr, boost)))
                    dfs[(kind, thr, boost)] = df
                    rows.append({"kind": kind, "thr": thr, "boost": boost, **summary_row(df)})
        tab = select_score(pd.DataFrame(rows), sizes)
        res["grid"][period] = tab.to_dict("records")
        step = max(1, WINDOW_DAYS // stride)
        res["paired"][period] = {f"{k}_{t}_{b}": paired(cur_df, dfs[(k, t, b)], step)
                                 for (k, t, b) in (("level", 25, 1.4), ("z", 2.0, 1.4))}
        c = tab[tab.kind == "current"].iloc[0]
        tab["d_sel"] = tab.selection - c.selection
        tab["d_rank_mean"] = tab.rank_mean - c.rank_mean
        print(period, f"{time.time() - t0:.0f}s")
        print(tab.pivot_table(index=["kind", "thr"], columns="boost", values="d_rank_mean").round(3).to_string())
        print(tab.pivot_table(index=["kind", "thr"], columns="boost", values="d_sel").round(4).to_string())
        print(json.dumps(res["paired"][period], indent=1), flush=True)
    (ROOT / "reports" / "sentiment_sensitivity.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
