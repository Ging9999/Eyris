"""Overnight search over dimensions not covered by tune.py / agentic_research.py.

New knobs: vol_power (strength of the low-vol tilt; 2 = inverse variance), top_k
(hold only the k lowest-vol names), stock_cap, and wide no-trade bands, crossed
with lookback, gross and lam.

1. Grid on validation (2025-01..09), scored by agentic_research.select_score.
2. The top SHORTLIST configs + current are checked on train (2021-24, stride 5) and
   the holdout (2025-Q4, indicative only).
3. Adoption rule (CLAUDE.md): beat current on validation AND not lose on train.
   Only gross == current gross may be adopted (gross 0.5 vs 0.7 is the user's call
   after Live Validation); other gross levels are reported.
Writes reports/overnight_research.{json,md}; does not edit models/params.json.
"""
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.config import Params  # noqa: E402
from eyris.experiment import Study, summary_row  # noqa: E402
from agentic_research import select_score  # noqa: E402

SHORTLIST = 40
COLS = ["lookback_days", "vol_power", "top_k", "stock_cap", "gross", "lam", "band", "selection", "pct_full",
        "pct_active", "pct_llm", "worst_pct", "cumulative_return_median", "sharpe_ratio_median",
        "maximum_drawdown_worst", "turnover_median"]


def grid(cur):
    out = []
    for lb, pw, k, cap, g, lam, band in itertools.product(
            (20, 30, 40), (0.5, 1.0, 1.5, 2.0, 3.0), (0, 10, 15, 20), (0.06, 0.08, 0.10, 0.15),
            (0.3, 0.4, 0.5, 0.6, 0.7, 0.85), (0.25, 0.5), (0.05, 0.10, 0.20)):
        try:
            out.append(replace(cur, lookback_days=lb, vol_power=pw, top_k=k, stock_cap=cap, gross=g,
                               lam=lam, band=band))
        except ValueError:  # gross unreachable under top_k * stock_cap, etc.
            pass
    return out


def run(study, params, sizes):
    rows = []
    t0 = time.time()
    for i, q in enumerate(params):
        rows.append({**q.to_dict(), **summary_row(study.evaluate_both(q))})
        if i % 500 == 0:
            print(f"  {study.period}: {i}/{len(params)} {time.time() - t0:.0f}s", flush=True)
    return select_score(pd.DataFrame(rows), sizes)


def main():
    t0 = time.time()
    cur = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    keys = list(cur.to_dict())
    val = Study("validation")
    sizes = {"score": len(val.field_for()[0]) + 1, "active": len(val.field_for(variant="active")[0]) + 1,
             "llm": len(val.field_for(variant="llm")[0]) + 1}

    cands = grid(cur)
    if cur not in cands:
        cands.append(cur)
    print(f"validation grid: {len(cands)} configs", flush=True)
    vsel = run(val, cands, sizes)
    vsel.to_csv(ROOT / "reports" / "overnight_validation.csv", index=False)

    is_cur = (vsel[keys] == pd.Series(cur.to_dict())).all(axis=1)
    short = pd.concat([vsel[~is_cur].head(SHORTLIST),
                       vsel[~is_cur & (vsel.gross == cur.gross)].head(SHORTLIST),
                       vsel[is_cur]]).drop_duplicates(subset=keys)
    short_params = [Params(**{k: row[k] for k in keys}) for _, row in short.iterrows()]

    checks = {}
    for period, stride in (("train", 5), ("holdout", 1)):
        st = Study(period, stride=stride)
        checks[period] = run(st, short_params, sizes)

    def sel_of(df):
        return df.set_index(keys)["selection"]

    merged = short.set_index(keys)[["selection", "worst_pct"]].rename(
        columns={"selection": "val_sel", "worst_pct": "val_worst"})
    merged["train_sel"] = sel_of(checks["train"])
    merged["train_worst"] = checks["train"].set_index(keys)["worst_pct"]
    merged["holdout_sel"] = sel_of(checks["holdout"])
    merged = merged.reset_index()
    c = merged[(merged[keys] == pd.Series(cur.to_dict())).all(axis=1)].iloc[0]
    merged["passes"] = (merged.val_sel < c.val_sel - 1e-9) & (merged.train_sel <= c.train_sel + 1e-9)
    merged["adoptable"] = merged.passes & (merged.gross == cur.gross)
    merged["combined"] = (merged.val_sel + merged.train_sel) / 2
    merged = merged.sort_values(["combined", "val_sel"]).reset_index(drop=True)
    merged.to_csv(ROOT / "reports" / "overnight_shortlist.csv", index=False)

    adopt = merged[merged.adoptable]
    winner = adopt.iloc[0].to_dict() if len(adopt) else None
    show = ["lookback_days", "vol_power", "top_k", "stock_cap", "gross", "lam", "band", "val_sel", "train_sel",
            "holdout_sel", "val_worst", "train_worst", "passes", "adoptable"]
    print(merged[show].head(25).round(4).to_string())
    print("current:", c[["val_sel", "train_sel", "holdout_sel"]].to_dict())
    print("winner:", winner)
    out = {"current": c.to_dict(), "winner": winner, "n_grid": len(cands), "field_sizes": sizes,
           "top_validation": vsel[COLS].head(25).to_dict("records"),
           "shortlist": merged.to_dict("records"), "elapsed_s": time.time() - t0}
    (ROOT / "reports" / "overnight_research.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
