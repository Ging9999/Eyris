"""Tune the baseline (no alpha) on 2025 validation windows.

Stage 1: broad grid (risk method, lookback, gross, lam, band) vs the full field.
Stage 2: focused grid around the stage-1 winner, scored against two reference
         fields ("full", and "active" = without the buy-and-hold strategies),
         because the real competitor field is unknown.
Selection: lowest mean of the two median rank percentiles (0 = best), tie-break
on the worse of the two worst-case windows. The top configs are re-checked on
2021-2024 train windows (stride 5) and written to reports/.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.config import Params  # noqa: E402
from eyris.experiment import Study, summary_row  # noqa: E402

OUT = ROOT / "reports"
STAGE1 = dict(
    risk_method=["invvol", "minvar", "blend"],
    lookback_days=[10, 20, 40],
    gross=[0.5, 0.7, 0.85, 1.0],
    lam=[0.1, 0.25, 0.5, 1.0],
    band=[0.0, 0.02, 0.05, 0.10],
)
STAGE2 = dict(
    risk_method=["invvol"],
    lookback_days=[30, 40],
    gross=[0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],
    lam=[0.1, 0.25, 0.5],
    band=[0.05, 0.10],
)
KEYS = list(STAGE1)


def grid(spec):
    return [dict(zip(KEYS, v)) for v in itertools.product(*(spec[k] for k in KEYS))]


def run(study, configs, both=False):
    rows = []
    for i, cfg in enumerate(configs):
        params = Params(**cfg)
        df = study.evaluate_both(params) if both else study.evaluate(params)
        rows.append({**cfg, **summary_row(df)})
        if i % 25 == 0:
            print(f"  {i}/{len(configs)}", flush=True)
    return pd.DataFrame(rows)


def add_selection_score(df, n_full, n_active):
    df["pct_full"] = (df["rank_score_median"] - 1) / (n_full - 1)
    df["pct_active"] = (df["rank_active_median"] - 1) / (n_active - 1)
    df["selection"] = 0.5 * (df["pct_full"] + df["pct_active"])
    df["worst_pct"] = pd.concat([(df["rank_score_worst"] - 1) / (n_full - 1),
                                 (df["rank_active_worst"] - 1) / (n_active - 1)], axis=1).max(axis=1)
    return df.sort_values(["selection", "worst_pct"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-stage1", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t = time.time()
    val = Study("validation")
    n_full = len(val.field_for()[0]) + 1
    n_active = len(val.field_for(variant="active")[0]) + 1
    print(f"validation: {len(val.wins)} windows, field {n_full} / active {n_active}")

    s1_path = OUT / "tuning_validation.csv"
    if a.rerun_stage1 or not s1_path.exists():
        s1 = run(val, grid(STAGE1)).sort_values(["rank_score_median", "rank_score_worst", "rank_mean"])
        s1.to_csv(s1_path, index=False)

    s2 = add_selection_score(run(val, grid(STAGE2), both=True), n_full, n_active)
    s2.to_csv(OUT / "tuning_stage2_validation.csv", index=False)
    cols = KEYS + ["selection", "pct_full", "pct_active", "worst_pct", "cumulative_return_median",
                   "maximum_drawdown_worst", "turnover_median"]
    print(s2[cols].head(20).to_string())

    train = Study("train", stride=5)
    top = s2.head(10)[KEYS].to_dict("records")
    tr = add_selection_score(run(train, top, both=True), n_full, n_active)
    tr.to_csv(OUT / "tuning_top_train.csv", index=False)
    print("train re-check:")
    print(tr[cols].to_string())

    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in s2.iloc[0][KEYS].items()}
    best_params = Params(**best).to_dict()
    (OUT / "best_baseline.json").write_text(json.dumps(
        {"params": best_params, "validation": s2.iloc[0].to_dict(),
         "rule": "min mean(median rank percentile in full & active fields), tie-break worst window"},
        indent=1, default=float))
    print("selected:", best_params, f"({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
