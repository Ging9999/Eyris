"""Grid-search the baseline (no alpha) on 2025 validation windows.

Selection rule: lowest median 15-day overall rank score in the reference field,
tie-break on worst-case window. Top configs are re-checked on train windows.
Writes reports/tuning_validation.csv and reports/tuning_top_train.csv.
"""
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eyris.config import Params  # noqa: E402
from eyris.experiment import Study, summary_row  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports"
GRID = dict(
    risk_method=["invvol", "minvar", "blend"],
    lookback_days=[10, 20, 40],
    gross=[0.5, 0.7, 0.85, 1.0],
    lam=[0.1, 0.25, 0.5, 1.0],
    band=[0.0, 0.02, 0.05, 0.10],
)
PARAM_KEYS = list(GRID)


def run(study, configs):
    rows = []
    for i, cfg in enumerate(configs):
        params = Params(**cfg)
        rows.append({**cfg, **summary_row(study.evaluate(params))})
        if i % 50 == 0:
            print(f"  {i}/{len(configs)}", flush=True)
    return pd.DataFrame(rows)


def rank_configs(df):
    return df.sort_values(["rank_score_median", "rank_score_worst", "rank_mean"]).reset_index(drop=True)


def main():
    OUT.mkdir(exist_ok=True)
    t = time.time()
    val = Study("validation")
    print(f"validation: {len(val.wins)} windows, field {len(val.field)} ({time.time() - t:.0f}s)")
    configs = [dict(zip(PARAM_KEYS, v)) for v in itertools.product(*GRID.values())]
    res = rank_configs(run(val, configs))
    res.to_csv(OUT / "tuning_validation.csv", index=False)
    cols = PARAM_KEYS + ["rank_score_median", "rank_score_worst", "cumulative_return_median",
                         "maximum_drawdown_worst", "turnover_median"]
    print(res[cols].head(20).to_string())

    train = Study("train", stride=5)
    print(f"train: {len(train.wins)} windows")
    top = res.head(15)[PARAM_KEYS].to_dict("records")
    tr = run(train, top)
    tr.to_csv(OUT / "tuning_top_train.csv", index=False)
    print(tr[cols].to_string())
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
