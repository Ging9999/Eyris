"""Alpha tilt experiment: does a LightGBM ranker beat plain risk weights after costs?

1. Fit on train rounds (2021-2024, label horizon purged at the boundary).
2. Out-of-sample rank IC on 2025 validation rounds.
3. Validation 15-day windows for tilt strengths on top of the chosen baseline
   (same lam/band, plus a wider band since the tilt adds turnover).
Keep rule: keep only if the best tilt beats the tilt=0 baseline on median AND
mean rank score in validation windows.
Writes reports/alpha_experiment.json and models/alpha_lgbm_h{h}.txt.
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris import alpha  # noqa: E402
from eyris.config import Params, ROUNDS_PER_DAY  # noqa: E402
from eyris.experiment import PERIODS, Study, summary_row  # noqa: E402

HORIZONS = (ROUNDS_PER_DAY, 5 * ROUNDS_PER_DAY)
TILTS = (0.0, 0.1, 0.25, 0.5, 1.0)


def main():
    t0 = time.time()
    best = json.loads((ROOT / "reports" / "best_baseline.json").read_text())["params"]
    base = Params(**best)
    val = Study("validation")
    p, r = val.p, val.r
    raw = alpha.raw_features(p.close, p.volume, p.high, p.low)
    X_all = alpha.cs_features(raw[r.info_end])
    day_dates = p.days[r.day]
    train_end = pd.Timestamp(PERIODS["train"][1])
    out = {"baseline_params": best, "horizons": {}}
    for h in HORIZONS:
        y = alpha.labels(r.price, h)
        ok = ~np.isnan(y).any(axis=1)
        tr = np.flatnonzero((r.info_end >= alpha.MIN_BARS) & (day_dates <= train_end) & ok)
        tr = tr[:-h]  # purge labels that overlap the validation period
        model = alpha.AlphaModel.fit(X_all[tr], y[tr])
        model.save(ROOT / "models" / f"alpha_lgbm_h{h}.txt")
        vk = val.ks[ok[val.ks]]
        scores = model.predict(X_all[vk])
        ics = np.array([spearmanr(s, t).statistic for s, t in zip(scores, y[vk])])
        by_round = {int(n): float(np.nanmean(ics[r.number[vk] == n])) for n in range(1, 8)}
        imp = dict(zip(alpha.FEATURES, map(int, model.booster.feature_importance("gain"))))
        res = {"train_rows": int(len(tr) * 30), "val_ic_mean": float(np.nanmean(ics)),
               "val_ic_tstat_daily": float(_daily_t(ics, r.day[vk])), "ic_by_round": by_round,
               "feature_gain": imp, "windows": []}
        print(f"h={h}: IC mean {res['val_ic_mean']:.4f}  t(daily) {res['val_ic_tstat_daily']:.2f}")
        for band in sorted({base.band, 0.05, 0.10}):
            for tilt in TILTS:
                params = replace(base, use_alpha=tilt > 0, tilt=tilt, band=band)
                row = {"tilt": tilt, "band": band, **summary_row(val.evaluate(params, model if tilt else None))}
                res["windows"].append(row)
                print(f"  band {band:.2f} tilt {tilt:.2f}: rank med {row['rank_score_median']:.2f} "
                      f"mean {row['rank_mean']:.2f} worst {row['rank_score_worst']:.2f} "
                      f"ret {row['cumulative_return_median']:.4f} turn {row['turnover_median']:.4f}")
        out["horizons"][h] = res
    base_row = summary_row(val.evaluate(base))
    out["baseline_validation"] = base_row
    cands = [(h, w) for h, res in out["horizons"].items() for w in res["windows"] if w["tilt"] > 0]
    h_best, w_best = min(cands, key=lambda c: (c[1]["rank_score_median"], c[1]["rank_mean"]))
    keep = (w_best["rank_score_median"] < base_row["rank_score_median"]
            and w_best["rank_mean"] < base_row["rank_mean"])
    out["best_alpha"] = {"horizon": h_best, **w_best}
    out["recommendation"] = "keep" if keep else "drop"
    (ROOT / "reports" / "alpha_experiment.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"baseline rank med {base_row['rank_score_median']:.2f} mean {base_row['rank_mean']:.2f}")
    print(f"best alpha h={h_best} tilt={w_best['tilt']} band={w_best['band']}: "
          f"med {w_best['rank_score_median']:.2f} mean {w_best['rank_mean']:.2f} -> {out['recommendation']}")
    print(f"done in {time.time() - t0:.0f}s")


def _daily_t(ics, days):
    s = pd.Series(ics).groupby(np.asarray(days)).mean().dropna()
    return s.mean() / s.std(ddof=1) * np.sqrt(len(s))


if __name__ == "__main__":
    main()
