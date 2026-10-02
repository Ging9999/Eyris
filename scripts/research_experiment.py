"""Test research-backed upgrades against the current baseline.

Ideas (each a pre-declared small grid on top of models/params.json):
  ewma      Man AHL-style EWMA risk estimates (half-life in days)
  hrp       Hierarchical Risk Parity (Lopez de Prado 2016)
  voltarget volatility-managed exposure (Moreira & Muir 2017; Man, AQR)
  trend     time-series-momentum exposure overlay (AQR tail-risk research)
  combined  vol target + trend
Selection: same rule as tune.py (mean of median rank percentiles in the full and
active fields, tie-break worst window). Adopt only if it beats the baseline on
validation AND is not worse on 2021-2024 train windows.
Writes reports/research_experiment.json and reports/research_experiment.md.
"""
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

sys.path.insert(0, str(ROOT / "scripts"))
from tune import add_selection_score  # noqa: E402


def variants(base):
    out = [("baseline", base)]
    for hl in (5, 10, 20):
        out.append(("ewma", replace(base, halflife_days=float(hl))))
    for lb in (20, 40):
        for g in (0.5, 0.7):
            out.append(("hrp", replace(base, risk_method="hrp", lookback_days=lb, gross=g)))
    for vt in (0.06, 0.08, 0.10, 0.12, 0.15):
        for g in (0.7, 1.0):
            out.append(("voltarget", replace(base, gross_mode="voltarget", vol_target=vt, gross=g)))
    for td in (5, 10, 20, 40):
        for fl in (0.0, 0.5):
            for g in (0.5, 0.7, 0.85):
                out.append(("trend", replace(base, gross_mode="trend", trend_days=td, trend_floor=fl, gross=g)))
    for vt in (0.08, 0.10, 0.12):
        for td in (10, 20):
            for g in (0.7, 1.0):
                out.append(("combined", replace(base, gross_mode="voltarget_trend", vol_target=vt,
                                                trend_days=td, trend_floor=0.5, gross=g)))
    return out


def evaluate(study, cands):
    rows = []
    for idea, params in cands:
        rows.append({"idea": idea, **params.to_dict(), **summary_row(study.evaluate_both(params))})
    return rows


def main():
    t0 = time.time()
    base = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    cands = variants(base)
    val = Study("validation")
    n_full, n_active = len(val.field_for()[0]) + 1, len(val.field_for(variant="active")[0]) + 1
    v = add_selection_score(pd.DataFrame(evaluate(val, cands)), n_full, n_active)
    v.to_csv(ROOT / "reports" / "research_validation.csv", index=False)
    best_per_idea = v.sort_values(["selection", "worst_pct"]).groupby("idea", sort=False).head(1)
    cols = ["idea", "risk_method", "lookback_days", "gross", "gross_mode", "halflife_days", "vol_target",
            "trend_days", "trend_floor", "selection", "pct_full", "pct_active", "worst_pct",
            "cumulative_return_median", "maximum_drawdown_worst", "turnover_median"]
    print(best_per_idea[cols].to_string())

    # train re-check of each idea's best variant (and the baseline)
    picks = [(r["idea"], Params(**{k: r[k] for k in base.to_dict()})) for _, r in best_per_idea.iterrows()]
    train = Study("train", stride=5)
    tr = add_selection_score(pd.DataFrame(evaluate(train, picks)), n_full, n_active)
    print("train:")
    print(tr[cols].to_string())
    hold = Study("holdout")
    ho = add_selection_score(pd.DataFrame(evaluate(hold, picks)), n_full, n_active)
    print("holdout (already viewed once for the baseline; indicative only):")
    print(ho[cols].to_string())

    vb = v[v.idea == "baseline"].iloc[0]
    tb = tr[tr.idea == "baseline"].iloc[0]
    verdict = {}
    for _, r in best_per_idea.iterrows():
        if r.idea == "baseline":
            continue
        t = tr[tr.idea == r.idea].iloc[0]
        better_val = (r.selection, r.worst_pct) < (vb.selection, vb.worst_pct)
        ok_train = t.selection <= tb.selection
        verdict[r.idea] = "adopt" if (better_val and ok_train) else "reject"
    out = {"baseline": base.to_dict(), "verdict": verdict,
           "validation_best": best_per_idea.to_dict("records"),
           "train": tr.to_dict("records"), "holdout": ho.to_dict("records")}
    (ROOT / "reports" / "research_experiment.json").write_text(json.dumps(out, indent=1, default=float))
    print(verdict, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
