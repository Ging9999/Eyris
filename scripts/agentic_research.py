"""Apply 2024-26 LLM-agent trading research to the Eyris agent.

1. Agentic factor mining (AlphaAgent / R&D-Agent-Quant / FaVOR style): the
   hypotheses in eyris/factors.py were proposed by an LLM and are validated here.
   Train 2021-24 daily rank IC vs the next 5-day return (non-overlapping t-stat).
   Keep a factor only if |t_train| >= 2.5 AND validation (2025-01..09) confirms the
   sign with t >= 1. Report decay = val IC / train IC.
2. Re-select the configuration against three competitor fields (full, active,
   and an LLM-agent field modelled on StockBench / FINSABER / production studies),
   including FINSABER's recommended regime-aware (trend) risk control.
3. Tilt the selected configuration with the surviving factors, if any.
Adopt changes only if they beat the current model on validation and do not lose on
train windows. Writes reports/agentic_research.{json,md}.
"""
import itertools
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
from eyris import factors  # noqa: E402
from eyris.config import Params  # noqa: E402
from eyris.experiment import PERIODS, Study, summary_row  # noqa: E402

FIELDS = ("score", "active", "llm")  # rank columns: rank_score (full), rank_active, rank_llm
HORIZON_DAYS = 5


def select_score(df, sizes):
    pct = []
    worst = []
    for f in FIELDS:
        med = df[f"rank_{f}_median"] if f != "score" else df["rank_score_median"]
        wst = df[f"rank_{f}_worst"] if f != "score" else df["rank_score_worst"]
        pct.append((med - 1) / (sizes[f] - 1))
        worst.append((wst - 1) / (sizes[f] - 1))
    df = df.copy()
    df["pct_full"], df["pct_active"], df["pct_llm"] = pct
    df["selection"] = sum(pct) / len(pct)
    df["worst_pct"] = pd.concat(worst, axis=1).max(axis=1)
    return df.sort_values(["selection", "worst_pct"]).reset_index(drop=True)


def factor_ics(p, r):
    z = factors.zscores(factors.daily_factors(p))           # (D, N, F), uses days <= d
    first_round = np.full(len(p.days), -1)
    ones = np.flatnonzero(r.number == 1)
    first_round[r.day[ones]] = ones
    rows = []
    for d in range(factors.MIN_DAYS, len(p.days) - HORIZON_DAYS - 1):
        k1, k2 = first_round[d + 1], first_round[d + 1 + HORIZON_DAYS]
        if k1 < 0 or k2 < 0:
            continue
        fwd = np.log(r.price[k2] / r.price[k1])               # executed after day d completes
        rows.append([p.days[d]] + [spearmanr(z[d, :, j], fwd).statistic for j in range(len(factors.NAMES))])
    return pd.DataFrame(rows, columns=["date", *factors.NAMES]).set_index("date")


def t_nonoverlap(s):
    s = s.dropna().iloc[::HORIZON_DAYS]
    return s.mean() / s.std(ddof=1) * np.sqrt(len(s)), s.mean()


def main():
    t0 = time.time()
    cur = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    val = Study("validation")
    p, r = val.p, val.r
    sizes = {"score": len(val.field_for()[0]) + 1, "active": len(val.field_for(variant="active")[0]) + 1,
             "llm": len(val.field_for(variant="llm")[0]) + 1}

    # ---- 1. agentic factor mining
    ic = factor_ics(p, r)
    tr = ic[(ic.index >= PERIODS["train"][0]) & (ic.index <= PERIODS["train"][1])]
    va = ic[(ic.index >= PERIODS["validation"][0]) & (ic.index <= PERIODS["validation"][1])]
    mining = []
    for n in factors.NAMES:
        t_tr, m_tr = t_nonoverlap(tr[n])
        t_va, m_va = t_nonoverlap(va[n])
        keep = abs(t_tr) >= 2.5 and np.sign(m_va) == np.sign(m_tr) and abs(t_va) >= 1.0
        mining.append({"factor": n, "hypothesis": factors.HYPOTHESES[n][0], "source": factors.HYPOTHESES[n][1],
                       "ic_train": m_tr, "t_train": t_tr, "ic_val": m_va, "t_val": t_va,
                       "decay": m_va / m_tr if m_tr else np.nan, "keep": bool(keep)})
    mining = pd.DataFrame(mining)
    print(mining[["factor", "ic_train", "t_train", "ic_val", "t_val", "decay", "keep"]].round(4).to_string())
    kept = mining[mining.keep]
    model = factors.FactorModel({row.factor: float(np.sign(row.ic_train)) for row in kept.itertuples()}) \
        if len(kept) else None

    # ---- 2. re-select configuration vs three fields
    grid = []
    for lb, g, lam, band in itertools.product((30, 40), (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0),
                                              (0.1, 0.25, 0.5), (0.05, 0.10)):
        grid.append(replace(cur, lookback_days=lb, gross=g, lam=lam, band=band))
    for td, g in itertools.product((10, 20), (0.5, 0.7, 0.85, 1.0)):
        grid.append(replace(cur, gross_mode="trend", trend_days=td, trend_floor=0.5, gross=g))
    rows = [{**q.to_dict(), **summary_row(val.evaluate_both(q))} for q in grid]
    sel = select_score(pd.DataFrame(rows), sizes)
    keys = list(cur.to_dict())
    best = Params(**{k: sel.iloc[0][k] for k in keys})
    cur_row = sel[(sel[keys] == pd.Series(cur.to_dict())).all(axis=1)]
    cols = ["lookback_days", "gross", "gross_mode", "trend_days", "lam", "band", "selection", "pct_full",
            "pct_active", "pct_llm", "worst_pct", "cumulative_return_median", "maximum_drawdown_worst",
            "turnover_median"]
    print(sel[cols].head(12).round(4).to_string())

    # ---- 3. factor tilt on the best config
    tilts = []
    if model is not None:
        for tilt in (0.1, 0.25, 0.5, 1.0):
            q = replace(best, use_alpha=True, tilt=tilt)
            tilts.append({**q.to_dict(), **summary_row(val.evaluate_both(q, model))})
        tilts = select_score(pd.DataFrame(tilts), sizes)
        print(tilts[["tilt", "selection", "pct_full", "pct_active", "pct_llm", "worst_pct",
                     "cumulative_return_median", "turnover_median"]].round(4).to_string())

    # ---- train + holdout check of current vs best (vs best+tilt)
    cands = {"current": (cur, None), "reselected": (best, None)}
    if len(tilts):
        tb = tilts.iloc[0]
        cands["reselected+factors"] = (replace(best, use_alpha=True, tilt=float(tb["tilt"])), model)
    checks = {}
    for period, stride in (("train", 5), ("holdout", 1)):
        st = Study(period, stride=stride)
        df = select_score(pd.DataFrame([{"name": k, **summary_row(st.evaluate_both(q, m))}
                                        for k, (q, m) in cands.items()]), sizes)
        checks[period] = df
        print(period)
        print(df[["name", "selection", "pct_full", "pct_active", "pct_llm", "worst_pct",
                  "cumulative_return_median", "maximum_drawdown_worst", "turnover_median"]].round(4).to_string())

    vsel = {"current": float(cur_row.iloc[0]["selection"]), "reselected": float(sel.iloc[0]["selection"])}
    if len(tilts):
        vsel["reselected+factors"] = float(tilts.iloc[0]["selection"])
    trs = checks["train"].set_index("name")["selection"]
    winner = "current"
    for name in ("reselected", "reselected+factors"):
        if name in vsel and vsel[name] < vsel[winner] - 1e-9 and trs[name] <= trs[winner] + 1e-9:
            winner = name
    out = {"factor_mining": mining.to_dict("records"), "kept_factors": list(kept.factor),
           "validation_selection": vsel, "best_config": best.to_dict(), "winner": winner,
           "tilt_results": tilts.to_dict("records") if len(tilts) else [],
           "top_configs": sel[cols].head(12).to_dict("records"),
           "train": checks["train"].to_dict("records"), "holdout": checks["holdout"].to_dict("records"),
           "field_sizes": sizes}
    (ROOT / "reports" / "agentic_research.json").write_text(json.dumps(out, indent=1, default=float))
    if model is not None:
        model.save(ROOT / "models" / "factor_model.json")
    print("winner:", winner, vsel, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
