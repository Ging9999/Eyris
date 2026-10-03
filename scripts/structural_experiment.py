"""Structurally different ideas, each with a few variants fixed in advance.

A. Portfolio drawdown brake: track our own NAV from prices visible at each
   deadline (close of bar info_end, no lookahead). If it falls `dd` below its
   running peak in the phase, scale the target by `floor` and go there at once;
   "perm" keeps the brake for the rest of the phase, "recover" lifts it once the
   drawdown is back under dd/2.
B. Equal risk contribution (risk parity with correlations).
C. Sector-balanced weights: equal or inverse-vol budgets for the 6 organizer sectors.
D. Entry ramp: build the first position linearly over n rounds.

Adoption rule (CLAUDE.md): beat current on validation AND not lose on train.
Writes reports/structural_experiment.{json,md tables in json}; does not edit params.
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
from eyris.backtest import evaluate_windows, rank_in_field, target_policy  # noqa: E402
from eyris.config import Params  # noqa: E402
from eyris.execution import HOLD, sanitize  # noqa: E402
from eyris.experiment import Study, summary_row  # noqa: E402
from agentic_research import select_score  # noqa: E402


class Wrapped:
    """Stateful wrapper around the base target policy; resets at each window start."""

    def __init__(self, study, params, kind, **kw):
        self.st, self.params, self.kind, self.kw = study, params, kind, kw
        self.targets = study.targets(params)
        self.base = target_policy(self.targets, study.k_first, params)
        self.last_k = None

    def reset(self, k):
        self.n = 0
        self.nav, self.peak, self.braked = 1.0, 1.0, False
        self.units, self.cash = np.zeros(self.targets.shape[1]), 1.0

    def seen_price(self, k):
        return self.st.p.close[int(self.st.r.info_end[k])]

    def __call__(self, k, w):
        if self.last_k is None or k != self.last_k + 1:
            self.reset(k)
        self.last_k = k
        px = self.seen_price(k)
        if self.n:  # mark our book to the prices visible at this deadline
            self.nav = self.cash + self.units @ px
            self.peak = max(self.peak, self.nav)
        self.n += 1
        out = self.decide(k, w)
        held = w if out is HOLD else np.asarray(out)
        self.units = held * self.nav / px
        self.cash = self.nav * (1.0 - held.sum())
        return out

    def decide(self, k, w):
        tgt = self.targets[k - self.st.k_first]
        if self.kind == "ramp":
            n = self.kw["n"]
            if self.n <= n:
                step = sanitize(tgt * self.n / n)
                return step if np.abs(step - w).sum() > 1e-4 else HOLD
            return self.base(k, w)
        # drawdown brake
        dd = 1.0 - self.nav / self.peak
        if not self.braked and dd >= self.kw["dd"]:
            self.braked = True
            return sanitize(tgt * self.kw["floor"])
        if self.braked:
            if self.kw["mode"] == "recover" and dd < self.kw["dd"] / 2:
                self.braked = False
                return sanitize(tgt)
            return HOLD
        return self.base(k, w)


def evaluate_policy(st, pol):
    df = evaluate_windows(st.r, st.wins, pol, st.field_for())
    recs = df.to_dict("records")
    for v in ("active", "llm"):
        df[f"rank_{v}"] = [rank_in_field(row, f) for row, f in zip(recs, st.field_for(variant=v))]
    return df


def candidates(cur):
    out = {"current": ("params", cur, {})}
    for lb in (20, 30, 40):
        out[f"erc_lb{lb}"] = ("params", replace(cur, risk_method="erc", lookback_days=lb), {})
    for m in ("sector_eq", "sector_invvol"):
        out[m] = ("params", replace(cur, risk_method=m), {})
    for dd in (0.02, 0.03, 0.04):
        for floor in (0.0, 0.5):
            for mode in ("perm", "recover"):
                out[f"brake_dd{dd}_f{floor}_{mode}"] = ("brake", cur, {"dd": dd, "floor": floor, "mode": mode})
    for n in (7, 14, 35):
        out[f"ramp_{n}"] = ("ramp", cur, {"n": n})
    return out


def run_period(st, cands, sizes):
    rows = []
    for name, (kind, q, kw) in cands.items():
        if kind == "params":
            df = st.evaluate_both(q)
        else:
            df = evaluate_policy(st, Wrapped(st, q, kind, **kw))
        rows.append({"name": name, **summary_row(df)})
    return select_score(pd.DataFrame(rows), sizes)


def main():
    t0 = time.time()
    cur = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    cands = candidates(cur)
    out = {}
    sizes = None
    for period, stride in (("validation", 1), ("train", 5), ("holdout", 1)):
        st = Study(period, stride=stride)
        if sizes is None:
            sizes = {"score": len(st.field_for()[0]) + 1, "active": len(st.field_for(variant="active")[0]) + 1,
                     "llm": len(st.field_for(variant="llm")[0]) + 1}
        df = run_period(st, cands, sizes)
        out[period] = df
        print(period, f"{time.time() - t0:.0f}s")
        print(df[["name", "selection", "worst_pct", "rank_mean", "rank_llm_mean", "cumulative_return_median",
                  "sharpe_ratio_median", "maximum_drawdown_worst", "turnover_median"]].round(4).to_string(), flush=True)
    v = out["validation"].set_index("name")
    t = out["train"].set_index("name")
    h = out["holdout"].set_index("name")
    table = pd.DataFrame({"val_sel": v.selection, "train_sel": t.selection, "holdout_sel": h.selection,
                          "val_rank_mean": v.rank_mean, "train_rank_mean": t.rank_mean,
                          "holdout_rank_mean": h.rank_mean})
    c = table.loc["current"]
    table["passes"] = (table.val_sel < c.val_sel - 1e-9) & (table.train_sel <= c.train_sel + 1e-9)
    table = table.sort_values(["passes", "val_sel"], ascending=[False, True])
    print(table.round(4).to_string())
    res = {p: d.to_dict("records") for p, d in out.items()}
    res["summary"] = table.reset_index().to_dict("records")
    (ROOT / "reports" / "structural_experiment.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
