"""Market-sentiment overlay from the VIX "fear index" (public, free, daily history).

For rounds on day d the signal uses closes of days strictly before d (the
latest VIX close known before 09:10), so there is no lookahead. The overlay
scales the current target's gross by m(d); execution (lam, band) is unchanged.

Pre-registered variants (fear = signal on):
  backwardation  VIX / VIX3M > 1 (panic: near-term fear above 3-month)
  level25        VIX > 25
  z2             VIX z-score vs the trailing 60 days > 2 (a spike)
  continuous     m = clip(median_250(VIX) / VIX, lo, hi)
each in two directions: de-risk (m = 0.5 on fear) and contrarian (m = 1.4,
i.e. gross 0.5 -> 0.7, on fear; the "buy when others are fearful" effect).
Adoption rule (CLAUDE.md): beat current on validation AND not lose on train.
Data: data/vix_daily.csv, downloaded from Yahoo Finance on first run.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris import sentiment  # noqa: E402
from eyris.backtest import target_policy  # noqa: E402
from eyris.config import Params  # noqa: E402
from eyris.execution import sanitize  # noqa: E402
from eyris.experiment import Study, summary_row  # noqa: E402
from agentic_research import select_score  # noqa: E402
from structural_experiment import evaluate_policy  # noqa: E402

VIX_FILE = ROOT / "data" / "vix_daily.csv"


def load_vix():
    if not VIX_FILE.exists():
        import yfinance as yf
        df = yf.download(["^VIX", "^VIX3M"], start="2020-06-01", interval="1d", auto_adjust=False,
                         progress=False)["Close"]
        df.to_csv(VIX_FILE)
    v = pd.read_csv(VIX_FILE, index_col=0, parse_dates=True)
    v.columns = ["vix", "vix3m"]
    return v.ffill().dropna()


def signals(vix, days):
    """Shared with the live agent (eyris/sentiment.py), plus the experiment-only 250-day median ratio."""
    sig = sentiment.signals(vix, days)
    rel = (vix.vix.rolling(250, min_periods=120).median() / vix.vix).to_frame("vix")
    rel["vix3m"] = 1.0
    sig["rel"] = sentiment.signals(rel, days).vix.to_numpy()   # same "strictly before d" alignment
    return sig


def multiplier(sig, name):
    direction, kind = name.split("_", 1)
    if kind == "continuous":
        rel = sig.rel.fillna(1.0).to_numpy()
        # rel < 1: VIX above its median (fear). De-risk follows rel, contrarian inverts it.
        m = rel if direction == "derisk" else 1.0 / rel
        return np.clip(m, 0.5, 1.4)
    fear = {"backwardation": sig.ratio > 1.0, "level25": sig.vix > 25, "z2": sig.z > 2}[kind]
    fear = fear.fillna(False).to_numpy()
    return np.where(fear, 0.5 if direction == "derisk" else 1.4, 1.0)


NAMES = [f"{d}_{k}" for d in ("derisk", "contrarian") for k in ("backwardation", "level25", "z2", "continuous")]


def overlay_policy(st, params, m_by_day):
    base = st.targets(params)
    day = st.r.day[st.ks]
    scaled = np.array([sanitize(t * m_by_day[d]) for t, d in zip(base, day)])
    return target_policy(scaled, st.k_first, params)


def main():
    t0 = time.time()
    cur = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    vix = load_vix()
    out, sizes = {}, None
    trigger = {}
    for period, stride in (("validation", 1), ("train", 5), ("holdout", 1)):
        st = Study(period, stride=stride)
        sizes = sizes or {"score": len(st.field_for()[0]) + 1, "active": len(st.field_for(variant="active")[0]) + 1,
                          "llm": len(st.field_for(variant="llm")[0]) + 1}
        sig = signals(vix, st.p.days)
        rows = [{"name": "current", **summary_row(st.evaluate_both(cur))}]
        in_period = np.unique(st.r.day[st.ks])
        for name in NAMES:
            m = multiplier(sig, name)
            trigger.setdefault(name, {})[period] = float(np.mean(np.abs(m[in_period] - 1.0) > 1e-9))
            rows.append({"name": name, **summary_row(evaluate_policy(st, overlay_policy(st, cur, m)))})
        df = select_score(pd.DataFrame(rows), sizes)
        out[period] = df
        print(period, f"{time.time() - t0:.0f}s")
        print(df[["name", "selection", "worst_pct", "rank_mean", "rank_llm_mean", "cumulative_return_median",
                  "sharpe_ratio_median", "maximum_drawdown_worst", "turnover_median"]].round(4).to_string(), flush=True)
    v, t, h = (out[p].set_index("name") for p in ("validation", "train", "holdout"))
    table = pd.DataFrame({"val_sel": v.selection, "train_sel": t.selection, "holdout_sel": h.selection,
                          "val_rank_mean": v.rank_mean, "train_rank_mean": t.rank_mean,
                          "holdout_rank_mean": h.rank_mean})
    table["active_days_val"] = pd.Series({n: trigger[n]["validation"] for n in NAMES})
    table["active_days_train"] = pd.Series({n: trigger[n]["train"] for n in NAMES})
    c = table.loc["current"]
    table["passes"] = (table.val_sel < c.val_sel - 1e-9) & (table.train_sel <= c.train_sel + 1e-9)
    table = table.sort_values(["passes", "val_sel"], ascending=[False, True])
    print(table.round(4).to_string())
    res = {p: d.to_dict("records") for p, d in out.items()}
    res["summary"] = table.reset_index().to_dict("records")
    (ROOT / "reports" / "sentiment_experiment.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
