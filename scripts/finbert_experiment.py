"""FinBERT headline sentiment (FNSPID 2021-23): does it add anything to Eyris?

Pre-registered on 2026-10-03, before any result was seen:
- Lookahead: a headline whose (unreliable) FNSPID timestamp is dated X is used only from the first
  trading day after X. The signal for day d is the mean FinBERT net sentiment (P(pos) - P(neg))
  of each stock's headlines dated in [previous trading day, d). Stocks without headlines score 0.
- 1. Predictive check: daily cross-sectional Spearman IC of the signal vs the next-day return
  (round-1 execution of d to round-1 execution of d+1), non-overlapping t-stat; plus an event study
  of negative spikes (signal < -0.5 with >= 2 headlines): next-day return minus the equal-weight basket.
- 2. Six variants on the current config, applied at round 1 of each day:
  trim (reduce-only, like the live news veto): threshold {-0.3, -0.5} x cut {0.5, 1.0}, needs >= 2
  headlines; restored the first day the signal is off.  tilt: w ~ invvol * exp(k * signal), k {0.25, 0.5}.
- Adoption bar: beat current (agentic_research.select_score) on the selection windows AND not lose on
  the confirmation windows. FNSPID ends in 2023, so a pass can't be checked on 2025: at most it
  justifies more shadow logging, never live use before Official.

Revision 2026-10-03, decided from coverage only, BEFORE any signal or backtest result was seen:
FNSPID turned out to cap each stock at about 8,700 headlines, so busy names only reach back to
2022-23 (AAPL 2022-06, MSFT 2022-04, AMZN 2023-03), and BAC, GOOGL, META, JNJ, JPM, LLY are
(nearly) absent. The planned 2021-22 / 2023 split would compare mostly news-free periods. Revised:
period 2022-04-01 .. last news date; selection = windows inside 2022-04-01 .. 2023-03-31;
confirmation = windows inside 2023-04-01 .. last news date. Stocks without headlines keep a
neutral signal (no trim, no tilt). The predictive check uses the same two halves.
Writes reports/finbert_experiment.json; FinBERT scores are cached in data/fnspid_finbert.parquet.
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
sys.path.insert(0, str(ROOT / "scripts"))
from eyris import finbert  # noqa: E402
from eyris.backtest import target_policy  # noqa: E402
from eyris.config import Params, UNIVERSE  # noqa: E402
from eyris.execution import sanitize  # noqa: E402
from eyris.experiment import Study, summary_row  # noqa: E402
from eyris.risk import cap_weights  # noqa: E402
from agentic_research import select_score  # noqa: E402
from structural_experiment import evaluate_policy  # noqa: E402

HEADLINES = ROOT / "data" / "fnspid_headlines.parquet"
SCORES = ROOT / "data" / "fnspid_finbert.parquet"
SPIKE, MIN_N = -0.5, 2
SEL = (pd.Timestamp("2022-04-01"), pd.Timestamp("2023-03-31"))
CONF = (pd.Timestamp("2023-04-01"), pd.Timestamp("2023-12-31"))


def scored_headlines():
    if SCORES.exists():
        return pd.read_parquet(SCORES)
    df = pd.read_parquet(HEADLINES)
    df = df[df.date >= "2020-12-01"].reset_index(drop=True)
    t0 = time.time()
    s = np.empty(len(df))
    step = 4096
    for i in range(0, len(df), step):
        s[i:i + step] = finbert.sentiment(df.title.iloc[i:i + step].tolist())
        print(f"  scored {min(i + step, len(df)):,}/{len(df):,} headlines, {time.time() - t0:.0f}s", flush=True)
    df["sentiment"] = s
    df.to_parquet(SCORES)
    return df


def daily_signal(df, days):
    """(D, N) mean sentiment and counts of headlines first usable on each trading day."""
    days = pd.DatetimeIndex(days).normalize()
    label = df.date.dt.normalize()
    # first trading day strictly after the label date
    pos = days.searchsorted(label, side="right")
    ok = pos < len(days)
    d = pd.DataFrame({"d": pos[ok], "i": df.symbol[ok].map({s: j for j, s in enumerate(UNIVERSE)}),
                      "s": df.sentiment[ok]}).dropna()
    g = d.groupby(["d", "i"]).s.agg(["mean", "count"])
    sig, cnt = np.zeros((len(days), len(UNIVERSE))), np.zeros((len(days), len(UNIVERSE)), dtype=int)
    for (di, ii), row in g.iterrows():
        sig[int(di), int(ii)], cnt[int(di), int(ii)] = row["mean"], row["count"]
    return sig, cnt


def predictive_check(st, sig, cnt, span):
    r, p = st.r, st.p
    ones = np.flatnonzero(r.number == 1)
    first = {int(r.day[k]): k for k in ones}
    rows = []
    for d in sorted(first):
        if d + 1 not in first or not span[0] <= p.days[d] <= span[1]:
            continue
        fwd = r.price[first[d + 1]] / r.price[first[d]] - 1
        has = cnt[d] > 0
        ic = spearmanr(sig[d][has], fwd[has]).statistic if has.sum() >= 5 else np.nan
        spikes = np.flatnonzero((sig[d] < SPIKE) & (cnt[d] >= MIN_N))
        rows.append({"day": p.days[d], "ic": ic, "n_names": int(has.sum()),
                     "spike_excess": [float(fwd[i] - fwd.mean()) for i in spikes]})
    df = pd.DataFrame(rows)
    ic = df.ic.dropna()
    ex = np.array([x for xs in df.spike_excess for x in xs])
    return {"days": len(df), "mean_names_with_news": float(df.n_names.mean()),
            "ic_mean": float(ic.mean()), "ic_t": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))),
            "spike_events": int(len(ex)),
            "spike_excess_mean_bps": float(1e4 * ex.mean()) if len(ex) else None,
            "spike_excess_t": float(ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))) if len(ex) > 2 else None}


def trim_flags(st, sig, cnt, thr, cut):
    r = st.r
    on = (sig < thr) & (cnt >= MIN_N)

    def flags(k):
        if r.number[k] != 1:
            return {}, set()
        d = int(r.day[k])
        trim = {int(i): cut for i in np.flatnonzero(on[d])}
        restore = set(int(i) for i in np.flatnonzero(on[d - 1] & ~on[d])) if d > 0 else set()
        return trim, restore
    return flags


def tilt_targets(st, params, sig, kappa):
    base = st.targets(params)
    out = np.empty_like(base)
    for j, k in enumerate(st.ks):
        t = base[j]
        g = t.sum()
        out[j] = sanitize(cap_weights(t * np.exp(kappa * sig[int(st.r.day[k])]), g, params.stock_cap)) if g > 0 else t
    return out


def main():
    t0 = time.time()
    cur = Params(**json.loads((ROOT / "models" / "params.json").read_text())["params"])
    df = scored_headlines()
    last_news = df.date.max()
    print(f"{len(df):,} scored headlines, {df.date.min():%Y-%m-%d} .. {last_news:%Y-%m-%d}")
    st = Study("train", stride=1)
    sig, cnt = daily_signal(df, st.p.days)
    out = {"headlines": len(df), "last_news": str(last_news),
           "coverage_by_year": df.groupby(df.date.dt.year).size().to_dict(),
           "coverage_by_symbol": df.groupby("symbol").date.agg(["count", "min"]).astype(str).to_dict("index"),
           "predictive": {"selection": predictive_check(st, sig, cnt, SEL),
                          "confirmation": predictive_check(st, sig, cnt, CONF)}}
    print(json.dumps(out["predictive"], indent=1))

    sizes = {"score": len(st.field_for()[0]) + 1, "active": len(st.field_for(variant="active")[0]) + 1,
             "llm": len(st.field_for(variant="llm")[0]) + 1}
    starts = np.array([st.p.days[st.r.day[k0]] for k0, _ in st.wins])
    ends = np.array([st.p.days[st.r.day[k1 - 1]] for _, k1 in st.wins])
    subsets = {"selection": (starts >= SEL[0]) & (ends <= SEL[1]),
               "confirmation": (starts >= CONF[0]) & (ends <= min(CONF[1], last_news))}

    variants = {"current": target_policy(st.targets(cur), st.k_first, cur)}
    for thr in (-0.3, -0.5):
        for cut in (0.5, 1.0):
            variants[f"trim_thr{thr}_cut{cut}"] = target_policy(st.targets(cur), st.k_first, cur,
                                                                trim_flags(st, sig, cnt, thr, cut))
    for kappa in (0.25, 0.5):
        variants[f"tilt_k{kappa}"] = target_policy(tilt_targets(st, cur, sig, kappa), st.k_first, cur)

    tables = {}
    per_window = {name: evaluate_policy(st, pol) for name, pol in variants.items()}
    for sub, mask in subsets.items():
        rows = [{"name": n, **summary_row(w[mask].reset_index(drop=True))} for n, w in per_window.items()]
        tables[sub] = select_score(pd.DataFrame(rows), sizes)
        print(sub, int(mask.sum()), "windows")
        print(tables[sub][["name", "selection", "worst_pct", "rank_mean", "cumulative_return_median",
                           "maximum_drawdown_worst", "turnover_median"]].round(4).to_string(), flush=True)
    a, b = (tables[s].set_index("name").selection for s in ("selection", "confirmation"))
    passes = {n: bool(a[n] < a["current"] - 1e-9 and b[n] <= b["current"] + 1e-9) for n in variants if n != "current"}
    out.update(tables={k: v.to_dict("records") for k, v in tables.items()}, passes=passes,
               windows={k: int(v.sum()) for k, v in subsets.items()})
    (ROOT / "reports" / "finbert_experiment.json").write_text(json.dumps(out, indent=1, default=str))
    print("passes:", passes, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
