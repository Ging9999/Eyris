"""Post-phase review: plumbing checks, simulator reconciliation, decision evidence.

Offline functions used by scripts/validation_review.py. Inputs are files saved by
that script (organizer API responses, Yahoo 1-minute bars, daily closes) plus
the per-round logs in private/<round_id>/.

Two days of Validation are far too few to tune strategy parameters on. What they
can do is (1) prove the plumbing, (2) check that our simulator reproduces the
organizer's accounting, which every backtest conclusion rests on, and
(3) supply the evidence named in DECISION_RULES, written before any result was seen.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import compute_metrics, trade_to
from .config import FEE_RATE, INITIAL_NAV, N_ASSETS, UNIVERSE

ET = "America/New_York"

# Pre-registered on 2026-10-02, before Live Validation. Change them only with a reason
# that does not depend on Validation P&L.
DECISION_RULES = {
    "plumbing": "Must pass before Official: every uploaded round accepted; holdings_source == portfolio_api "
                "after the first round; replay MATCH on every round; no unexplained breaker trip; last_bar is "
                "the latest completed bar before each deadline.",
    "simulator": "Our replica of the organizer accounting must match every period NAV within 2 bps and traded "
                 "notional within 1%. If not, find the convention difference (fees, weights of pre- or "
                 "post-fee NAV, prices) and re-run the key backtests before Official. Do not change the strategy.",
    "fill_proxy": "Informational: the gap between the backtest's hourly-mid fill proxy and the real 1-minute "
                  "open, in bps. Our strategy trades almost only at 09:30, which is modelled exactly.",
    "news_veto": "Keep ON for Official only if all hold: the API answered in >= 90% of rounds; <= 2 trims per "
                 "day on average; every trim's reason names a concrete new event on reading. Otherwise OFF. "
                 "Its P&L over two days is not evidence either way.",
    "vix_mode": "Stays OFF (the backtest edge was not significant and it raised the 2022 drawdown).",
    "gross": "Stays 0.5 (best in train, validation and holdout). Validation P&L does not change it.",
    "anything_else": "No parameter changes based on Validation P&L.",
}


def md_table(df, digits=4):
    """Markdown table without extra dependencies."""
    if df is None or len(df) == 0:
        return "(none)"
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.{digits}g}"
        return "" if v is None else str(v).replace("|", "/")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# --------------------------------------------------------------------------- inputs
def load_round_logs(private_dir, phase):
    """{round_id: decision_log dict} for logged rounds of ``phase``."""
    out = {}
    for d in sorted(Path(private_dir).glob(f"{phase}-*-r*")):
        f = d / "decision_log.json"
        if f.exists():
            log = json.loads(f.read_text())
            log["_uploaded_file"] = (d / "decision.json").exists()
            llm = d / "llm_log.json"
            log["_llm"] = json.loads(llm.read_text()) if llm.exists() else None
            out[d.name] = log
    return out


def phase_rounds(schedule, phase, until=None):
    """Non-cancelled rounds of ``phase`` (dicts with id, exec, close), executed before ``until``."""
    rows = []
    for r in schedule["rounds"]:
        if r.get("phase") != phase or r.get("status") == "CANCELLED":
            continue
        ex = pd.Timestamp(r["execution_time"]).tz_convert(ET)
        if until is not None and ex > pd.Timestamp(until).tz_convert(ET):
            continue
        rows.append({"id": r["id"], "exec": ex, "close": pd.Timestamp(r["close_time"]).tz_convert(ET)})
    return sorted(rows, key=lambda x: x["exec"])


def exec_prices(minute, rounds):
    """(K, N) open of the 1-minute bar starting at each execution time.

    ``minute``: long frame with columns timestamp (tz-aware or naive ET), ticker, open, high, low, close.
    """
    m = minute.copy()
    ts = pd.to_datetime(m["timestamp"])
    m["timestamp"] = ts.dt.tz_convert(ET) if ts.dt.tz is not None else ts.dt.tz_localize(ET)
    wide = m.pivot_table(index="timestamp", columns="ticker", values="open").reindex(columns=list(UNIVERSE))
    return np.array([wide.loc[r["exec"]].to_numpy(dtype=float) if r["exec"] in wide.index
                     else np.full(N_ASSETS, np.nan) for r in rounds])


def hourly_mid_proxy(minute, rounds):
    """The backtest's fill proxy: (open + close) / 2 of the clock-hour bar containing each HH:30 execution."""
    m = minute.copy()
    ts = pd.to_datetime(m["timestamp"])
    m["timestamp"] = ts.dt.tz_convert(ET) if ts.dt.tz is not None else ts.dt.tz_localize(ET)
    o = m.pivot_table(index="timestamp", columns="ticker", values="open").reindex(columns=list(UNIVERSE))
    c = m.pivot_table(index="timestamp", columns="ticker", values="close").reindex(columns=list(UNIVERSE))
    out = []
    for r in rounds:
        ex = r["exec"]
        if ex.hour == 9:                       # 09:30 executes at the open in the backtest too
            out.append(o.loc[ex].to_numpy(dtype=float) if ex in o.index else np.full(N_ASSETS, np.nan))
            continue
        start, end = ex.floor("h"), ex.floor("h") + pd.Timedelta(minutes=59)
        oo, cc = o.loc[start:end], c.loc[start:end]
        out.append(0.5 * (oo.iloc[0].to_numpy(dtype=float) + cc.iloc[-1].to_numpy(dtype=float))
                   if len(oo) else np.full(N_ASSETS, np.nan))
    return np.array(out)


# --------------------------------------------------------------------------- replica of the organizer ledger
def replicate(rounds, weights, prices, day_close, fee=FEE_RATE):
    """Our accounting of the submitted decisions at the given execution prices.

    weights: {round_id: array(N)} for rounds with a selected valid decision (others hold).
    day_close: {date: array(N)} official 16:00 closes.
    Returns periods (dicts like the metrics API) and timestamped valuation points.
    """
    q, cash = np.zeros(N_ASSETS), INITIAL_NAV
    periods, points = [], [{"as_of": rounds[0]["exec"], "nav": INITIAL_NAV}]
    for k, r in enumerate(rounds):
        px = prices[k]
        nav_before = cash + q @ px
        notional = 0.0
        if r["id"] in weights:
            q, cash, notional = trade_to(q, cash, px, np.asarray(weights[r["id"]], dtype=float), fee)
        last = k + 1 == len(rounds)
        if last or rounds[k + 1]["exec"].date() != r["exec"].date():
            close = day_close[r["exec"].date()]
            points.append({"as_of": r["close"], "nav": cash + q @ close})
        if last:
            end_nav, end_t = points[-1]["nav"], r["close"]
        else:
            end_nav, end_t = cash + q @ prices[k + 1], rounds[k + 1]["exec"]
            points.append({"as_of": end_t, "nav": end_nav})
        periods.append({"round_id": r["id"], "start_time": r["exec"], "end_time": end_t,
                        "nav_before": nav_before, "nav_after_period": end_nav, "traded_notional": notional})
    return periods, points


def metrics_of(periods, points):
    """Official metrics (eyris.backtest.compute_metrics) from periods + valuation points."""
    pts = sorted({pd.Timestamp(p["as_of"]): float(p["nav"]) for p in points}.items())
    return compute_metrics(np.array([float(p["nav_before"]) for p in periods]),
                           np.array([float(p["nav_after_period"]) for p in periods]),
                           np.array([float(p["traded_notional"]) for p in periods]),
                           np.array([v for _, v in pts]))


def organizer_periods(metrics_payload):
    """Periods from a metrics API response, keyed by start time."""
    out = {}
    for p in metrics_payload.get("periods", []):
        if "start_time" in p:
            out[pd.Timestamp(p["start_time"]).tz_convert(ET)] = {
                "nav_before": float(p["nav_before"]), "nav_after_period": float(p["nav_after_period"]),
                "traded_notional": float(p["traded_notional"])}
    return out


def reconcile(ours, theirs):
    """Per-period differences: NAV in bps of NAV, notional in % (of theirs)."""
    rows = []
    for p in ours:
        t = theirs.get(pd.Timestamp(p["start_time"]))
        if t is None:
            continue
        rows.append({"round_id": p["round_id"],
                     "nav_before_bps": 1e4 * (p["nav_before"] / t["nav_before"] - 1),
                     "nav_after_bps": 1e4 * (p["nav_after_period"] / t["nav_after_period"] - 1),
                     "notional_ours": p["traded_notional"], "notional_theirs": t["traded_notional"],
                     "notional_pct": (100 * (p["traded_notional"] / t["traded_notional"] - 1)
                                      if t["traded_notional"] > 0 else (0.0 if p["traded_notional"] == 0 else np.inf))})
    df = pd.DataFrame(rows)
    ok = (len(df) > 0 and df[["nav_before_bps", "nav_after_bps"]].abs().max().max() <= 2.0
          and (df["notional_pct"].abs() <= 1.0).all())
    return df, bool(ok)


# --------------------------------------------------------------------------- operations and decision evidence
def operations(logs, rounds, replay=None, receipts=None):
    """One row per scheduled round: what happened and whether the plumbing was right."""
    rep = {r["round_id"]: r["status"] for r in (replay or [])}
    rows = []
    for i, r in enumerate(rounds):
        log = logs.get(r["id"])
        row = {"round_id": r["id"], "logged": log is not None}
        if log:
            row.update(hold=log["hold"], reason=log["reason"], holdings=log["holdings_source"],
                       last_bar=log["last_bar"], breaker=log.get("breaker", False),
                       uploaded=log["_uploaded_file"] and not log["hold"],
                       news=(log.get("news_status") or "")[:40], replay=rep.get(r["id"]))
            row["holdings_ok"] = log["holdings_source"] in ("portfolio_api", "paper") or (
                i == 0 and log["holdings_source"] == "assumed_initial_cash")
        if receipts and r["id"] in receipts:
            row["receipt"] = receipts[r["id"]]
        rows.append(row)
    return pd.DataFrame(rows)


def _llm_flags(llm):
    """Flags from an llm_log.json: the model's JSON reply is stored as text in "response"."""
    try:
        return json.loads((llm or {}).get("response") or "{}").get("flags") or []
    except (ValueError, AttributeError):
        return []


def news_veto_evidence(logs, rounds, prices, last_close):
    """Each news trim: names, reasons, and the counterfactual P&L of the cut to the end of the phase."""
    idx = {r["id"]: k for k, r in enumerate(rounds)}
    rows, answered, asked = [], 0, 0
    for rid, log in logs.items():
        status = log.get("news_status")
        if status is None or str(status).startswith("skipped"):
            continue
        asked += 1
        answered += 0 if (log.get("_llm") or {}).get("error") else 1
        for sym, cut in (log.get("news_trims") or {}).items():
            j = UNIVERSE.index(sym)
            k = idx.get(rid)
            w = (log.get("target") or {}).get(sym, 0.0) * cut
            ret = (last_close[j] / prices[k][j] - 1) if k is not None and np.isfinite(prices[k][j]) else np.nan
            reasons = [f.get("reason") for f in _llm_flags(log.get("_llm")) if f.get("symbol") == sym]
            rows.append({"round_id": rid, "symbol": sym, "cut": cut, "weight_cut": w, "ret_after": ret,
                         "pnl_avoided_bps": -1e4 * w * ret if np.isfinite(ret) else np.nan,
                         "fee_cost_bps": 1e4 * 2 * w * FEE_RATE, "reason": (reasons or [""])[0]})
    days = len({r["exec"].date() for r in rounds}) or 1
    summary = {"rounds_asked": asked, "answer_rate": answered / asked if asked else None,
               "trims": len(rows), "trims_per_day": len(rows) / days}
    return pd.DataFrame(rows), summary
