"""Backtest at the exact decision rounds with fees and the official metrics.

Each simulation starts from USD 1,000,000 cash, like the live phases. Period k
runs from round k's execution to round k+1's execution (round 7 includes the
overnight hold and the 16:00 close as an extra drawdown observation). The last
period ends at the final day's close. Metric formulas mirror the starter kit's
``kit/evaluation.py``.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import alpha
from .config import ANNUALIZATION, FEE_RATE, INITIAL_NAV, MAX_WEIGHT, N_ASSETS, WINDOW_DAYS
from .data import Panels, Rounds
from .execution import HOLD, apply_events, rebalance, sanitize
from .risk import cap_weights

METRICS = ("cumulative_return", "sharpe_ratio", "maximum_drawdown", "turnover")
HIGHER_IS_BETTER = {"cumulative_return": True, "sharpe_ratio": True,
                    "maximum_drawdown": False, "turnover": False}
EQUAL_WEIGHTS = {m: 0.25 for m in METRICS}


# --------------------------------------------------------------------------- simulation
def trade_to(q, cash, price, w, fee=FEE_RATE):
    """Rebalance holdings to target weights of post-fee NAV. Returns q, cash, notional."""
    nav = cash + q @ price
    hold_val = q * price
    net = nav
    for _ in range(4):  # fixed point: fee depends on post-fee NAV
        fee_amt = fee * np.abs(w * net - hold_val).sum()
        net = nav - fee_amt
    new_val = w * net
    q_new = new_val / price
    notional = np.abs(new_val - hold_val).sum()
    cash_new = nav - fee * notional - new_val.sum()
    return q_new, cash_new, notional


@dataclass
class SimResult:
    nav_before: np.ndarray
    nav_end: np.ndarray
    notional: np.ndarray
    valuations: np.ndarray
    traded: np.ndarray
    weights: np.ndarray    # post-trade weights per round (K, N)

    def metrics(self):
        return compute_metrics(self.nav_before, self.nav_end, self.notional, self.valuations)


def simulate(r: Rounds, k0, k1, policy, fee=FEE_RATE):
    """Run ``policy(k, w_current) -> weights | HOLD`` over rounds [k0, k1)."""
    n = k1 - k0
    q = np.zeros(N_ASSETS)
    cash = INITIAL_NAV
    nav_before, nav_end, notional = np.empty(n), np.empty(n), np.zeros(n)
    traded = np.zeros(n, dtype=bool)
    weights = np.zeros((n, N_ASSETS))
    vals = [INITIAL_NAV]
    for j, k in enumerate(range(k0, k1)):
        price = r.price[k]
        nav = cash + q @ price
        nav_before[j] = nav
        w_new = policy(k, q * price / nav)
        if w_new is not HOLD:
            q, cash, notional[j] = trade_to(q, cash, price, np.asarray(w_new), fee)
            traded[j] = True
        weights[j] = q * price / (cash + q @ price)
        last = k + 1 >= k1
        if last or r.day[k + 1] != r.day[k]:
            vals.append(cash + q @ r.day_close[r.day[k]])  # 16:00 close valuation
        if last:
            nav_end[j] = vals[-1]
        else:
            nav_end[j] = cash + q @ r.price[k + 1]
            vals.append(nav_end[j])
    return SimResult(nav_before, nav_end, notional, np.array(vals), traded, weights)


def compute_metrics(nav_before, nav_end, notional, valuations):
    ret = nav_end / nav_before - 1.0
    k = len(ret)
    sd = ret.std(ddof=1) if k > 1 else 0.0
    sharpe = ANNUALIZATION * ret.mean() / sd if sd > 0 else 0.0
    peak = np.maximum.accumulate(valuations)
    mdd = float(((peak - valuations) / peak).max())
    return {"cumulative_return": float(nav_end[-1] / INITIAL_NAV - 1.0),
            "sharpe_ratio": float(sharpe),
            "maximum_drawdown": mdd,
            "turnover": float((notional / nav_before).mean())}


# --------------------------------------------------------------------------- targets
def agent_targets(agent, p: Panels, r: Rounds, ks):
    """Targets the agent would compute at each round in ``ks`` (bars <= info_end)."""
    ks = np.asarray(ks)
    out = np.empty((len(ks), N_ASSETS))
    for j, k in enumerate(ks):
        out[j] = agent.risk_target(p.head(int(r.info_end[k])))
    if agent.alpha_on:
        if hasattr(agent.model, "batch_features"):
            X = agent.model.batch_features(p, r.info_end[ks])
        else:
            raw = alpha.raw_features(p.close, p.volume, p.high, p.low)
            X = alpha.cs_features(raw[r.info_end[ks]])
        scores = agent.model.predict(X)
        for j in range(len(ks)):
            out[j] = alpha.tilt(out[j], scores[j], agent.params.tilt, agent.params.stock_cap)
    return np.array([sanitize(t) for t in out])


def target_policy(targets, k_first, params, event_flags=None):
    """Partial rebalancing toward precomputed targets (targets[k - k_first]).

    event_flags(k) -> (trim, restore) applies the same overlay as Agent.decide.
    """
    def policy(k, w):
        tgt = targets[k - k_first]
        out = rebalance(w, tgt, params.lam, params.band, params.min_trade)
        if event_flags is not None:
            trim, restore = event_flags(k)
            out = apply_events(w, out, tgt, trim, restore, params.min_trade)
        return out
    return policy


# --------------------------------------------------------------------------- baselines
def baseline_policies(p: Panels, r: Rounds, k_first, invvol_targets, n_random=10):
    """Reference field of simple strategies, standing in for other teams."""
    ew = np.full(N_ASSETS, 1.0 / N_ASSETS)
    rng = np.random.default_rng(0)
    randoms = [cap_weights(rng.dirichlet(np.ones(N_ASSETS)), 1.0, MAX_WEIGHT) for _ in range(n_random)]

    def buy_hold(w0):
        return lambda k, w: w0 if w.sum() < 1e-9 else HOLD

    def kit_momentum(k, w):
        e = r.info_end[k]
        c = p.close[e - 5:e + 1]
        mom = c[-1] / c[0] - 1
        out = np.zeros(N_ASSETS)
        order = [i for i in np.argsort(-mom, kind="stable") if mom[i] > 0][:5]
        out[order] = 0.2
        return out

    def mom20_daily(k, w):
        if r.number[k] != 1 and w.sum() > 1e-9:
            return HOLD
        e = r.info_end[k]
        mom = p.close[e] / p.close[e - 140] - 1
        out = np.zeros(N_ASSETS)
        out[np.argsort(-mom, kind="stable")[:10]] = 0.1
        return out

    field = {
        "cash": lambda k, w: HOLD,
        "ew_buy_hold": buy_hold(ew),
        "ew_rebal_every_round": lambda k, w: ew,
        "ew_rebal_daily": lambda k, w: ew if (r.number[k] == 1 or w.sum() < 1e-9) else HOLD,
        "ew_half_cash_daily": lambda k, w: ew * 0.5 if (r.number[k] == 1 or w.sum() < 1e-9) else HOLD,
        "invvol_full_every_round": lambda k, w: invvol_targets[k - k_first],
        "kit_momentum_top5": kit_momentum,
        "mom20_top10_daily": mom20_daily,
    }
    for i, w0 in enumerate(randoms):
        field[f"random_buy_hold_{i}"] = buy_hold(w0)
    # LLM-agent-style teams, modelled on behaviour documented in 2024-26 studies:
    # daily discretionary picks (StockBench), volatility-blind concentrated sizing
    # and overtrading (production fleet study, arXiv 2609.05663), and exposure that
    # is too cautious after rallies / too aggressive after losses (FINSABER).
    def daily_only(fn):
        return lambda k, w: fn(k) if (r.number[k] == 1 or w.sum() < 1e-9) else HOLD

    def top(scores, n, wt):
        out = np.zeros(N_ASSETS)
        out[np.argsort(-scores, kind="stable")[:n]] = wt
        return out

    def ret(k, bars):
        e = r.info_end[k]
        return p.close[e] / p.close[max(0, e - bars)] - 1

    field["llm_news_chaser"] = daily_only(lambda k: top(ret(k, 7), 5, 0.2))
    field["llm_vol_blind_top3"] = daily_only(lambda k: top(ret(k, 35), 3, 0.3))
    field["llm_overtrader_hourly"] = lambda k, w: top(ret(k, 1), 5, 0.2)

    def finsaber_exposure(k):
        g = float(np.clip(0.6 - 4.0 * ret(k, 140).mean(), 0.2, 1.0))
        return top(ret(k, 140), 10, g / 10)
    field["llm_regime_contrarian"] = daily_only(finsaber_exposure)

    # Noisy active "teams": fresh random portfolio every day (e.g. an LLM picker).
    n_days = len(p.days)
    for i in range(n_random // 2):
        daily = np.random.default_rng(100 + i).dirichlet(np.full(N_ASSETS, 0.5), size=n_days)
        daily = np.array([cap_weights(x, 1.0, MAX_WEIGHT) for x in daily])
        field[f"random_daily_{i}"] = (lambda d: lambda k, w: d[r.day[k]]
                                      if (r.number[k] == 1 or w.sum() < 1e-9) else HOLD)(daily)
    return field


# --------------------------------------------------------------------------- windows & ranking
def windows(r: Rounds, day_idx, length=WINDOW_DAYS, stride=1):
    """(k0, k1) round ranges for rolling ``length``-day windows inside ``day_idx``."""
    day_idx = np.asarray(day_idx)
    out = []
    for s in range(0, len(day_idx) - length + 1, stride):
        d0, d1 = day_idx[s], day_idx[s + length - 1]
        if r.day_last_round[d1] < 0:
            continue
        k0 = int(np.searchsorted(r.day, d0))
        out.append((k0, int(r.day_last_round[d1]) + 1))
    return out


def rank_in_field(cand, field, weights=EQUAL_WEIGHTS):
    """Overall rank score of ``cand`` (dict of metrics) among ``field`` + cand.

    Official rule: rank per metric (ties share the average rank), then average.
    ``weights`` lets the metric mix be stressed; competition = equal weights.
    """
    score = 0.0
    for m in METRICS:
        f = np.asarray([x[m] for x in field])
        c = cand[m]
        better = (f > c + 1e-12) if HIGHER_IS_BETTER[m] else (f < c - 1e-12)
        ties = np.abs(f - c) <= 1e-12
        score += weights[m] * (1 + better.sum() + 0.5 * ties.sum())
    return score / sum(weights.values())


def field_ranks(table: pd.DataFrame, weights=EQUAL_WEIGHTS):
    """Overall rank score for every row of a metrics table (one window)."""
    ranks = sum(weights[m] * table[m].rank(ascending=not HIGHER_IS_BETTER[m], method="average")
                for m in METRICS)
    return ranks / sum(weights.values())


def evaluate_windows(r, wins, policy, field_metrics=None, weights=EQUAL_WEIGHTS):
    """Metrics (and field rank, if field given) for each window."""
    rows = []
    for i, (k0, k1) in enumerate(wins):
        m = simulate(r, k0, k1, policy).metrics()
        if field_metrics is not None:
            m["rank_score"] = rank_in_field(m, field_metrics[i], weights)
            m["field_size"] = len(field_metrics[i]) + 1
        m["start_day"] = int(r.day[k0])
        rows.append(m)
    return pd.DataFrame(rows)


def summarize(df):
    """Median / mean / worst-case summary of per-window results."""
    out = {}
    for m in METRICS + (("rank_score",) if "rank_score" in df else ()):
        worst = df[m].min() if HIGHER_IS_BETTER.get(m, False) else df[m].max()
        out[m] = {"median": df[m].median(), "mean": df[m].mean(), "worst": worst}
    return pd.DataFrame(out).T
