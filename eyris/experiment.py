"""Shared experiment plumbing: periods, field caching and config evaluation."""
from dataclasses import replace

import numpy as np
import pandas as pd

from . import alpha, data
from .agent import Agent
from .backtest import (agent_targets, baseline_policies, evaluate_windows, rank_in_field, simulate,
                       summarize, target_policy, windows)
from .config import Params

# Train 2021-2024, validate 2025-01..09 (tuning), hold out 2025-Q4 (touch once).
PERIODS = {
    "train": ("2021-01-01", "2024-12-31"),
    "validation": ("2025-01-01", "2025-09-30"),
    "holdout": ("2025-10-01", "2025-12-31"),
}
# Field of LLM-agent-style competitors (see backtest.baseline_policies) plus a
# team that misses rounds (cash) and simple diversifiers.
LLM_FIELD = {"kit_momentum_top5", "mom20_top10_daily", "ew_rebal_daily", "cash", "llm_news_chaser",
             "llm_vol_blind_top3", "llm_overtrader_hourly", "llm_regime_contrarian",
             "random_daily_0", "random_daily_1", "random_daily_2", "random_daily_3", "random_daily_4"}
FIELD_REFERENCE = Params(risk_method="invvol", lookback_days=20, stock_cap=0.30, gross=1.0)


class Study:
    """Caches data, round ranges, windows and reference-field metrics per period."""

    def __init__(self, period, fill="mid", stride=1):
        self.p, self.r = data.load(fill=fill)
        self.period = period
        days = data.day_range(self.p, *PERIODS[period])
        # warm-up: every lookback (and alpha feature) needs MIN_BARS of history
        days = days[days > self.p.day[alpha.MIN_BARS]]
        self.ks = np.arange(np.searchsorted(self.r.day, days[0]), self.r.day_last_round[days[-1]] + 1)
        self.k_first = int(self.ks[0])
        self.wins = windows(self.r, days, stride=stride)
        ref = agent_targets(Agent(FIELD_REFERENCE), self.p, self.r, self.ks)
        self.field = baseline_policies(self.p, self.r, self.k_first, ref)
        self.field_window_metrics = [
            {name: simulate(self.r, k0, k1, pol).metrics() for name, pol in self.field.items()}
            for k0, k1 in self.wins]
        self._targets = {}

    def field_for(self, exclude=None, variant="full"):
        """Reference field per window. "active" drops the passive buy-and-hold
        strategies, approximating a field of teams that trade every day."""
        def keep(n):
            if n == exclude:
                return False
            if variant == "active":
                return "buy_hold" not in n
            if variant == "llm":
                return n in LLM_FIELD
            return True
        return [[m for n, m in w.items() if keep(n)] for w in self.field_window_metrics]

    def targets(self, params, model=None):
        # every field except the execution ones changes the targets
        key = (replace(params, lam=1.0, band=0.0, min_trade=0.0), id(model))
        if key not in self._targets:
            self._targets[key] = agent_targets(Agent(params, model), self.p, self.r, self.ks)
        return self._targets[key]

    def evaluate(self, params, model=None, variant="full"):
        pol = target_policy(self.targets(params, model), self.k_first, params)
        return evaluate_windows(self.r, self.wins, pol, self.field_for(variant=variant))

    def evaluate_both(self, params, model=None):
        """Simulate once, rank against all field variants."""
        df = self.evaluate(params, model)
        recs = df.to_dict("records")
        for v in ("active", "llm"):
            df[f"rank_{v}"] = [rank_in_field(row, f) for row, f in zip(recs, self.field_for(variant=v))]
        return df

    def evaluate_baseline(self, name, variant="full"):
        return evaluate_windows(self.r, self.wins, self.field[name], self.field_for(name, variant))

    def evaluate_baseline_both(self, name):
        df = self.evaluate_baseline(name)
        recs = df.to_dict("records")
        for v in ("active", "llm"):
            df[f"rank_{v}"] = [rank_in_field(row, f) for row, f in zip(recs, self.field_for(name, v))]
        return df

    def continuous(self, policy):
        """One portfolio across the whole period (for equity curves)."""
        return simulate(self.r, self.k_first, int(self.ks[-1]) + 1, policy)


def summary_row(df):
    s = summarize(df)
    row = {f"{m}_{stat}": s.loc[m, stat] for m in s.index for stat in ("median", "worst")}
    row["rank_mean"] = df["rank_score"].mean()
    if "rank_active" in df:
        row["rank_active_median"] = df["rank_active"].median()
        row["rank_active_mean"] = df["rank_active"].mean()
        row["rank_active_worst"] = df["rank_active"].max()
    if "rank_llm" in df:
        row["rank_llm_median"] = df["rank_llm"].median()
        row["rank_llm_mean"] = df["rank_llm"].mean()
        row["rank_llm_worst"] = df["rank_llm"].max()
    return row
