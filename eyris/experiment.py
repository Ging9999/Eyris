"""Shared experiment plumbing: periods, field caching and config evaluation."""
from dataclasses import replace

import numpy as np
import pandas as pd

from . import data
from .agent import Agent
from .backtest import (agent_targets, baseline_policies, evaluate_windows, simulate,
                       summarize, target_policy, windows)
from .config import Params

# Train 2021-2024, validate 2025-01..09 (tuning), hold out 2025-Q4 (touch once).
PERIODS = {
    "train": ("2021-01-01", "2024-12-31"),
    "validation": ("2025-01-01", "2025-09-30"),
    "holdout": ("2025-10-01", "2025-12-31"),
}
FIELD_REFERENCE = Params(risk_method="invvol", lookback_days=20, stock_cap=0.30, gross=1.0)


class Study:
    """Caches data, round ranges, windows and reference-field metrics per period."""

    def __init__(self, period, fill="mid", stride=1):
        self.p, self.r = data.load(fill=fill)
        self.period = period
        days = data.day_range(self.p, *PERIODS[period])
        self.ks = np.arange(np.searchsorted(self.r.day, days[0]), self.r.day_last_round[days[-1]] + 1)
        self.k_first = int(self.ks[0])
        self.wins = windows(self.r, days, stride=stride)
        ref = agent_targets(Agent(FIELD_REFERENCE), self.p, self.r, self.ks)
        self.field = baseline_policies(self.p, self.r, self.k_first, ref)
        self.field_window_metrics = [
            {name: simulate(self.r, k0, k1, pol).metrics() for name, pol in self.field.items()}
            for k0, k1 in self.wins]
        self._targets = {}

    def field_for(self, exclude=None):
        return [[m for n, m in w.items() if n != exclude] for w in self.field_window_metrics]

    def targets(self, params, model=None):
        key = (params.risk_method, params.lookback_days, params.stock_cap, params.gross,
               params.use_alpha, params.tilt, id(model))
        if key not in self._targets:
            self._targets[key] = agent_targets(Agent(params, model), self.p, self.r, self.ks)
        return self._targets[key]

    def evaluate(self, params, model=None):
        pol = target_policy(self.targets(params, model), self.k_first, params)
        return evaluate_windows(self.r, self.wins, pol, self.field_for())

    def evaluate_baseline(self, name):
        return evaluate_windows(self.r, self.wins, self.field[name], self.field_for(exclude=name))

    def continuous(self, policy):
        """One portfolio across the whole period (for equity curves)."""
        return simulate(self.r, self.k_first, int(self.ks[-1]) + 1, policy)


def summary_row(df):
    s = summarize(df)
    row = {f"{m}_{stat}": s.loc[m, stat] for m in s.index for stat in ("median", "worst")}
    row["rank_mean"] = df["rank_score"].mean()
    return row
