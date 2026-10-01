"""Competition agent: data (bars <= cutoff) + current weights -> decision.

The decision path is pure and offline: no network, no LLM, no randomness.
Any error or non-finite value yields HOLD (no upload, portfolio unchanged).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import alpha, risk
from .config import N_ASSETS, UNIVERSE, Params
from .data import EARLY_CLOSE_DAYS, EARLY_CLOSE_LAST_SLOT, Panels
from .execution import HOLD, rebalance, sanitize


@dataclass
class Decision:
    weights: Optional[np.ndarray]          # None = HOLD, do not submit
    target: Optional[np.ndarray] = None
    reason: str = ""
    info: dict = field(default_factory=dict)

    @property
    def hold(self):
        return self.weights is None

    def as_dict(self):
        if self.hold:
            return None
        return {s: float(w) for s, w in zip(UNIVERSE, self.weights)}


def last_complete_bar(p: Panels):
    """Index of the last bar of the last fully completed trading day in ``p``."""
    i = len(p) - 1
    s, date = int(p.slot[i]), p.days[p.day[i]]
    last_slot = EARLY_CLOSE_LAST_SLOT if date in EARLY_CLOSE_DAYS else 6
    if s >= last_slot:
        return i
    return int(np.flatnonzero(p.day < p.day[i])[-1]) if p.day[0] < p.day[i] else -1


class Agent:
    def __init__(self, params: Params = Params(), model: Optional[alpha.AlphaModel] = None):
        self.params = params
        self.model = model
        self._risk_cache = {}

    @property
    def alpha_on(self):
        return self.params.use_alpha and self.params.tilt > 0 and self.model is not None

    def risk_target(self, p: Panels):
        """Risk weights from completed days only, so they are stable intraday."""
        cut = last_complete_bar(p)
        if cut < 1:
            raise ValueError("not enough history")
        key = p.times[cut]
        if key not in self._risk_cache:
            self._risk_cache = {key: risk.risk_weights(p.close[:cut + 1], self.params)}
        return self._risk_cache[key]

    def target(self, p: Panels):
        w = self.risk_target(p)
        if self.alpha_on:
            if len(p) < alpha.MIN_BARS:
                raise ValueError("not enough history for alpha features")
            scores = self.model.predict(alpha.features_at(p))
            w = alpha.tilt(w, scores, self.params.tilt, self.params.stock_cap)
        return w

    def decide(self, p: Panels, current_weights) -> Decision:
        """``p`` must contain only bars completed before the submission deadline."""
        try:
            cur = np.asarray(current_weights, dtype=float)
            if cur.shape != (N_ASSETS,) or not np.isfinite(cur).all():
                return Decision(HOLD, reason="invalid current weights")
            if not np.isfinite(p.close[-max(2, len(p)):]).all():
                return Decision(HOLD, reason="non-finite prices")
            tgt = self.target(p)
            if tgt.shape != (N_ASSETS,) or not np.isfinite(tgt).all():
                return Decision(HOLD, reason="non-finite target")
            tgt = sanitize(tgt)
            w = rebalance(cur, tgt, self.params.lam, self.params.band, self.params.min_trade)
            if w is HOLD:
                return Decision(HOLD, tgt, reason="inside no-trade band")
            return Decision(w, tgt, reason="rebalance")
        except Exception as e:  # never crash: hold the existing portfolio
            return Decision(HOLD, reason=f"error: {type(e).__name__}: {e}")
