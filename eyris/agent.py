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
from .execution import HOLD, apply_events, rebalance, sanitize


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

    def target_jump(self, p: Panels, reference=None):
        """(L1, max per-name) change of the base target vs a reference.

        reference: the previous round's logged base target (an independent data
        fetch, so a bad download shows up as a jump). Without one, the target
        recomputed from this snapshot as of the previous completed day.
        """
        if reference is None:
            cut = last_complete_bar(p)
            prev = np.flatnonzero(p.day[:cut + 1] < p.day[cut])
            if len(prev) < 2:
                return 0.0, 0.0
            reference = risk.risk_weights(p.close[:int(prev[-1]) + 1], self.params)
        d = np.abs(self.target(p) - np.asarray(reference, dtype=float))
        return float(d.sum()), float(d.max())

    def target(self, p: Panels):
        w = self.risk_target(p)
        if self.alpha_on:
            if len(p) < alpha.MIN_BARS:
                raise ValueError("not enough history for alpha features")
            feats = self.model.features_at(p) if hasattr(self.model, "features_at") else alpha.features_at(p)
            scores = self.model.predict(feats)
            w = alpha.tilt(w, scores, self.params.tilt, self.params.stock_cap)
        return w

    def decide(self, p: Panels, current_weights, trim=None, restore=None, gross_mult=1.0,
               reference=None) -> Decision:
        """``p`` must contain only bars completed before the submission deadline.

        trim / restore: event flags for this round ({asset: cut}, {asset}), from
        events.EventSchedule and/or the reduce-only news veto.
        gross_mult: target gross multiplier from the VIX overlay (sentiment.py).
        reference: previous round's base target for the circuit breaker (see target_jump).
        """
        try:
            cur = np.asarray(current_weights, dtype=float)
            if cur.shape != (N_ASSETS,) or not np.isfinite(cur).all():
                return Decision(HOLD, reason="invalid current weights")
            if not np.isfinite(p.close[-max(2, len(p)):]).all():
                return Decision(HOLD, reason="non-finite prices")
            if not (np.isfinite(gross_mult) and 0 < gross_mult <= 2):
                return Decision(HOLD, reason="invalid gross multiplier")
            if self.params.breaker_l1 > 0:
                l1, worst = self.target_jump(p, reference)
                if not np.isfinite(l1) or l1 > self.params.breaker_l1 or worst > self.params.breaker_name:
                    return Decision(HOLD, reason=f"circuit breaker: target moved {l1:.3f} L1 "
                                                 f"(max name {worst:.3f}) since the previous day; check the data",
                                    info={"breaker": True, "jump_l1": l1, "jump_name": worst})
            base = self.target(p)
            tgt = base * gross_mult
            if tgt.shape != (N_ASSETS,) or not np.isfinite(tgt).all():
                return Decision(HOLD, reason="non-finite target")
            tgt = sanitize(tgt)
            w = rebalance(cur, tgt, self.params.lam, self.params.band, self.params.min_trade)
            w = apply_events(cur, w, tgt, trim or {}, restore or set(), self.params.min_trade)
            info = {"base_target": [float(x) for x in base]}
            if w is HOLD:
                return Decision(HOLD, tgt, reason="inside no-trade band", info=info)
            reason = "rebalance" + (f"; event trims {sorted(trim)}" if trim else "") + \
                (f"; restores {sorted(restore)}" if restore else "")
            return Decision(w, tgt, reason=reason, info=info)
        except Exception as e:  # never crash: hold the existing portfolio
            return Decision(HOLD, reason=f"error: {type(e).__name__}: {e}")
