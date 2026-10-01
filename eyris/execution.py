"""Turn a target portfolio into the weights actually submitted.

    w = w_old + lam * (target - w_old)

with a no-trade band: if the whole move is smaller than ``band`` (L1 distance
between target and current weights) the round is skipped. Skipping means *not
uploading* a decision: the backend then holds the portfolio with zero fee and
zero turnover. Re-submitting the drifted current weights instead would still
trade a little, because prices move between the deadline and the execution.
"""
import numpy as np

from .config import MAX_WEIGHT

HOLD = None  # sentinel: do not submit this round


def sanitize(w, cap=MAX_WEIGHT):
    """Force weights into the official constraints: finite, 0 <= w <= cap, sum <= 1."""
    w = np.asarray(w, dtype=float)
    if w.shape != (len(w),) or not np.isfinite(w).all():
        raise ValueError("weights must be a finite vector")
    w = np.clip(w, 0.0, cap)
    s = w.sum()
    if s > 1.0:
        w = w / s
    # Floor to 1e-8 so JSON decimals can never sum above 1 or exceed the cap.
    return np.floor(w * 1e8) / 1e8


def rebalance(w_old, target, lam, band, min_trade=0.0):
    """Return new weights, or HOLD when the trade is not worth its cost."""
    w_old = np.asarray(w_old, dtype=float)
    target = np.asarray(target, dtype=float)
    if not (np.isfinite(w_old).all() and np.isfinite(target).all()):
        return HOLD
    gap = target - w_old
    if np.abs(gap).sum() < band:
        return HOLD
    # Building the first position from cash: go straight to target. Turnover
    # is the same either way and partial entry only delays exposure.
    step = 1.0 if w_old.sum() < 1e-6 else lam
    delta = step * gap
    if min_trade > 0:
        delta = np.where(np.abs(delta) < min_trade, 0.0, delta)
    if not delta.any():
        return HOLD
    return sanitize(w_old + delta)
