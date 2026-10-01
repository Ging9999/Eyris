"""Risk-based target weights: inverse volatility and long-only minimum variance."""
import numpy as np
from sklearn.covariance import ledoit_wolf

from .config import ROUNDS_PER_DAY


def bar_returns(close, lookback_days):
    """Log returns of the last ``lookback_days`` days of bars (rows = bars)."""
    n = lookback_days * ROUNDS_PER_DAY
    c = close[-(n + 1):]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(c), axis=0)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def project_capped_simplex(v, total, cap):
    """Euclidean projection onto {0 <= w <= cap, sum(w) = total}."""
    lo, hi = v.min() - cap, v.max()
    for _ in range(60):
        tau = 0.5 * (lo + hi)
        s = np.clip(v - tau, 0.0, cap).sum()
        if s > total:
            lo = tau
        else:
            hi = tau
    return np.clip(v - 0.5 * (lo + hi), 0.0, cap)


def cap_weights(w, total, cap):
    """Scale positive scores to sum ``total`` with per-name cap, redistributing excess."""
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum() * total
    for _ in range(len(w)):
        over = w > cap + 1e-15
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return np.minimum(w, cap)


def inverse_vol(returns, total, cap):
    vol = returns.std(axis=0)
    vol = np.where(vol > 1e-12, vol, np.nanmedian(vol[vol > 1e-12]) if (vol > 1e-12).any() else 1.0)
    return cap_weights(1.0 / vol, total, cap)


def shrunk_cov(returns):
    cov, _ = ledoit_wolf(returns, assume_centered=False)
    return cov


def min_variance(cov, total, cap, iters=300):
    """Long-only, capped min-variance via projected gradient (deterministic)."""
    n = cov.shape[0]
    step = 1.0 / max(np.linalg.eigvalsh(cov)[-1], 1e-18)
    w = np.full(n, total / n)
    w = project_capped_simplex(w, total, cap)
    for _ in range(iters):
        w_new = project_capped_simplex(w - step * (cov @ w), total, cap)
        if np.abs(w_new - w).sum() < 1e-10:
            w = w_new
            break
        w = w_new
    return w


def risk_weights(close, params):
    """Risk target from a close-price history (bars <= info cutoff).

    Returns weights summing to ``params.gross`` with each <= ``params.stock_cap``.
    """
    n = close.shape[1]
    total, cap = params.gross, params.stock_cap
    if total <= 0:
        return np.zeros(n)
    if params.risk_method == "ew":
        return cap_weights(np.ones(n), total, cap)
    r = bar_returns(close, params.lookback_days)
    if r.shape[0] < 2 * n:
        return cap_weights(np.ones(n), total, cap)
    if params.risk_method == "invvol":
        return inverse_vol(r, total, cap)
    cov = shrunk_cov(r)
    if params.risk_method == "minvar":
        return min_variance(cov, total, cap)
    if params.risk_method == "blend":
        return 0.5 * inverse_vol(r, total, cap) + 0.5 * min_variance(cov, total, cap)
    raise ValueError(params.risk_method)
