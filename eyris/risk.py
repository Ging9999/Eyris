"""Risk-based target weights: inverse volatility and long-only minimum variance."""
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import ledoit_wolf

from .config import ANNUALIZATION, ROUNDS_PER_DAY, SECTORS


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


def ewma_weights(n, halflife_days):
    """Row weights (oldest first) summing to n; flat when halflife_days <= 0."""
    if halflife_days <= 0:
        return np.ones(n)
    age = np.arange(n)[::-1]
    w = 0.5 ** (age / (halflife_days * ROUNDS_PER_DAY))
    return w * n / w.sum()


def inverse_vol(returns, total, cap, halflife_days=0.0, power=1.0, top_k=0):
    w = ewma_weights(len(returns), halflife_days)
    mu = (w[:, None] * returns).sum(0) / w.sum()
    vol = np.sqrt((w[:, None] * (returns - mu) ** 2).sum(0) / w.sum())
    vol = np.where(vol > 1e-12, vol, np.nanmedian(vol[vol > 1e-12]) if (vol > 1e-12).any() else 1.0)
    score = vol ** -power
    if 0 < top_k < len(vol):
        score = np.where(np.argsort(np.argsort(vol, kind="stable"), kind="stable") < top_k, score, 0.0)
    return cap_weights(score, total, cap)


def shrunk_cov(returns, halflife_days=0.0):
    """Ledoit-Wolf covariance; EWMA via sqrt-weighted rows when halflife_days > 0."""
    w = ewma_weights(len(returns), halflife_days)
    mu = (w[:, None] * returns).sum(0) / w.sum()
    cov, _ = ledoit_wolf((returns - mu) * np.sqrt(w)[:, None], assume_centered=True)
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


def hrp(cov, total, cap):
    """Hierarchical Risk Parity (Lopez de Prado, 2016), then capped."""
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = np.ones(len(sd))
    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        left, right = items[:len(items) // 2], items[len(items) // 2:]
        var = []
        for c in (left, right):
            sub = cov[np.ix_(c, c)]
            ivp = 1 / np.diag(sub)
            ivp /= ivp.sum()
            var.append(ivp @ sub @ ivp)
        a = 1 - var[0] / (var[0] + var[1])
        w[left] *= a
        w[right] *= 1 - a
        stack += [left, right]
    return cap_weights(w, total, cap)


def erc(cov, total, cap, iters=500):
    """Equal risk contribution (long-only risk parity) via damped fixed point, then capped."""
    n = cov.shape[0]
    w = 1.0 / np.sqrt(np.diag(cov))
    w /= w.sum()
    for _ in range(iters):
        mrc = cov @ w
        w_new = 0.5 * w + 0.5 * (1.0 / np.maximum(mrc, 1e-18)) / n / np.maximum(w @ mrc, 1e-18) * w.sum()
        w_new /= w_new.sum()
        if np.abs(w_new - w).sum() < 1e-12:
            w = w_new
            break
        w = w_new
    return cap_weights(w, total, cap)


def sector_weights(r, cap, budget="equal", halflife_days=0.0):
    """Inverse vol inside each sector; sector budgets equal or inverse sector-portfolio vol."""
    n = r.shape[1]
    w = np.zeros(n)
    budgets = []
    for idx in SECTORS:
        idx = list(idx)
        ws = inverse_vol(r[:, idx], 1.0, 1.0, halflife_days)
        w[idx] = ws
        budgets.append(1.0 if budget == "equal" else 1.0 / max(np.std(r[:, idx] @ ws), 1e-12))
    b = np.asarray(budgets) / np.sum(budgets)
    for j, idx in enumerate(SECTORS):
        w[list(idx)] *= b[j]
    return cap_weights(w, 1.0, cap)


def base_weights(r, params, cap):
    """Shape of the stock book, summing to 1."""
    m = params.risk_method
    if m == "ew":
        return cap_weights(np.ones(r.shape[1]), 1.0, cap)
    if m == "invvol":
        return inverse_vol(r, 1.0, cap, params.halflife_days, params.vol_power, params.top_k)
    if m in ("sector_eq", "sector_invvol"):
        return sector_weights(r, cap, "equal" if m == "sector_eq" else "invvol", params.halflife_days)
    cov = shrunk_cov(r, params.halflife_days)
    if m == "minvar":
        return min_variance(cov, 1.0, cap)
    if m == "blend":
        return 0.5 * inverse_vol(r, 1.0, cap, params.halflife_days) + 0.5 * min_variance(cov, 1.0, cap)
    if m == "hrp":
        return hrp(cov, 1.0, cap)
    if m == "erc":
        return erc(cov, 1.0, cap)
    raise ValueError(m)


def exposure(r, close, w, params):
    """Gross exposure in [gross_min, gross] from the vol-target / trend overlays."""
    g = params.gross
    mode = params.gross_mode
    if mode == "fixed":
        return g
    if "voltarget" in mode:
        vol = np.sqrt(w @ shrunk_cov(r, params.halflife_days) @ w) * ANNUALIZATION
        g = min(g, params.vol_target / vol) if vol > 0 else g
    if "trend" in mode:
        n = params.trend_days * ROUNDS_PER_DAY
        basket = np.log(close[-1] / close[-1 - n]).mean() if len(close) > n else 0.0
        if basket <= 0:
            g *= params.trend_floor
    return float(np.clip(g, params.gross_min, params.gross))


def risk_weights(close, params):
    """Risk target from a close-price history (bars <= info cutoff).

    Returns weights summing to the exposure (``params.gross`` when fixed), each
    <= ``params.stock_cap``.
    """
    n = close.shape[1]
    cap = params.stock_cap
    if params.gross <= 0:
        return np.zeros(n)
    r = bar_returns(close, params.lookback_days)
    if r.shape[0] < 2 * n:
        return cap_weights(np.ones(n), params.gross, cap)
    w = base_weights(r, params, cap)
    return cap_weights(w, exposure(r, close, w, params), cap)
