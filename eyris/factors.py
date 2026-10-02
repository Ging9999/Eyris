"""Slow, daily cross-sectional factors and a linear factor model for the tilt.

Hypotheses were proposed AlphaAgent-style (LLM-generated, literature-grounded)
and are validated by scripts/agentic_factor_mining.py. Each factor is computed
from *completed* trading days only, so it changes at most once a day (low
turnover) and is identical for all rounds of a day.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROUNDS_PER_DAY, UNIVERSE

SECTOR = np.repeat(np.arange(6), 5)  # UNIVERSE is ordered in six sector blocks of five

# name -> (hypothesis, source)
HYPOTHESES = {
    "bab": ("low-beta stocks earn more per unit risk (sign: -beta, 20d)", "Frazzini & Pedersen 2014"),
    "ivol": ("high idiosyncratic volatility underperforms (sign: -ivol)", "Ang, Hodrick, Xing & Zhang 2006"),
    "max": ("lottery-like max daily return underperforms (sign: -max)", "Bali, Cakici & Whitelaw 2011"),
    "mom20": ("20-day momentum, skipping the last day", "Jegadeesh & Titman; short-horizon variant"),
    "resid_mom": ("residual (beta-adjusted) momentum is cleaner", "Blitz, Huij & Martens 2011"),
    "rev5": ("weekly reversal (sign: -5d return)", "Lehmann 1990; Lo & MacKinlay 1990"),
    "sector_rev5": ("industry-relative weekly reversal", "Da, Liu & Schaumburg 2014"),
    "near_high": ("near the recent high continues (price / 40d high)", "George & Hwang 2004"),
    "volume_trend": ("rising attention reverses (sign: -volume trend)", "Barber & Odean 2008"),
    "overnight_mom": ("overnight returns persist", "Lou, Polk & Skouras 2019"),
    "intraday_rev": ("intraday returns reverse across periods (sign: -intraday)", "Lou, Polk & Skouras 2019"),
}
NAMES = tuple(HYPOTHESES)
MIN_DAYS = 45  # longest chain: 20d beta -> 20d residual window (+ lags); live has ~59


def daily_panels(p):
    """Daily open/close/volume from bars; row d uses day d's bars only."""
    first = np.flatnonzero(np.r_[True, p.day[1:] != p.day[:-1]])
    last = np.r_[first[1:] - 1, len(p) - 1]
    vol = np.add.reduceat(p.volume, first, axis=0)
    return p.open[first], p.close[last], vol


def daily_factors(p):
    """(D, N, F) raw factor values; row d depends only on days <= d."""
    o, c, v = daily_panels(p)
    lc = pd.DataFrame(np.log(c))
    r = lc.diff()
    mkt = r.mean(axis=1)
    beta = _rolling_beta(r, mkt, 20)
    resid = r - beta.mul(mkt, axis=0)
    on = pd.DataFrame(np.log(o[1:] / c[:-1]))
    on = pd.concat([pd.DataFrame(np.nan, index=[0], columns=on.columns), on], ignore_index=True)
    intra = pd.DataFrame(np.log(c / o))
    r5 = lc - lc.shift(5)
    sect = pd.DataFrame({j: r5.iloc[:, SECTOR == SECTOR[j]].mean(axis=1) for j in range(c.shape[1])})
    lv = pd.DataFrame(np.log1p(v))
    f = {
        "bab": -beta,
        "ivol": -resid.rolling(20).std(),
        "max": -r.rolling(20).max(),
        "mom20": lc.shift(1) - lc.shift(21),
        "resid_mom": resid.shift(1).rolling(20).sum(),
        "rev5": -r5,
        "sector_rev5": -(r5 - sect),
        "near_high": lc - np.log(pd.DataFrame(c).rolling(40).max()),
        "volume_trend": -(lv.rolling(5).mean() - lv.rolling(40).mean()),
        "overnight_mom": on.rolling(20).sum(),
        "intraday_rev": -intra.rolling(20).sum(),
    }
    return np.stack([f[n].to_numpy() for n in NAMES], axis=-1)


def _rolling_beta(r, mkt, n):
    m_mean = mkt.rolling(n).mean()
    cov = r.mul(mkt, axis=0).rolling(n).mean().sub(r.rolling(n).mean().mul(m_mean, axis=0))
    var = (mkt * mkt).rolling(n).mean() - m_mean ** 2
    return cov.div(var, axis=0)


def zscores(x):
    """Cross-sectional z-score along the asset axis, NaN -> 0, clipped at 3."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=-2, keepdims=True)
        sd = np.nanstd(x, axis=-2, keepdims=True)
    z = (x - mu) / np.where(sd > 1e-12, sd, np.inf)
    return np.clip(np.nan_to_num(z, nan=0.0), -3, 3)


class FactorModel:
    """Linear combination of z-scored daily factors; drop-in for AlphaModel."""

    def __init__(self, weights):
        self.weights = {k: float(v) for k, v in weights.items() if k in NAMES and v != 0}
        self._w = np.array([self.weights.get(n, 0.0) for n in NAMES])

    def features_at(self, p):
        """(N, F) z-scored factors at the last completed day of ``p``."""
        from .agent import last_complete_bar
        cut = last_complete_bar(p)
        head = p.head(cut)
        start = max(0, int(np.searchsorted(head.day, head.day[-1] - MIN_DAYS - 1)))
        return zscores(daily_factors(_tail_days(head, start))[-1])

    def batch_features(self, p, info_end):
        """(K, N, F) features for many rounds (same values as ``features_at``)."""
        from .agent import last_complete_bar
        z = zscores(daily_factors(p))
        days = np.array([p.day[last_complete_bar(p.head(int(e)))] for e in info_end])
        return z[days]

    def predict(self, X):
        return X @ self._w

    def save(self, path):
        Path(path).write_text(json.dumps({"weights": self.weights}, indent=1))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["weights"])


def _tail_days(p, start_bar):
    from .data import Panels
    s = slice(start_bar, len(p))
    day = p.day[s] - p.day[start_bar]
    return Panels(p.times[s], day, p.slot[s], p.days[p.day[start_bar]:p.day[-1] + 1],
                  p.open[s], p.high[s], p.low[s], p.close[s], p.volume[s])
