"""Optional LightGBM cross-sectional ranker used as a small tilt on risk weights.

Features at bar t use only bars <= t (backward rolling windows). The label is
the cross-sectional rank of the log return from a round's execution price to
the execution price ``horizon`` rounds later.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROUNDS_PER_DAY
from .risk import cap_weights

FEATURES = ("rev_1b", "rev_1d", "mom_5d", "mom_20d", "mom_40d_skip1d",
            "vol_20d", "volume_z", "range_1d")
MIN_BARS = 40 * ROUNDS_PER_DAY + 2
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  random_state=0, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)


def _cs_rank(x):
    """Cross-sectional rank scaled to [-0.5, 0.5] along the last axis (NaN-safe)."""
    r = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy()
    return np.nan_to_num(r - 0.5, nan=0.0)


def raw_features(close, volume, high, low):
    """(T, N, F) raw features; row t depends only on rows <= t."""
    d = ROUNDS_PER_DAY
    lc = pd.DataFrame(np.log(close))
    ret = lc.diff()
    vol = pd.DataFrame(volume)
    vsum = vol.rolling(d).sum()
    rng = pd.DataFrame((high - low) / close)
    f = [
        ret,
        lc - lc.shift(d),
        lc - lc.shift(5 * d),
        lc - lc.shift(20 * d),
        lc.shift(d) - lc.shift(40 * d),
        ret.rolling(20 * d).std(),
        np.log1p(vsum) - np.log1p(vsum.rolling(20 * d).mean()),
        rng.rolling(d).mean(),
    ]
    return np.stack([x.to_numpy() for x in f], axis=-1)


def cs_features(raw):
    """Cross-sectionally rank each feature: (K, N, F) -> (K, N, F)."""
    return np.stack([_cs_rank(raw[:, :, j]) for j in range(raw.shape[2])], axis=-1)


def features_at(panels):
    """Features for a single decision from bars up to the cutoff: (N, F)."""
    tail = slice(max(0, len(panels) - MIN_BARS), len(panels))
    raw = raw_features(panels.close[tail], panels.volume[tail], panels.high[tail], panels.low[tail])
    return cs_features(raw[-1:])[0]


def labels(price, horizon=ROUNDS_PER_DAY):
    """Cross-sectional rank of forward log return over ``horizon`` rounds: (K, N)."""
    lp = np.log(price)
    fwd = np.full_like(lp, np.nan)
    fwd[:-horizon] = lp[horizon:] - lp[:-horizon]
    y = _cs_rank(fwd)
    y[np.isnan(fwd).any(axis=1)] = np.nan
    return y


class AlphaModel:
    def __init__(self, booster=None):
        self.booster = booster

    @classmethod
    def fit(cls, X, y):
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X.reshape(-1, X.shape[-1]), y.reshape(-1))
        return cls(model.booster_)

    def predict(self, X):
        """X: (..., N, F) -> scores (..., N)."""
        shape = X.shape[:-1]
        return self.booster.predict(X.reshape(-1, X.shape[-1]), num_threads=1).reshape(shape)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path):
        import lightgbm as lgb
        return cls(lgb.Booster(model_file=str(path)))


def zscore(scores):
    s = np.asarray(scores, dtype=float)
    sd = s.std(axis=-1, keepdims=True)
    return np.where(sd > 1e-12, (s - s.mean(axis=-1, keepdims=True)) / np.where(sd > 1e-12, sd, 1), 0.0)


def tilt(risk_w, scores, strength, cap):
    """Multiplicative tilt w_i * exp(strength * z_i), same gross, same cap."""
    total = risk_w.sum()
    if strength == 0 or total <= 0:
        return risk_w
    z = np.clip(zscore(scores), -3, 3)
    return cap_weights(risk_w * np.exp(strength * z), total, cap)
