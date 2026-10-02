"""Market sentiment from the VIX "fear index": a contrarian gross overlay.

When fear is high (VIX above a level, or a VIX spike vs its last 60 days) the
target's gross is scaled by ``vix_boost`` (e.g. 0.5 -> 0.7). For a round on day
d only closes of days strictly before d are used: the VIX close is published
after 16:00, so day d's own close is never known at a day-d deadline.

The functions here are pure. Fetching happens only in the live pre-step
(``live.vix_multiplier``); any failure there means multiplier 1 (no overlay).
Evidence and caveats: reports/sentiment_experiment.md.
"""
import numpy as np
import pandas as pd

Z_WINDOW = 60


def signals(vix, days):
    """Per trading day in ``days``: the latest VIX / VIX3M values known before that day.

    ``vix``: DataFrame indexed by date with columns ``vix`` and ``vix3m`` (daily closes).
    """
    v = vix[["vix", "vix3m"]].astype(float).ffill().copy()
    v.index = pd.DatetimeIndex(v.index).normalize()
    v = v[~v.index.duplicated(keep="last")].sort_index()
    v["ratio"] = v.vix / v.vix3m
    v["z"] = (v.vix - v.vix.rolling(Z_WINDOW).mean()) / v.vix.rolling(Z_WINDOW).std()
    days = pd.DatetimeIndex(days).normalize()
    # last row dated strictly before each day, whether or not the feed already has a row for that day
    pos = v.index.searchsorted(days, side="left") - 1
    out = v.iloc[np.maximum(pos, 0)].copy()
    out[pos < 0] = np.nan
    out.index = days
    return out


def fear(sig, mode, threshold):
    if mode == "level":
        f = sig.vix > threshold
    elif mode == "z":
        f = sig.z > threshold
    else:
        return np.zeros(len(sig), dtype=bool)
    return f.fillna(False).to_numpy(dtype=bool)


def multipliers(vix, days, params):
    """Gross multiplier for each day in ``days`` (1.0 when the overlay is off or calm)."""
    if params.vix_mode == "off":
        return np.ones(len(days))
    return np.where(fear(signals(vix, days), params.vix_mode, params.vix_threshold), params.vix_boost, 1.0)


def multiplier(vix, day, params):
    """Gross multiplier for one trading day (date-like)."""
    return float(multipliers(vix, [pd.Timestamp(day)], params)[0])
