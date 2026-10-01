import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eyris.config import BAR_SLOTS, UNIVERSE  # noqa: E402
from eyris.data import DEFAULT_PARQUET, build_panels, build_rounds  # noqa: E402


def synthetic_long(n_days=70, seed=0, start="2025-03-03"):
    """Random-walk hourly bars on the historical grid for all 30 symbols."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    times = [d + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:])) for d in days for s in BAR_SLOTS]
    rows = []
    for j, tk in enumerate(UNIVERSE):
        lp = np.log(50 + 10 * j) + np.cumsum(rng.normal(0, 0.004 + 0.0002 * j, len(times)))
        c = np.exp(lp)
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.001, len(times)))
        hi = np.maximum(o, c) * 1.002
        lo = np.minimum(o, c) * 0.998
        v = rng.integers(1e5, 1e6, len(times))
        rows.append(pd.DataFrame({"timestamp_et": times, "ticker": tk, "open": o, "high": hi,
                                  "low": lo, "close": c, "volume": v}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def synth():
    df = synthetic_long()
    p = build_panels(df)
    return df, p, build_rounds(p)


@pytest.fixture(scope="session")
def real():
    if not Path(DEFAULT_PARQUET).exists():
        pytest.skip("historical parquet not downloaded")
    from eyris import data
    return data.load()
