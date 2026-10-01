"""Load and clean hourly bars and align them to the competition decision rounds.

Timeline (all US/Eastern). Bars are labelled by start time:
    slot 0: 09:30-10:00, slot 1: 10:00-11:00, ..., slot 6: 15:00-16:00.

Round r executes at EXEC_TIMES[r-1]:
    r = 1: 09:30 -> official price is the open of slot 0.
    r >= 2: HH:30 falls inside slot r-1. The historical data has no HH:30 print,
            so the backtest proxies the fill with that bar's (open+close)/2.
The submission deadline for round r is before the end of slot r-1, so the last
fully completed bar at decision time is always the bar *before* the execution
bar: ``info_end = exec_bar - 1``. Every feature must use bars <= info_end.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BAR_SLOTS, UNIVERSE

DATA_URL = ("https://hackathon2.deepintomlf.ai/datasets/download/"
            "8bfeba71-171b-47a1-93f8-2abf846293be/")
DEFAULT_PARQUET = Path(__file__).resolve().parents[1] / "data" / "hourly_market_data_2021_2026.parquet"

# NYSE 13:00 early closes in the data range and the competition window. The
# dataset keeps thin after-hours bars on these days; they are dropped.
EARLY_CLOSE_DAYS = frozenset(pd.Timestamp(d) for d in (
    "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24", "2024-07-03",
    "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
))
EARLY_CLOSE_LAST_SLOT = 3  # 12:00-13:00 bar is the last one

COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class Panels:
    """Wide bar arrays, shape (T, N) with columns in UNIVERSE order."""
    times: pd.DatetimeIndex      # bar start times (naive ET)
    day: np.ndarray              # (T,) trading-day index of each bar
    slot: np.ndarray             # (T,) slot 0..6
    days: pd.DatetimeIndex       # (D,) trading dates
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self):
        return len(self.times)

    def head(self, end):
        """Bars [0, end] inclusive, as views. Used to enforce the info cutoff."""
        s = slice(0, end + 1)
        dmax = int(self.day[end]) + 1 if end >= 0 else 0
        return Panels(self.times[s], self.day[s], self.slot[s], self.days[:dmax],
                      self.open[s], self.high[s], self.low[s], self.close[s], self.volume[s])


@dataclass
class Rounds:
    """One row per scheduled (non-cancelled) decision round."""
    day: np.ndarray          # (K,) trading-day index
    number: np.ndarray       # (K,) round number 1..7
    exec_bar: np.ndarray     # (K,) bar index containing the execution time
    info_end: np.ndarray     # (K,) last bar index visible to the agent
    price: np.ndarray        # (K, N) execution price proxy
    day_close: np.ndarray    # (D, N) official 16:00 (or early) close
    day_last_round: np.ndarray  # (D,) index of each day's last round, -1 if none

    def __len__(self):
        return len(self.day)


def load_long(path=DEFAULT_PARQUET):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python scripts/download_data.py`")
    return pd.read_parquet(path)


def clean_long(df):
    """Normalize a long OHLCV frame to the historical schema and bar grid."""
    df = df.rename(columns={"timestamp": "timestamp_et"})
    need = {"timestamp_et", "ticker", *COLUMNS}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    df = df[list(need)].copy()
    df["timestamp_et"] = pd.to_datetime(df["timestamp_et"])
    if df["timestamp_et"].dt.tz is not None:
        df["timestamp_et"] = df["timestamp_et"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df[df["ticker"].isin(UNIVERSE)]
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    bad = ~np.isfinite(df[["open", "high", "low", "close"]]).all(axis=1) | (df["close"] <= 0)
    df = df[~bad]
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    hhmm = df["timestamp_et"].dt.strftime("%H:%M")
    df = df[hhmm.isin(BAR_SLOTS)]
    date = df["timestamp_et"].dt.normalize()
    slot = df["timestamp_et"].dt.strftime("%H:%M").map({s: i for i, s in enumerate(BAR_SLOTS)})
    early = date.isin(EARLY_CLOSE_DAYS) & (slot > EARLY_CLOSE_LAST_SLOT)
    df = df[~early]
    return df.drop_duplicates(["timestamp_et", "ticker"], keep="last").sort_values(["timestamp_et", "ticker"])


def build_panels(df):
    """Long frame -> Panels on the union bar grid, forward-filling missing bars."""
    df = clean_long(df)
    wide = {c: df.pivot(index="timestamp_et", columns="ticker", values=c).reindex(columns=list(UNIVERSE))
            for c in COLUMNS}
    close = wide["close"].ffill()
    filled = wide["close"].isna()
    out = {"close": close}
    for c in ("open", "high", "low"):
        out[c] = wide[c].where(~filled, close)
    out["volume"] = wide["volume"].fillna(0.0)
    times = close.index
    first_full = int(np.argmax(close.notna().all(axis=1).to_numpy()))
    if not close.iloc[first_full].notna().all():
        raise ValueError("no bar where all symbols have a price")
    times = times[first_full:]
    dates = times.normalize()
    days = pd.DatetimeIndex(dates.unique())
    day = days.get_indexer(dates)
    slot = np.array([BAR_SLOTS.index(t) for t in times.strftime("%H:%M")])
    arr = {c: out[c].iloc[first_full:].to_numpy(dtype=float) for c in COLUMNS}
    return Panels(times=times, day=day, slot=slot, days=days, **arr)


def build_rounds(p: Panels, fill="mid"):
    """Decision rounds and their execution prices.

    fill: proxy for HH:30 executions inside a clock-hour bar.
        "mid" = (open+close)/2 (default), "open", or "close" (conservative).
    """
    if fill not in ("mid", "open", "close"):
        raise ValueError(fill)
    rows = []
    for i in range(1, len(p)):  # bar 0 has no prior information
        s = int(p.slot[i])
        if s > 0 and p.slot[i - 1] != s - 1:
            continue  # gap in the grid: no clean round
        rows.append((int(p.day[i]), s + 1, i))
    day = np.array([r[0] for r in rows])
    number = np.array([r[1] for r in rows])
    exec_bar = np.array([r[2] for r in rows])
    if fill == "mid":
        mid = 0.5 * (p.open[exec_bar] + p.close[exec_bar])
    else:
        mid = getattr(p, fill)[exec_bar]
    price = np.where((number == 1)[:, None], p.open[exec_bar], mid)
    nd = len(p.days)
    last_bar = np.full(nd, -1)
    last_bar[p.day] = np.arange(len(p))  # last write wins -> last bar of each day
    day_close = p.close[last_bar]
    day_last_round = np.full(nd, -1)
    day_last_round[day] = np.arange(len(day))
    return Rounds(day=day, number=number, exec_bar=exec_bar, info_end=exec_bar - 1,
                  price=price, day_close=day_close, day_last_round=day_last_round)


def load(path=DEFAULT_PARQUET, fill="mid"):
    p = build_panels(load_long(path))
    return p, build_rounds(p, fill=fill)


def day_range(p: Panels, start, end):
    """Indices of trading days in [start, end] (inclusive dates)."""
    d = p.days
    return np.flatnonzero((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)))
