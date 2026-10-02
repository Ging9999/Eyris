"""Earnings-event risk control: trim a stock before its earnings price jump.

An announcement before 09:30 ET moves the stock at that day's open; one at or
after 16:00 moves it at the next trading day's open; one during the session is
treated like a pre-open release (trim the day before). The jump date is known
in advance (companies confirm dates weeks ahead); only the outcome is unknown.

Rounds affected for an event whose jump is at the open of day J:
    trim    day before J, rounds >= event_trim_round, and day J rounds < event_restore_round
    restore day J, round == event_restore_round (event_mode "trim_restore" only)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import UNIVERSE

ROOT = Path(__file__).resolve().parents[1]
CALENDAR_FILE = ROOT / "data" / "earnings_dates.csv"
MOVES_FILE = ROOT / "data" / "earnings_moves.csv"


def load_calendar(path=CALENDAR_FILE):
    df = pd.read_csv(path, parse_dates=["announced_at"])
    return df[df["symbol"].isin(UNIVERSE)].drop_duplicates(["symbol", "announced_at"])


def trading_dates(history_days, until="2027-12-31"):
    """Known trading days plus future weekdays (NYSE holidays after the data excluded)."""
    hist = pd.DatetimeIndex(history_days)
    future = pd.bdate_range(hist[-1] + pd.Timedelta(days=1), until)
    holidays = pd.DatetimeIndex(["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
                                 "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
                                 "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
                                 "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"])
    return hist.append(future.difference(holidays))


def jump_dates(calendar, tdays):
    """Trading date whose open reflects each announcement."""
    ts = calendar["announced_at"]
    date = ts.dt.normalize()
    after_close = ts.dt.hour >= 16
    # first trading day >= date (pre-open / intraday), or strictly after it (after the close)
    pos = np.where(after_close, tdays.searchsorted(date, side="right"), tdays.searchsorted(date, side="left"))
    ok = pos < len(tdays)
    out = pd.Series(pd.NaT, index=calendar.index, dtype="datetime64[ns]")
    out[ok] = tdays[pos[ok]]
    return out


class EventSchedule:
    """Per-round trim/restore flags from an earnings calendar."""

    def __init__(self, calendar, tdays, expected_move=None):
        self.tdays = pd.DatetimeIndex(tdays)
        cal = calendar.copy()
        cal["jump"] = jump_dates(cal, self.tdays)
        cal = cal.dropna(subset=["jump"])
        pos = self.tdays.get_indexer(cal["jump"])
        cal = cal[pos > 0]
        cal["prev"] = self.tdays[self.tdays.get_indexer(cal["jump"]) - 1]
        cal["asset"] = cal["symbol"].map({s: i for i, s in enumerate(UNIVERSE)})
        cal = cal.drop_duplicates(["asset", "jump"])
        if expected_move is not None:
            cal["move"] = [expected_move(row.symbol, row.jump) for row in cal.itertuples()]
        else:
            cal["move"] = np.inf
        self.events = cal.reset_index(drop=True)
        self._by_prev = {d: g for d, g in cal.groupby("prev")}
        self._by_jump = {d: g for d, g in cal.groupby("jump")}

    def flags(self, date, number, params):
        """(trim {asset: cut}, restore {asset}) for the round on ``date``, ``number``."""
        if params.event_mode == "off":
            return {}, set()
        date = pd.Timestamp(date).normalize()
        trim, restore = {}, set()
        g = self._by_prev.get(date)
        if g is not None and number >= params.event_trim_round:
            for row in g.itertuples():
                if row.move >= params.event_min_move:
                    trim[int(row.asset)] = params.event_cut
        g = self._by_jump.get(date)
        if g is not None:
            for row in g.itertuples():
                if row.move < params.event_min_move:
                    continue
                if number < params.event_restore_round:
                    trim[int(row.asset)] = params.event_cut
                elif number == params.event_restore_round and params.event_mode == "trim_restore":
                    restore.add(int(row.asset))
        return trim, restore


def historical_moves(p, calendar):
    """Abnormal |open gap| on each past earnings jump, from the price panels."""
    tdays = pd.DatetimeIndex(p.days)
    cal = calendar.copy()
    cal["jump"] = jump_dates(cal, trading_dates(tdays))
    first = np.flatnonzero(np.r_[True, p.day[1:] != p.day[:-1]])
    last = np.r_[first[1:] - 1, len(p) - 1]
    gaps = np.log(p.open[first[1:]] / p.close[last[:-1]])          # gap into day d (d >= 1)
    abn = gaps - gaps.mean(axis=1, keepdims=True)
    rows = []
    for row in cal.dropna(subset=["jump"]).itertuples():
        d = tdays.get_indexer([row.jump])[0]
        if d >= 1:
            rows.append({"symbol": row.symbol, "jump": row.jump,
                         "abs_gap": abs(abn[d - 1, UNIVERSE.index(row.symbol)])})
    return pd.DataFrame(rows).sort_values("jump")


def trailing_move_fn(moves, min_events=2):
    """expected_move(symbol, jump): mean |gap| of that symbol's *earlier* events.

    Falls back to the median of all earlier events. Lookahead-safe.
    """
    by_sym = {s: g for s, g in moves.groupby("symbol")}

    def fn(symbol, jump):
        g = by_sym.get(symbol)
        prior = g[g["jump"] < jump]["abs_gap"] if g is not None else pd.Series(dtype=float)
        if len(prior) >= min_events:
            return float(prior.mean())
        allp = moves[moves["jump"] < jump]["abs_gap"]
        return float(allp.median()) if len(allp) else np.inf
    return fn


def static_move_fn(path=MOVES_FILE):
    """Live: per-symbol mean |gap| over the full development history."""
    m = pd.read_csv(path).set_index("symbol")["mean_abs_gap"]
    return lambda symbol, jump: float(m.get(symbol, np.inf))


def refresh_calendar(cache_dir, today):
    """Live: merge newly confirmed upcoming dates into the committed calendar (once a day)."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"earnings_{pd.Timestamp(today):%Y-%m-%d}.csv"
    base = load_calendar()
    if path.exists():
        return load_calendar(path)
    try:
        import yfinance as yf
        rows = []
        for sym in UNIVERSE:
            d = yf.Ticker(sym).get_earnings_dates(limit=4)
            for ts in d.index:
                ts = pd.Timestamp(ts).tz_convert("America/New_York").tz_localize(None)
                rows.append({"symbol": sym, "announced_at": ts})
        new = pd.DataFrame(rows)
        # a re-dated announcement replaces the stale one in the same quarter (+-20 days)
        keep = []
        for row in base.itertuples():
            same = new[(new.symbol == row.symbol) & ((new.announced_at - row.announced_at).abs() < pd.Timedelta(days=20))]
            if same.empty:
                keep.append({"symbol": row.symbol, "announced_at": row.announced_at})
        merged = pd.concat([pd.DataFrame(keep), new]).drop_duplicates(["symbol", "announced_at"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        merged.sort_values(["symbol", "announced_at"]).to_csv(path, index=False)
        return load_calendar(path)
    except Exception:
        return base
