import json

import numpy as np
import pandas as pd

from eyris.config import UNIVERSE
from eyris.data import build_panels, build_rounds, clean_long
from eyris.live import decision_payload, resample_to_grid, weights_from_portfolio
from eyris.agent import Decision


def _thirty_minute(day="2026-10-08"):
    times = pd.date_range(f"{day} 09:30", f"{day} 15:30", freq="30min")
    rows = []
    for i, t in enumerate(times):
        for tk in UNIVERSE:
            rows.append({"timestamp": t, "ticker": tk, "open": 100 + i, "high": 101 + i,
                         "low": 99 + i, "close": 100.5 + i, "volume": 10})
    return pd.DataFrame(rows)


def test_resample_matches_historical_grid_and_drops_incomplete_bars():
    df = _thirty_minute()
    g = resample_to_grid(df, pd.Timestamp("2026-10-08 11:25"))
    aapl = g[g.ticker == "AAPL"].set_index("timestamp_et")
    # 09:30 half bar and the completed 10:00-11:00 bar; 11:00 bar still open at 11:25
    assert list(aapl.index.strftime("%H:%M")) == ["09:30", "10:00"]
    bar = aapl.loc["2026-10-08 10:00"]
    assert bar.open == 101 and bar.close == 102.5 and bar.high == 103 and bar.low == 100 and bar.volume == 20


def test_resample_full_day_and_aware_as_of():
    g = resample_to_grid(_thirty_minute(), pd.Timestamp("2026-10-08 16:05", tz="America/New_York"))
    assert sorted(g.timestamp_et.dt.strftime("%H:%M").unique()) == \
        ["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]


def test_early_close_rounds_are_cancelled():
    days = pd.bdate_range("2025-11-24", "2025-12-01")  # includes 2025-11-28 early close
    rows = []
    for d in days:
        for s in ("09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "14:30", "15:00"):
            t = d + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:]))
            for tk in UNIVERSE:
                rows.append({"timestamp_et": t, "ticker": tk, "open": 1.0, "high": 1.0, "low": 1.0,
                             "close": 1.0, "volume": 1})
    df = pd.DataFrame(rows)
    df = df[~((df.timestamp_et.dt.date == pd.Timestamp("2025-11-27").date()))]  # Thanksgiving
    p = build_panels(df)
    r = build_rounds(p)
    early = p.days.get_loc(pd.Timestamp("2025-11-28"))
    assert sorted(r.number[r.day == early]) == [1, 2, 3, 4]
    assert "14:30" not in set(clean_long(df).timestamp_et.dt.strftime("%H:%M"))


def test_portfolio_parsing_shapes():
    prices = np.full(len(UNIVERSE), 100.0)
    cash_only = {"cash": 1e6, "nav": 1e6}
    assert np.all(weights_from_portfolio(cash_only, prices) == 0)
    pos = {"nav": 1e6, "cash": 5e5, "positions": [{"symbol": "AAPL", "shares": 5000}]}
    w = weights_from_portfolio(pos, prices)
    assert w[UNIVERSE.index("AAPL")] == 0.5 and w.sum() == 0.5
    assert weights_from_portfolio({"unexpected": 1}, prices) is None


def test_decision_payload_matches_kit_schema():
    w = np.full(len(UNIVERSE), 1 / 30)
    payload = decision_payload(Decision(w), "validation", "validation-2026-10-08-r1")
    assert list(payload) == ["submission_type", "team_id", "team_token", "phase", "round_id", "weights"]
    assert list(payload["weights"]) == list(UNIVERSE)
    assert sum(payload["weights"].values()) <= 1.0
    assert all(0 <= v <= 0.3 for v in payload["weights"].values())
    json.dumps(payload, allow_nan=False)
