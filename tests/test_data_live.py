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


def test_decide_never_writes_state_and_hold_clears_stale_decision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from eyris import live
    monkeypatch.setattr(live, "PRIVATE", tmp_path)
    monkeypatch.setattr(live, "load_agent", lambda: SimpleNamespace(params=live.Params()))
    monkeypatch.setattr(live, "round_flags", lambda *a: (None, None, {}))
    snap = tmp_path / "snap.parquet"
    _thirty_minute().to_parquet(snap)
    as_of = pd.Timestamp("2026-10-08 16:30", tz="America/New_York")
    rid = "validation-2026-10-08-r1"

    w = np.full(len(UNIVERSE), 0.01)
    monkeypatch.setattr(live, "decide_round", lambda *a, **k: Decision(w, target=w, reason="rebalance"))
    path, log = live.prepare("validation", rid, as_of=as_of, snapshot=snap)
    assert path.exists() and log["prices"]["AAPL"] > 0
    assert not (tmp_path / "state.json").exists()      # nothing was uploaded, so no state

    monkeypatch.setattr(live, "decide_round", lambda *a, **k: Decision(None, reason="inside no-trade band"))
    path, _ = live.prepare("validation", rid, as_of=as_of, snapshot=snap)
    assert path is None and not (tmp_path / rid / "decision.json").exists()


# --------------------------------------------------------------------------- replay, breaker, alerts
def _random_30m(days=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-07-01", periods=days)
    times = [d + pd.Timedelta(minutes=30 * i) + pd.Timedelta(hours=9, minutes=30) for d in dates for i in range(13)]
    rows = []
    for j, tk in enumerate(UNIVERSE):
        px = 100 * np.exp(np.cumsum(rng.standard_normal(len(times)) * 0.002 * (1 + j / 10)))
        for t, c in zip(times, px):
            rows.append({"timestamp": t, "ticker": tk, "open": c, "high": c * 1.001, "low": c * 0.999,
                         "close": c, "volume": 100})
    return pd.DataFrame(rows), dates


def _live_env(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from eyris import live
    from eyris.agent import Agent
    from eyris.config import Params
    monkeypatch.setattr(live, "PRIVATE", tmp_path / "private")
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live, "load_params", lambda: Params(lookback_days=20, gross=0.5, lam=0.25))
    monkeypatch.setattr(live, "round_flags", lambda *a: ({}, set(), {}))
    df, dates = _random_30m()
    snap = tmp_path / "snap.parquet"
    df.to_parquet(snap)
    return live, snap, dates


def test_replay_reproduces_logged_decisions_and_detects_tampering(tmp_path, monkeypatch):
    live, snap, dates = _live_env(tmp_path, monkeypatch)
    rid = f"validation-{dates[-1].date()}-r1"
    as_of = pd.Timestamp(f"{dates[-1].date()} 09:09", tz="America/New_York")
    path, log = live.prepare("validation", rid, as_of=as_of, snapshot=snap, first_round=True)
    assert path is not None and log["inputs"]["current_weights"] == [0.0] * len(UNIVERSE)
    [res] = live.replay_all()
    assert res["status"] == "MATCH" and res["max_diff_vs_file"] == 0.0

    sub = json.loads(path.read_text())                       # an edited decision.json is caught
    sub["weights"]["AAPL"] += 0.01
    path.write_text(json.dumps(sub))
    assert live.replay_all()[0]["status"].startswith("MISMATCH")

    df = pd.read_parquet(snap)                               # a changed snapshot is caught
    df.loc[0, "close"] *= 1.01
    df.to_parquet(snap)
    assert live.replay_all()[0]["status"] == "SNAPSHOT_CHANGED"


def test_circuit_breaker_holds_on_a_jump_vs_the_previous_round(tmp_path, monkeypatch):
    live, snap, dates = _live_env(tmp_path, monkeypatch)
    d1, d2 = dates[-2].date(), dates[-1].date()
    as1 = pd.Timestamp(f"{d1} 09:09", tz="America/New_York")
    _, log1 = live.prepare("validation", f"validation-{d1}-r1", as_of=as1, snapshot=snap, first_round=True)
    assert not log1["breaker"] and log1["base_target"]
    # normal next day: reference found, no breaker, and replay still matches
    as2 = pd.Timestamp(f"{d2} 09:09", tz="America/New_York")
    _, log2 = live.prepare("validation", f"validation-{d2}-r1", as_of=as2, snapshot=snap, first_round=True)
    assert log2["inputs"]["reference_round"] == f"validation-{d1}-r1" and not log2["breaker"]
    # a corrupted reference (as if yesterday's download had been very different) trips it
    f = tmp_path / "private" / f"validation-{d1}-r1" / "decision_log.json"
    bad = json.loads(f.read_text())
    bad["base_target"] = [0.5 / len(UNIVERSE)] * len(UNIVERSE)
    bad["base_target"][0] += 0.05
    f.write_text(json.dumps(bad))
    path, log3 = live.prepare("validation", f"validation-{d2}-r1", as_of=as2, snapshot=snap, first_round=True)
    assert path is None and log3["breaker"] and log3["reason"].startswith("circuit breaker")
    assert {r["status"] for r in live.replay_all()} == {"MATCH"}


def test_alerts_are_safe_and_flag_problems(monkeypatch):
    from eyris import alerts
    for k in ("NTFY_TOPIC", "DISCORD_WEBHOOK_URL"):
        monkeypatch.delenv(k, raising=False)
    assert alerts.notify("t", "m") == []                     # nothing configured: silent no-op
    sent = []
    monkeypatch.setattr(alerts, "_post", lambda url, data, headers: sent.append((url, data, headers)))
    monkeypatch.setenv("NTFY_TOPIC", "eyris-test-topic")
    log = {"reason": "rebalance", "holdings_source": "portfolio_api", "last_bar": "x",
           "weights": {"AAPL": 0.1}, "inputs": {"params": {}}}
    title, msg, urgent = alerts.round_summary("UPLOADED", "official-2026-10-12-r1", log, {"status": "PENDING_SELECTION",
                                                                                           "team_token": "SECRET"})
    assert not urgent and "SECRET" not in msg and "AAPL" not in msg
    assert alerts.notify(title, msg, urgent) == ["ntfy"] and sent[0][0].endswith("/eyris-test-topic")
    assert alerts.round_summary("ERROR RuntimeError: x", "r", None)[2]
    assert alerts.round_summary("UPLOADED", "r", {**log, "holdings_source": "local_state"})[2]
    assert alerts.round_summary("HOLD_NO_UPLOAD", "r", {**log, "breaker": True})[2]
    monkeypatch.setattr(alerts, "_post", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    assert alerts.notify(title, msg) == []                   # network failure never raises
