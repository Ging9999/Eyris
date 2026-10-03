"""Post-phase review tools: the ledger replica, reconciliation, evidence, config changes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eyris import review
from eyris.backtest import compute_metrics, simulate
from eyris.config import N_ASSETS, UNIVERSE
from eyris.data import Rounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _phase(days=("2026-10-08", "2026-10-09"), seed=0):
    rng = np.random.default_rng(seed)
    rounds = []
    for d in days:
        for n, hh in enumerate((9, 10, 11, 12, 13, 14, 15), start=1):
            rounds.append({"id": f"validation-{d}-r{n}", "exec": pd.Timestamp(f"{d} {hh}:30", tz=review.ET),
                           "close": pd.Timestamp(f"{d} 16:00", tz=review.ET)})
    px = 100 * np.exp(np.cumsum(rng.standard_normal((len(rounds), N_ASSETS)) * 0.003, axis=0))
    day_close = {pd.Timestamp(d).date(): px[7 * i + 6] * 1.001 for i, d in enumerate(days)}
    return rounds, px, day_close


def test_replica_matches_the_backtest_simulator():
    rounds, px, day_close = _phase()
    w = np.full(N_ASSETS, 0.5 / N_ASSETS)
    weights = {rounds[0]["id"]: w, rounds[9]["id"]: w * 1.2}
    periods, points = review.replicate(rounds, weights, px, day_close)
    # the same thing through eyris.backtest.simulate (used by every backtest)
    days = np.repeat([0, 1], 7)
    r = Rounds(day=days, number=np.tile(np.arange(1, 8), 2), exec_bar=np.arange(14), info_end=np.arange(14) - 1,
               price=px, day_close=np.array(list(day_close.values())), day_last_round=np.array([6, 13]))
    pol = lambda k, cur: weights.get(rounds[k]["id"])  # noqa: E731
    sim = simulate(r, 0, 14, pol)
    assert np.allclose([p["nav_after_period"] for p in periods], sim.nav_end)
    assert np.allclose([p["traded_notional"] for p in periods], sim.notional)
    assert review.metrics_of(periods, points) == pytest.approx(sim.metrics())


def test_reconcile_passes_on_identical_and_fails_on_a_fee_difference():
    rounds, px, day_close = _phase()
    weights = {rounds[0]["id"]: np.full(N_ASSETS, 0.5 / N_ASSETS)}
    ours, _ = review.replicate(rounds, weights, px, day_close)
    theirs = {p["start_time"]: p for p in ours}
    assert review.reconcile(ours, theirs)[1]
    other, _ = review.replicate(rounds, weights, px, day_close, fee=0.002)   # organizer charging double
    rec, ok = review.reconcile(ours, {p["start_time"]: p for p in other})
    assert not ok and rec["nav_after_bps"].abs().max() > 2


def test_exec_prices_and_hourly_proxy_from_minute_bars():
    rounds, _, _ = _phase(days=("2026-10-08",))
    t = pd.date_range("2026-10-08 09:30", "2026-10-08 15:59", freq="1min", tz=review.ET)
    rows = [{"timestamp": ts, "ticker": s, "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i}
            for i, ts in enumerate(t) for s in UNIVERSE]
    minute = pd.DataFrame(rows)
    px = review.exec_prices(minute, rounds)
    assert px[0, 0] == 100 and px[1, 0] == 100 + 60      # 09:30 open; 10:30 is the 61st minute
    proxy = review.hourly_mid_proxy(minute, rounds)
    assert proxy[0, 0] == px[0, 0]                       # 09:30: both the open
    assert proxy[1, 0] == 0.5 * ((100 + 30) + (100.5 + 89))   # (10:00 open + 10:59 close) / 2


def test_news_veto_evidence_reads_reasons_and_counterfactual():
    rounds, px, day_close = _phase()
    log = {"news_status": "ok", "news_trims": {"AAPL": 1.0}, "target": {"AAPL": 0.02},
           "_llm": {"response": json.dumps({"flags": [{"symbol": "AAPL", "risk": "severe", "reason": "recall"}]})}}
    last = px[-1].copy()
    last[0] = px[3][0] * 0.9                              # AAPL fell 10% after the trim
    df, summ = review.news_veto_evidence({rounds[3]["id"]: log}, rounds, px, last)
    assert summ["trims"] == 1 and summ["answer_rate"] == 1.0
    row = df.iloc[0]
    assert row.reason == "recall" and row.pnl_avoided_bps == pytest.approx(1e4 * 0.02 * 0.1)


def test_set_config_validates_and_records_history(tmp_path):
    import set_config
    pf, hf = tmp_path / "params.json", tmp_path / "hist.json"
    pf.write_text(json.dumps({"params": {"gross": 0.5, "stock_cap": 0.1, "news_veto": True}}))
    assert set_config.apply({"news_veto": "off"}, "validation rule", pf, hf) == {"news_veto": (True, False)}
    assert json.loads(hf.read_text())[0]["note"] == "validation rule"
    with pytest.raises(KeyError):
        set_config.apply({"news_vetoo": "off"}, "", pf, hf)        # typo refused
    with pytest.raises(ValueError):
        set_config.apply({"gross": "1.5"}, "", pf, hf)             # invalid value refused
    assert json.loads(pf.read_text())["params"]["gross"] == 0.5   # nothing written on refusal


def test_paper_round_uses_its_own_ledger_and_never_touches_state(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from eyris import live, paper
    monkeypatch.setattr(live, "PRIVATE", tmp_path)
    monkeypatch.setattr(paper.alerts, "notify", lambda *a, **k: [])
    seen = {}

    def fake_prepare(phase, rid, as_of=None, holdings_fn=None, **k):
        seen["cur"] = holdings_fn(np.full(N_ASSETS, 100.0))
        w = {s: 0.5 / N_ASSETS for s in UNIVERSE}
        return tmp_path / "decision.json", {"round_id": rid, "reason": "rebalance", "last_bar": "x", "breaker": False,
                                            "weights": w, "prices": {s: 100.0 for s in UNIVERSE}}
    monkeypatch.setattr(live, "prepare", fake_prepare)
    assert paper.run_one("2026-10-05", 1) == "PAPER_TRADE"
    assert not seen["cur"].any()                                  # first paper round starts from cash
    assert paper.run_one("2026-10-05", 2) == "PAPER_TRADE"
    assert np.isclose(seen["cur"].sum(), 0.5)                     # second round sees the paper holdings
    assert (tmp_path / "paper_state.json").exists() and not (tmp_path / "state.json").exists()


def test_exec_prices_fill_a_missing_minute_from_the_next_one():
    rounds, _, _ = _phase(days=("2026-10-08",))
    t = pd.date_range("2026-10-08 09:30", "2026-10-08 15:59", freq="1min", tz=review.ET)
    rows = [{"timestamp": ts, "ticker": s, "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i}
            for i, ts in enumerate(t) for s in UNIVERSE
            if not (s == "GS" and ts == pd.Timestamp("2026-10-08 10:30", tz=review.ET))]   # Yahoo gap
    minute = pd.DataFrame(rows)
    px = review.exec_prices(minute, rounds)
    assert np.isfinite(px).all()
    assert px[1, UNIVERSE.index("GS")] == 100 + 61          # 10:31 open
    assert review.filled_prices(minute, rounds) == 1
