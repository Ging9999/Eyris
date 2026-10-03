"""News veto (offline, fake model) and earnings-event overlay tests."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from eyris import events, news
from eyris.agent import Agent
from eyris.config import N_ASSETS, UNIVERSE, Params
from eyris.execution import HOLD, apply_events

UNTIL = datetime(2026, 10, 13, 9, 7, tzinfo=news.ET_TZ)


def fake_fetch(items):
    return lambda until, since: ([x for x in items if since <= x["published"] < until], [])


class FakeClient:
    def __init__(self, payload=None, stop="end_turn", exc=None):
        self.payload, self.stop, self.exc, self.calls = payload, stop, exc, 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        assert kw["model"] == news.MODEL and "temperature" not in kw
        if self.exc:
            raise self.exc
        text = json.dumps(self.payload)
        return SimpleNamespace(model=kw["model"], stop_reason=self.stop, _request_id="req_test",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                               content=[SimpleNamespace(type="text", text=text)])


ITEMS = [
    {"symbol": "BA", "source": "Yahoo Finance", "published": UNTIL - timedelta(hours=2),
     "title": "FAA grounds fleet after incident", "text": ""},
    {"symbol": "KO", "source": "Yahoo Finance", "published": UNTIL + timedelta(minutes=1),
     "title": "published after the deadline", "text": ""},
]


def test_veto_maps_risk_to_reduce_only_trims(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "CACHE", tmp_path / "cache")
    client = FakeClient({"flags": [{"symbol": "BA", "risk": "severe", "reason": "grounding"},
                                   {"symbol": "JPM", "risk": "elevated", "reason": "x"},
                                   {"symbol": "KO", "risk": "none", "reason": "y"}]})
    trims, log = news.veto(UNTIL, log_dir=tmp_path, client=client, fetch=fake_fetch(ITEMS))
    assert trims == {UNIVERSE.index("BA"): 1.0, UNIVERSE.index("JPM"): 0.5}
    assert "published after the deadline" not in log["prompt"]       # no post-deadline info
    assert (tmp_path / "llm_log.json").exists()
    # cached: same prompt -> no second API call, same answer
    trims2, log2 = news.veto(UNTIL, log_dir=tmp_path, client=client, fetch=fake_fetch(ITEMS))
    assert client.calls == 1 and trims2 == trims and log2["meta"]["cache_hit"]


def test_veto_failures_mean_no_veto(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "CACHE", tmp_path / "cache")
    for client in (FakeClient(exc=RuntimeError("no network")), FakeClient({"flags": []}, stop="refusal")):
        trims, log = news.veto(UNTIL, client=client, fetch=fake_fetch(ITEMS))
        assert trims == {}
        monkeypatch.setattr(news, "CACHE", tmp_path / f"c{client.calls}")
    assert news.veto(UNTIL, fetch=fake_fetch([]))[0] == {}


def test_apply_events_is_reduce_only_and_restores():
    cur = np.full(N_ASSETS, 0.5 / N_ASSETS)
    tgt = cur.copy()
    out = apply_events(cur, HOLD, tgt, {3: 1.0, 4: 0.5}, set())
    assert out[3] == 0 and abs(out[4] - tgt[4] / 2) < 1e-8 and (out <= cur + 1e-12).all()
    back = apply_events(out, HOLD, tgt, {}, {3})
    assert abs(back[3] - tgt[3]) < 1e-8
    assert apply_events(cur, HOLD, tgt, {}, set()) is HOLD


def test_event_schedule_timing():
    cal = pd.DataFrame({"symbol": ["JPM", "TSLA"],
                        "announced_at": pd.to_datetime(["2026-10-13 08:00", "2026-10-21 16:00"])})
    tdays = events.trading_dates(pd.bdate_range("2026-09-01", "2026-09-30"))
    sched = events.EventSchedule(cal, tdays)
    p = Params(event_mode="trim_restore")
    jpm, tsla = UNIVERSE.index("JPM"), UNIVERSE.index("TSLA")
    assert sched.flags("2026-10-12", 6, p) == ({}, set())             # before the trim round
    assert sched.flags("2026-10-12", 7, p)[0] == {jpm: 1.0}           # day before a pre-open release
    assert sched.flags("2026-10-13", 1, p) == ({}, {jpm})             # restore at the open after
    assert sched.flags("2026-10-21", 7, p)[0] == {tsla: 1.0}          # after-close: trim same day
    assert sched.flags("2026-10-22", 1, p)[1] == {tsla}


def test_agent_decide_applies_trims(synth):
    _, p, r = synth
    head = p.head(int(r.info_end[-1]))
    agent = Agent(Params())
    tgt = agent.decide(head, np.zeros(N_ASSETS)).weights
    d = agent.decide(head, tgt, trim={0: 1.0})
    assert not d.hold and d.weights[0] == 0 and (d.weights <= tgt + 1e-9).all()


# --------------------------------------------------------------------------- VIX sentiment overlay
def _vix_frame():
    import pandas as pd
    idx = pd.bdate_range("2026-06-01", "2026-10-09")
    v = pd.DataFrame({"vix": 15.0, "vix3m": 17.0}, index=idx)
    v.loc["2026-10-07", "vix"] = 30.0          # fear on the 7th's close
    return v


def test_vix_signal_uses_only_closes_before_the_day():
    import pandas as pd
    from eyris import sentiment
    from eyris.config import Params
    q = Params(gross=0.5, vix_mode="level", vix_threshold=25, vix_boost=1.4)
    v = _vix_frame()
    assert sentiment.multiplier(v, "2026-10-07", q) == 1.0   # its own close is not known yet
    assert sentiment.multiplier(v, "2026-10-08", q) == 1.4   # known the next morning
    # same answer whether or not the feed already has a (partial) row for the day
    assert sentiment.multiplier(v[v.index < "2026-10-08"], "2026-10-08", q) == 1.4
    # corrupting the day itself and later days changes nothing
    bad = v.copy()
    bad.loc[bad.index >= "2026-10-08", "vix"] = 99.0
    assert sentiment.multiplier(bad, "2026-10-08", q) == 1.4
    assert sentiment.multiplier(v, "2026-10-08", Params()) == 1.0           # off by default


def test_vix_overlay_scales_target_and_fails_safe(tmp_path, monkeypatch):
    import pytest
    from eyris import live
    from eyris.config import Params
    monkeypatch.setattr(live, "SNAPSHOTS", tmp_path)
    q = Params(gross=0.5, vix_mode="level", vix_threshold=25, vix_boost=1.4)
    m, info = live.vix_multiplier(q, "official-2026-10-08-r1", fetch=_vix_frame)
    assert m == 1.4 and info["vix_prev_close"] == 30.0
    def boom():
        raise RuntimeError("network down")
    m, info = live.vix_multiplier(q, "official-2026-10-08-r1", fetch=boom)
    assert m == 1.0 and "vix_error" in info
    with pytest.raises(ValueError):
        Params(gross=0.8, vix_mode="level", vix_boost=1.4)          # 0.8 x 1.4 > 100%


# --------------------------------------------------------------------------- FinBERT shadow mode
def test_finbert_shadow_logs_scores_and_never_raises(tmp_path, monkeypatch):
    import json as _json
    from datetime import datetime, timedelta
    import numpy as _np
    from eyris import finbert
    until = datetime(2026, 10, 8, 9, 10)
    items = [{"symbol": "AAPL", "published": until - timedelta(hours=1), "source": "Yahoo Finance",
              "title": "Apple recalls devices", "text": ""},
             {"symbol": "AAPL", "published": until - timedelta(hours=2), "source": "Yahoo Finance",
              "title": "Apple beats estimates", "text": ""}]
    monkeypatch.setattr(finbert, "available", lambda: True)
    monkeypatch.setattr(finbert, "sentiment", lambda titles, batch=64: _np.array([-0.9, 0.7]))
    s = finbert.shadow(until, until - timedelta(hours=18), log_dir=tmp_path, collect=lambda u, s: (items, []))
    assert s["status"] == "ok: 2 headlines" and s["by_symbol"]["AAPL"]["n"] == 2
    assert s["by_symbol"]["AAPL"]["min"] == -0.9
    log = _json.loads((tmp_path / "finbert_log.json").read_text())
    assert [h["title"] for h in log["headlines"]] == ["Apple recalls devices", "Apple beats estimates"]

    def boom(u, s):
        raise RuntimeError("feed down")
    assert finbert.shadow(until, until, collect=boom)["status"].startswith("error")
    monkeypatch.setattr(finbert, "available", lambda: False)
    assert finbert.shadow(until, until)["status"].startswith("skipped")


def test_finbert_shadow_is_off_unless_enabled():
    from eyris import live
    from eyris.config import Params
    assert live.finbert_post_step(Params(), "official-2026-10-12-r1") is None
