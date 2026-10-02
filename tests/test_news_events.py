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
