"""Lookahead, constraint, failure, determinism and latency tests."""
import time

import numpy as np
import pandas as pd
import pytest

from eyris import alpha
from eyris.agent import Agent
from eyris.backtest import agent_targets, compute_metrics, simulate, target_policy
from eyris.config import MAX_WEIGHT, N_ASSETS, Params
from eyris.data import build_panels, build_rounds
from eyris.execution import HOLD, rebalance, sanitize

from conftest import synthetic_long

PARAM_SETS = [
    Params(risk_method="invvol", lookback_days=20),
    Params(risk_method="minvar", lookback_days=10, gross=0.7, lam=0.25, band=0.02),
    Params(risk_method="blend", lookback_days=40, gross=0.85, lam=1.0, band=0.0),
]


@pytest.fixture(scope="module")
def toy_model(synth):
    """A real LightGBM model fitted on synthetic data (exercises the alpha path)."""
    _, p, r = synth
    raw = alpha.raw_features(p.close, p.volume, p.high, p.low)
    ks = np.arange(300, len(r) - 7)
    X = alpha.cs_features(raw[r.info_end[ks]])
    y = alpha.labels(r.price)[ks]
    return alpha.AlphaModel.fit(X, y)


# --------------------------------------------------------------------------- lookahead
@pytest.mark.parametrize("params", PARAM_SETS + [Params(use_alpha=True, tilt=0.5, lookback_days=20)])
def test_corrupting_future_data_does_not_change_past_decisions(synth, toy_model, params):
    df, p, r = synth
    cut_time = p.times[int(r.exec_bar[len(r) - 60])]       # corrupt everything from here on
    bad = df.copy()
    fut = bad["timestamp_et"] >= cut_time
    rng = np.random.default_rng(1)
    for c in ("open", "high", "low", "close"):
        bad.loc[fut, c] *= rng.uniform(0.5, 1.5, fut.sum())
    bad.loc[fut, "volume"] = rng.integers(1, 1e8, fut.sum())
    p2 = build_panels(bad)
    r2 = build_rounds(p2)
    model = toy_model if params.use_alpha else None
    past = np.flatnonzero(p.times[r.info_end] < cut_time)
    past = past[past >= 300]
    assert len(past) > 50
    t1 = agent_targets(Agent(params, model), p, r, past)
    t2 = agent_targets(Agent(params, model), p2, r2, past)
    np.testing.assert_array_equal(t1, t2)
    # single-decision path, with all future bars physically present in the panel
    for k in past[-5:]:
        a = Agent(params, model).decide(p.head(int(r.info_end[k])), np.zeros(N_ASSETS))
        b = Agent(params, model).decide(p2.head(int(r2.info_end[k])), np.zeros(N_ASSETS))
        np.testing.assert_array_equal(a.weights, b.weights)


def test_batch_targets_match_single_decision_path(synth, toy_model):
    _, p, r = synth
    params = Params(use_alpha=True, tilt=0.5, lookback_days=20)
    ks = np.arange(300, len(r), 13)
    batch = agent_targets(Agent(params, toy_model), p, r, ks)
    for j, k in enumerate(ks):
        single = Agent(params, toy_model).target(p.head(int(r.info_end[k])))
        np.testing.assert_allclose(batch[j], sanitize(single), atol=2e-8)


def test_decision_uses_only_bars_before_execution(synth):
    _, p, r = synth
    assert (r.info_end < r.exec_bar).all()
    # Round 1 sees the full previous day, round r>=2 sees today's bars up to slot r-2.
    first = r.number == 1
    assert (p.slot[r.info_end[first]] == 6).all()
    assert (p.slot[r.info_end[~first]] == r.number[~first] - 2).all()


# --------------------------------------------------------------------------- constraints
@pytest.mark.parametrize("params", PARAM_SETS)
def test_weights_valid_on_every_decision(synth, params):
    _, p, r = synth
    ks = np.arange(300, len(r))
    agent = Agent(params)
    T = agent_targets(agent, p, r, ks)
    submitted = []

    def policy(k, w):
        out = rebalance(w, T[k - ks[0]], params.lam, params.band, params.min_trade)
        if out is not HOLD:
            submitted.append(out)
        return out

    res = simulate(r, int(ks[0]), int(ks[-1]) + 1, policy)
    assert submitted, "agent never traded"
    for w in submitted + list(T):
        assert w.shape == (N_ASSETS,)
        assert np.isfinite(w).all()
        assert (w >= 0).all() and (w <= MAX_WEIGHT).all()
        assert w.sum() <= 1.0
        assert (w <= params.stock_cap + 1e-12).all()
    assert (res.weights >= -1e-12).all()


def test_sanitize_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(500):
        w = sanitize(rng.uniform(-0.5, 0.8, N_ASSETS) * rng.uniform(0, 3))
        assert (w >= 0).all() and (w <= MAX_WEIGHT).all() and w.sum() <= 1.0


# --------------------------------------------------------------------------- failures
def test_nan_latest_price_holds(synth):
    _, p, r = synth
    head = p.head(int(r.info_end[-1]))
    head.close = head.close.copy()
    head.close[-1, 3] = np.nan
    d = Agent().decide(head, np.zeros(N_ASSETS))
    assert d.hold and "non-finite" in d.reason


def test_nan_current_weights_hold(synth):
    _, p, r = synth
    w = np.full(N_ASSETS, 0.02)
    w[0] = np.nan
    assert Agent().decide(p.head(int(r.info_end[-1])), w).hold


def test_wrong_shape_current_weights_hold(synth):
    _, p, r = synth
    assert Agent().decide(p.head(int(r.info_end[-1])), np.zeros(5)).hold


def test_short_history_holds(synth):
    _, p, _ = synth
    assert Agent(Params(lookback_days=40)).decide(p.head(3), np.zeros(N_ASSETS)).hold


def test_internal_error_holds(synth):
    _, p, r = synth

    class Broken:
        def predict(self, X):
            raise RuntimeError("boom")

    d = Agent(Params(use_alpha=True, tilt=0.3), Broken()).decide(p.head(int(r.info_end[-1])), np.zeros(N_ASSETS))
    assert d.hold and "boom" in d.reason


def test_missing_bars_are_forward_filled():
    df = synthetic_long(n_days=50)
    drop = (df["ticker"] == "AAPL") & (df["timestamp_et"].dt.day == 10)
    p = build_panels(df[~drop])
    assert np.isfinite(p.close).all()
    r = build_rounds(p)
    d = Agent().decide(p.head(int(r.info_end[-1])), np.zeros(N_ASSETS))
    assert not d.hold


def test_missing_symbol_in_live_snapshot_holds():
    from eyris.live import decide_round
    df = synthetic_long(n_days=50)
    d = decide_round(df[df["ticker"] != "MSFT"], np.zeros(N_ASSETS), Agent())
    assert d.hold


def test_inside_band_holds(synth):
    _, p, r = synth
    agent = Agent(Params(band=0.05))
    head = p.head(int(r.info_end[-1]))
    tgt = sanitize(agent.target(head))
    assert agent.decide(head, tgt).hold


# --------------------------------------------------------------------------- determinism / speed
def test_deterministic_and_fast(synth, toy_model):
    _, p, r = synth
    head = p.head(int(r.info_end[-1]))
    params = Params(risk_method="blend", use_alpha=True, tilt=0.5, lookback_days=40)
    t = time.perf_counter()
    a = Agent(params, toy_model).decide(head, np.zeros(N_ASSETS))
    elapsed = time.perf_counter() - t
    b = Agent(params, toy_model).decide(head, np.zeros(N_ASSETS))
    np.testing.assert_array_equal(a.weights, b.weights)
    assert elapsed < 1.0


# --------------------------------------------------------------------------- metrics
def test_metrics_match_starter_kit_example():
    # starter-kit/examples/evaluation.json -> evaluation_expected.json
    m = compute_metrics(np.array([1e6, 9e5]), np.array([9e5, 99e4]), np.array([3e5, 4.5e5]),
                        np.array([1e6, 9e5, 8e5, 99e4]))
    assert m["cumulative_return"] == pytest.approx(-0.01)
    assert m["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert m["maximum_drawdown"] == pytest.approx(0.2)
    assert m["turnover"] == pytest.approx(0.4)


def test_buy_and_hold_turnover_is_initial_allocation_only(synth):
    _, p, r = synth
    ew = np.full(N_ASSETS, 1 / N_ASSETS)
    k0 = int(np.searchsorted(r.day, 10))
    k1 = int(r.day_last_round[24]) + 1
    res = simulate(r, k0, k1, lambda k, w: ew if w.sum() < 1e-9 else HOLD)
    m = res.metrics()
    # notional = NAV / (1 + fee): the fee is paid from the same cash
    assert m["turnover"] == pytest.approx(1 / 1.001 / (k1 - k0), rel=1e-9)
    assert res.traded.sum() == 1
    assert len(res.valuations) == 1 + (k1 - k0) + 15 - 1  # init + endpoints + intermediate closes


def test_overlay_params_change_targets(synth):
    _, p, r = synth
    ks = np.arange(300, len(r), 7)
    base = agent_targets(Agent(Params(lookback_days=20, gross=0.7)), p, r, ks)
    for kw in (dict(halflife_days=5.0), dict(gross_mode="voltarget", vol_target=0.02),
               dict(gross_mode="trend", trend_days=5, trend_floor=0.0), dict(risk_method="hrp")):
        other = agent_targets(Agent(Params(lookback_days=20, gross=0.7, **kw)), p, r, ks)
        assert not np.allclose(base, other), kw
        assert (other.sum(1) <= 0.7 + 1e-9).all() and (other >= 0).all()


def test_study_target_cache_distinguishes_all_target_params():
    from eyris.experiment import Study
    st = Study.__new__(Study)
    st._targets = {}
    calls = []
    import eyris.experiment as ex
    orig = ex.agent_targets
    ex.agent_targets = lambda agent, *a: calls.append(agent.params) or np.zeros(1)
    try:
        st.p = st.r = st.ks = None
        st.targets(Params())
        st.targets(Params(lam=0.9, band=0.1))          # execution-only change: cached
        st.targets(Params(halflife_days=5.0))          # target change: recomputed
        st.targets(Params(gross_mode="trend"))
    finally:
        ex.agent_targets = orig
    assert len(calls) == 3


def test_factor_model_has_no_lookahead_and_matches_single_path():
    from eyris.factors import NAMES, FactorModel
    df = synthetic_long(n_days=90, seed=3)
    p = build_panels(df)
    r = build_rounds(p)
    model = FactorModel({n: 1.0 for n in NAMES})
    params = Params(use_alpha=True, tilt=0.5, lookback_days=20)
    cut_time = p.times[int(r.exec_bar[len(r) - 40])]
    bad = df.copy()
    fut = bad["timestamp_et"] >= cut_time
    bad.loc[fut, ["open", "high", "low", "close"]] *= 1.3
    p2 = build_panels(bad)
    r2 = build_rounds(p2)
    past = np.flatnonzero(p.times[r.info_end] < cut_time)
    past = past[past >= 50 * 7]
    t1 = agent_targets(Agent(params, model), p, r, past)
    t2 = agent_targets(Agent(params, model), p2, r2, past)
    np.testing.assert_array_equal(t1, t2)
    for j in range(0, len(past), 11):
        single = Agent(params, model).target(p.head(int(r.info_end[past[j]])))
        np.testing.assert_allclose(t1[j], sanitize(single), atol=2e-8)


def test_vol_power_and_top_k():
    from eyris.risk import inverse_vol
    rng = np.random.default_rng(1)
    vols = np.linspace(0.005, 0.03, N_ASSETS)
    r = rng.standard_normal((400, N_ASSETS)) * vols
    w1 = inverse_vol(r, 0.5, 0.30)
    w2 = inverse_vol(r, 0.5, 0.30, power=2.0)
    assert np.isclose(w1.sum(), 0.5) and np.isclose(w2.sum(), 0.5)
    assert w2[0] / w2[-1] > w1[0] / w1[-1]                 # stronger tilt to low vol
    wk = inverse_vol(r, 0.5, 0.10, top_k=10)
    assert (wk > 0).sum() == 10 and (wk[:10] > 0).all()    # the 10 lowest-vol names
    assert np.isclose(wk.sum(), 0.5) and wk.max() <= 0.10 + 1e-12
    with pytest.raises(ValueError):
        Params(gross=0.5, stock_cap=0.10, top_k=4)        # 4 x 10% cannot reach 50%


@pytest.mark.parametrize("method", ["erc", "sector_eq", "sector_invvol"])
def test_structural_risk_methods_respect_constraints(synth, method):
    _, p, _ = synth
    from eyris.risk import risk_weights
    w = risk_weights(p.close[:2000], Params(risk_method=method, lookback_days=20, gross=0.5))
    assert np.isfinite(w).all() and (w >= 0).all() and w.max() <= 0.10 + 1e-12
    assert np.isclose(w.sum(), 0.5)


def test_gross_multiplier_scales_target(synth):
    _, p, _ = synth
    head = p.head(len(p) - 1)
    a = Agent(Params(gross=0.5, lookback_days=20, lam=1.0, band=0.0, min_trade=0.0))
    base = a.decide(head, np.zeros(N_ASSETS))
    boosted = a.decide(head, np.zeros(N_ASSETS), gross_mult=1.4)
    assert np.isclose(boosted.weights.sum(), 1.4 * base.weights.sum(), atol=1e-6)
    assert a.decide(head, np.zeros(N_ASSETS), gross_mult=float("nan")).hold
