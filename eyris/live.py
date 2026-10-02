"""Live round pipeline: fetch public bars -> pure decision -> decision.json.

Network is used only in the fetch pre-step (Yahoo Finance via yfinance). The
decision itself (``decide_round``) is offline and deterministic given the
snapshot. HOLD means: do not upload anything this round (the backend keeps the
portfolio with no fee).

Usage (from the repo root):
    python -m eyris.live decide --phase validation --round-id validation-2026-10-08-r1
    python -m eyris.live run --phase validation   # decide + upload via the official kit
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .agent import Agent, Decision
from .alpha import AlphaModel
from .config import BAR_END, BAR_SLOTS, N_ASSETS, UNIVERSE, Params
from .data import EARLY_CLOSE_DAYS, EARLY_CLOSE_LAST_SLOT, build_panels

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "private"
SNAPSHOTS = ROOT / "data" / "live"
PARAMS_FILE = ROOT / "models" / "params.json"
MODEL_FILE = ROOT / "models" / "alpha_lgbm.txt"


# --------------------------------------------------------------------------- data
def fetch_30m(days=59):
    """Download 30-minute regular-session bars for the universe (network)."""
    import yfinance as yf
    raw = yf.download(list(UNIVERSE), period=f"{days}d", interval="30m", auto_adjust=False,
                      prepost=False, progress=False, threads=False, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")
    frames = []
    for field, name in (("Open", "open"), ("High", "high"), ("Low", "low"),
                        ("Close", "close"), ("Volume", "volume")):
        s = raw[field].stack(future_stack=True).rename(name)
        frames.append(s)
    df = pd.concat(frames, axis=1).reset_index()
    df.columns = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df["timestamp"] = ts.dt.tz_convert(ET).dt.tz_localize(None)
    return df.dropna(subset=["close"])


def resample_to_grid(df30, as_of):
    """30m bars -> historical grid (09:30 half bar, then clock hours).

    Only bars whose *end* is <= ``as_of`` are kept, so an in-progress bar can
    never leak into a decision.
    """
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert(ET).tz_localize(None)
    df = df30.copy()
    t = pd.to_datetime(df["timestamp"])
    hhmm = t.dt.strftime("%H:%M")
    df = df[(hhmm >= "09:30") & (hhmm < "16:00")]
    t = pd.to_datetime(df["timestamp"])
    start = t.dt.floor("h").where(t.dt.strftime("%H:%M") != "09:30", t)
    df = df.assign(bar=start)
    g = df.sort_values("timestamp").groupby(["ticker", "bar"], sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum")).reset_index()
    slot = out["bar"].dt.strftime("%H:%M").map({s: i for i, s in enumerate(BAR_SLOTS)})
    out = out[slot.notna()]
    slot = slot[slot.notna()].astype(int)
    end_hhmm = slot.map(dict(enumerate(BAR_END)))
    bar_end = pd.to_datetime(out["bar"].dt.strftime("%Y-%m-%d ") + end_hhmm)
    early = out["bar"].dt.normalize().isin(EARLY_CLOSE_DAYS)
    out = out[(bar_end <= as_of) & ~(early & (slot > EARLY_CLOSE_LAST_SLOT))]
    return out.rename(columns={"bar": "timestamp_et"})


# --------------------------------------------------------------------------- state
def load_params():
    if PARAMS_FILE.exists():
        return Params(**json.loads(PARAMS_FILE.read_text())["params"])
    return Params()


def load_agent():
    params = load_params()
    model = AlphaModel.load(MODEL_FILE) if params.use_alpha and MODEL_FILE.exists() else None
    if params.use_alpha and model is None:
        params = Params(**{**params.to_dict(), "use_alpha": False, "tilt": 0.0})
    return Agent(params, model)


def weights_from_portfolio(portfolio, prices):
    """Best-effort current weights from the organizer portfolio API response.

    The backend schema is not documented; this accepts common shapes
    (positions/holdings lists or symbol maps with shares or weights). Returns
    None when it cannot parse, so the caller can fall back to local state.
    """
    if not isinstance(portfolio, dict):
        return None
    body = portfolio.get("result", portfolio)
    body = body.get("portfolio", body) if isinstance(body, dict) else body
    if not isinstance(body, dict):
        return None
    nav = next((float(body[k]) for k in ("nav", "total_value", "portfolio_value", "equity", "value")
                if isinstance(body.get(k), (int, float, str)) and _num(body.get(k))), None)
    cash = next((float(body[k]) for k in ("cash", "cash_balance") if _num(body.get(k))), None)
    holdings = None
    for k in ("positions", "holdings", "shares", "weights"):
        if k in body:
            holdings = (k, body[k])
            break
    if holdings is None:
        return np.zeros(N_ASSETS) if cash is not None and nav is not None and abs(cash - nav) < 1e-6 else None
    kind, h = holdings
    items = {}
    if isinstance(h, dict):
        items = {s: v for s, v in h.items()}
    elif isinstance(h, list):
        for row in h:
            if isinstance(row, dict):
                sym = row.get("symbol") or row.get("ticker")
                val = next((row[k] for k in ("weight", "shares", "quantity", "qty") if k in row), None)
                if sym is not None and val is not None:
                    items[sym] = val if not any(k in row for k in ("weight",)) else {"weight": row["weight"]}
    if not items:
        return None
    w = np.zeros(N_ASSETS)
    use_weight = kind == "weights" or any(isinstance(v, dict) for v in items.values())
    if use_weight:
        for i, s in enumerate(UNIVERSE):
            v = items.get(s, 0)
            w[i] = float(v["weight"] if isinstance(v, dict) else v)
        return w
    shares = np.array([float(items.get(s, 0) or 0) for s in UNIVERSE])
    value = shares * prices
    total = (cash if cash is not None else 0.0) + value.sum() if nav is None else nav
    return value / total if total > 0 else None


def _num(x):
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def weights_from_state(prices):
    """Fallback: last submitted weights drifted by price moves since then."""
    path = PRIVATE / "state.json"
    if not path.exists():
        return None
    st = json.loads(path.read_text())
    w = np.array([st["weights"][s] for s in UNIVERSE], dtype=float)
    p0 = np.array([st["prices"][s] for s in UNIVERSE], dtype=float)
    grown = w * prices / p0
    return grown / (1.0 - w.sum() + grown.sum())


def save_state(decision: Decision, prices, round_id):
    PRIVATE.mkdir(exist_ok=True)
    st = {"round_id": round_id, "weights": decision.as_dict(),
          "prices": {s: float(x) for s, x in zip(UNIVERSE, prices)}}
    (PRIVATE / "state.json").write_text(json.dumps(st, indent=1))


# --------------------------------------------------------------------------- decision
def decide_round(bars_grid, current_weights, agent=None):
    """Pure decision from a completed-bar snapshot. Never raises."""
    agent = agent or load_agent()
    try:
        panels = build_panels(bars_grid)
    except Exception as e:
        return Decision(None, reason=f"error: bad snapshot: {e}")
    return agent.decide(panels, current_weights)


def decision_payload(decision, phase, round_id):
    return {"submission_type": "decision", "team_id": os.environ.get("TEAM_ID", "REPLACE_TEAM_ID"),
            "team_token": os.environ.get("TEAM_TOKEN", "REPLACE_TEAM_TOKEN"), "phase": phase,
            "round_id": round_id, "weights": {s: round(float(w), 8) for s, w in decision.as_dict().items()}}


def deadline_of(round_id):
    """Submission deadline (aware ET) from a round id like official-2026-10-12-r3."""
    from .config import DEADLINES
    day, n = round_id.split("-", 1)[1].rsplit("-r", 1)
    hh, mm = DEADLINES[int(n) - 1].split(":")
    return datetime.fromisoformat(day).replace(hour=int(hh), minute=int(mm), tzinfo=ET)


def _shape(x, depth=0):
    """Keys/types of an API response without any values (safe to log)."""
    if depth > 3:
        return "..."
    if isinstance(x, dict):
        return {k: _shape(v, depth + 1) for k, v in list(x.items())[:40]}
    if isinstance(x, list):
        return [_shape(x[0], depth + 1)] if x else []
    return type(x).__name__


def prepare(phase, round_id, portfolio=None, as_of=None, snapshot=None, first_round=None):
    """Fetch (or load) bars, decide, write private/<round_id>/decision.json if trading."""
    as_of = as_of or min(datetime.now(ET), deadline_of(round_id))
    out_dir = PRIVATE / round_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if snapshot is None:
        df30 = fetch_30m()
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        snap_path = SNAPSHOTS / f"{round_id}_30m.parquet"
        df30.to_parquet(snap_path)
    else:
        df30 = pd.read_parquet(snapshot)
    grid = resample_to_grid(df30, as_of)
    last = grid.sort_values("timestamp_et").groupby("ticker").last()
    prices = last.reindex(list(UNIVERSE))["close"].to_numpy(dtype=float)
    cur = weights_from_portfolio(portfolio, prices) if portfolio is not None else None
    source = "portfolio_api"
    if cur is None:
        cur, source = weights_from_state(prices), "local_state"
    # Only the phase's very first round may assume the USD 1M starting cash.
    if first_round is None:
        first_round = round_id in ("validation-2026-10-08-r1", "official-2026-10-12-r1")
    if cur is None and first_round:
        cur, source = np.zeros(N_ASSETS), "assumed_initial_cash"
    if cur is None:
        source = "none"
        decision = Decision(None, reason="unknown current holdings")
    else:
        decision = decide_round(grid, cur)
    log = {"round_id": round_id, "as_of": str(as_of), "holdings_source": source,
           "portfolio_shape": _shape(portfolio),
           "last_bar": str(grid["timestamp_et"].max()), "hold": decision.hold, "reason": decision.reason,
           "target": None if decision.target is None else dict(zip(UNIVERSE, map(float, decision.target))),
           "weights": decision.as_dict()}
    (out_dir / "decision_log.json").write_text(json.dumps(log, indent=1))
    if decision.hold:
        return None, log
    path = out_dir / "decision.json"
    path.write_text(json.dumps(decision_payload(decision, phase, round_id), indent=2))
    save_state(decision, prices, round_id)
    return path, log


# --------------------------------------------------------------------------- CLI
SCHEDULE_URL = "https://hackathon2.deepintomlf.ai/extensions/icaif2026/99/backend/api/v1/schedule"


def public_schedule():
    """Organizer schedule (public endpoint, no credentials)."""
    import urllib.request
    with urllib.request.urlopen(SCHEDULE_URL, timeout=30) as resp:
        return json.loads(resp.read())


def open_round(schedule):
    """The round whose window is open at the organizer's clock, else None."""
    now = pd.Timestamp(schedule["current_time"])
    for row in schedule["rounds"]:
        if row.get("status") != "CANCELLED" and \
                pd.Timestamp(row["opens_at"]) <= now < pd.Timestamp(row["deadline"]):
            return row
    return None


def _kit_session():
    kit = Path(os.environ.get("ICAIF_KIT", ROOT / "starter-kit"))
    sys.path.insert(0, str(kit))
    os.chdir(kit)
    os.environ.setdefault("ICAIF_PROFILE", str(kit / "profiles" / "profile99-production.json"))
    from kit.config import load_environment
    from kit.original_client import OriginalSession, load_profile
    load_environment()
    session = OriginalSession(profile=load_profile(os.environ["ICAIF_PROFILE"]),
                              token=os.environ["CODABENCH_TOKEN"],
                              checkpoint=".icaif/checkpoint.json", credentials=".icaif/credentials.json")
    if session.creds is None:  # fresh container: import organizer-issued credentials from env
        session.save_credentials(os.environ["TEAM_ID"], os.environ["TEAM_TOKEN"])
    return session


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="write decision.json for a round (no upload)")
    d.add_argument("--phase", required=True, choices=["validation", "official"])
    d.add_argument("--round-id", required=True)
    d.add_argument("--snapshot", help="use a saved 30m parquet instead of fetching")
    d.add_argument("--as-of", help="ISO time; defaults to now (capped at the deadline)")
    run = sub.add_parser("run", help="decide and upload the currently open round via the kit")
    run.add_argument("--phase", choices=["validation", "official"], help="only act in this phase")
    run.add_argument("--dry-run", action="store_true", help="decide but never upload")
    a = ap.parse_args(argv)
    if a.cmd == "decide":
        as_of = datetime.fromisoformat(a.as_of) if a.as_of else None
        path, log = prepare(a.phase, a.round_id, as_of=as_of, snapshot=a.snapshot)
        print(json.dumps({k: log[k] for k in ("round_id", "hold", "reason", "holdings_source", "last_bar")}))
        print(path or "HOLD: nothing to upload this round")
        return 0
    sched = public_schedule()
    row = open_round(sched)
    if row is None or (a.phase and row["phase"] != a.phase):
        print(json.dumps({"status": "NO_OPEN_ROUND", "server_time": sched.get("current_time"),
                          "next_deadline": sched.get("next_deadline")}))
        return 0
    phase = row["phase"]
    first = min((r for r in sched["rounds"] if r["phase"] == phase and r.get("status") != "CANCELLED"),
                key=lambda r: r["opens_at"])["id"] == row["id"]
    have_creds = all(os.environ.get(k) for k in ("CODABENCH_TOKEN", "TEAM_ID", "TEAM_TOKEN"))
    if not have_creds and not a.dry_run:
        print(json.dumps({"status": "MISSING_CREDENTIALS", "round_id": row["id"],
                          "need": ["CODABENCH_TOKEN", "TEAM_ID", "TEAM_TOKEN"]}))
        return 2
    if a.dry_run and not have_creds:
        path, log = prepare(phase, row["id"], first_round=first)
        print(json.dumps({"status": "DRY_RUN", **{k: log[k] for k in ("round_id", "hold", "reason",
                                                                      "holdings_source", "last_bar")}}))
        return 0
    with _kit_session() as client:
        own = client.round(row["id"])
        if client._occupied(own):
            print(json.dumps({"status": "SLOT_ALREADY_USED", "round_id": row["id"]}))
            return 0
        portfolio = client.portfolio(phase)
        path, log = prepare(phase, row["id"], portfolio=portfolio, first_round=first)
        print(json.dumps({k: log[k] for k in ("round_id", "hold", "reason", "holdings_source",
                                              "last_bar", "portfolio_shape")}))
        if path is None:
            print(json.dumps({"status": "HOLD_NO_UPLOAD", "round_id": row["id"]}))
        elif a.dry_run:
            print(json.dumps({"status": "DRY_RUN", "decision": str(path)}))
        else:
            receipt = client.decision(str(path))
            print(json.dumps({"status": "UPLOADED", "receipt": receipt}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
