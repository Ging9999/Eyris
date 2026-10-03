"""Paper trading: the real live pipeline on a normal trading day, never uploading.

  python -m eyris.live paper --loop            # run today's remaining rounds on time, then the review
  python -m eyris.live paper --round 3         # one round now (smoke test)
  python -m eyris.live paper --loop --reset    # start the paper portfolio from cash
  python -m eyris.live paper --from 2026-09-21 --to 2026-10-02 --reset   # replay past days, then review

Each round runs 17 minutes before its deadline (08:53, 10:08 ... 15:08 ET) through
live.prepare, exactly like `run`. The differences: holdings come from a separate
paper ledger (private/paper_state.json), which assumes every decision executed, and
nothing is uploaded. private/state.json is never touched. Rounds are logged as
private/paper-<day>-r<n>/, so `replay` and scripts/validation_review.py --phase paper
work on them. Alerts are sent if configured, titled "PAPER". After round 7 the loop
waits until 16:20 ET, downloads the 1-minute bars, and writes private/review/paper/review.md.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np

from . import alerts, live
from .config import DEADLINES, UNIVERSE

LEAD = timedelta(minutes=17)
REVIEW_AT = (16, 20)


def state_file():
    return live.PRIVATE / "paper_state.json"


def round_id(day, n):
    return f"paper-{day}-r{n}"


def holdings(prices):
    """Paper weights now: the last paper decision drifted by prices; cash if none."""
    w = live.weights_from_state(prices, state_file())
    return np.zeros(len(UNIVERSE)) if w is None else w


def run_one(day, n, now=None, snapshot=None, notify=True):
    rid = round_id(day, n)
    deadline = live.deadline_of(rid)
    as_of = min(now or datetime.now(live.ET), deadline)
    status, log = "ERROR", None
    try:
        path, log = live.prepare("paper", rid, as_of=as_of, holdings_fn=holdings, snapshot=snapshot)
        if path is not None:
            st = {"round_id": rid, "weights": log["weights"], "prices": log["prices"]}
            state_file().write_text(json.dumps(st, indent=1))
            status = "PAPER_TRADE"
        else:
            status = "PAPER_HOLD"
        print(json.dumps({"status": status, **{k: log[k] for k in ("round_id", "reason", "last_bar", "breaker")}}))
    except Exception as e:  # a paper round must never stop the loop
        status = f"ERROR {type(e).__name__}: {str(e)[:200]}"
        print(json.dumps({"status": status, "round_id": rid}))
    if log is not None and now is None and snapshot is None:   # live paper rounds only, after the decision
        live.finbert_post_step(live.load_params(), rid)
    if notify:
        title, msg, _ = alerts.round_summary(status, rid, log)
        alerts.notify("PAPER " + title, msg, urgent=status.startswith("ERROR") or bool(log and log.get("breaker")))
    return status


def _sleep_until(t):
    while True:
        left = (t - datetime.now(live.ET)).total_seconds()
        if left <= 0:
            return
        time.sleep(min(left, 60))


def review():
    script = live.ROOT / "scripts" / "validation_review.py"
    for step in ("market", "analyze"):
        subprocess.run([sys.executable, str(script), step, "--phase", "paper"], cwd=live.ROOT, check=False)


def loop(day):
    for n in range(1, 8):
        deadline = live.deadline_of(round_id(day, n))
        now = datetime.now(live.ET)
        if now >= deadline - timedelta(minutes=2):
            print(f"r{n}: deadline {DEADLINES[n - 1]} ET already passed, skipped")
            continue
        start = deadline - LEAD
        if now < start:
            print(f"r{n}: waiting until {start:%H:%M} ET", flush=True)
            _sleep_until(start)
        run_one(day, n)
    _sleep_until(live.deadline_of(round_id(day, 7)).replace(hour=REVIEW_AT[0], minute=REVIEW_AT[1]))
    review()


def replay_days(start, end):
    """Replay past trading days round by round (one download, cut at each deadline), then review.

    Yahoo keeps ~60 days of 30-minute bars and the agent needs 40 days of history,
    so ``start`` can be at most ~2-3 weeks back.
    """
    import pandas as pd
    df30 = live.fetch_30m()
    live.SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    snap = live.SNAPSHOTS / f"paper-replay-{start}-{end}_30m.parquet"
    df30.to_parquet(snap)
    counts = {}
    for day in pd.bdate_range(start, end):
        for n in range(1, 8):
            s = run_one(day.date().isoformat(), n, snapshot=snap, notify=False)
            counts[s.split(":")[0]] = counts.get(s.split(":")[0], 0) + 1
    print(json.dumps(counts))
    review()


def main(a):
    day = a.day or datetime.now(live.ET).date().isoformat()
    if a.reset and state_file().exists():
        state_file().unlink()
        print("paper portfolio reset to cash")
    if datetime.fromisoformat(day).weekday() >= 5 and not (a.round or a.start):
        print(f"{day} is a weekend: no market session. Use --round N for a smoke test.")
        return 1
    if a.start:
        replay_days(a.start, a.end or day)
        return 0
    if a.round:
        run_one(day, a.round)
        return 0
    if a.loop:
        loop(day)
        return 0
    print("use --loop (whole day) or --round N (one round now)")
    return 1
