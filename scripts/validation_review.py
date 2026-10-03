"""Review a live phase: plumbing, simulator reconciliation, decision evidence.

  python scripts/validation_review.py fetch   --phase validation   # organizer API, read-only (needs credentials)
  python scripts/validation_review.py market  --phase validation   # Yahoo 1-minute bars + 16:00 closes + VIX
  python scripts/validation_review.py analyze --phase validation   # offline report
  python scripts/validation_review.py all     --phase validation

Run `market` within ~25 days of the phase: Yahoo keeps 1-minute bars for about 30 days.
Everything is saved under private/review/<phase>/ (gitignored). The report is
private/review/<phase>/review.md, and the pre-registered rules it checks are in
eyris/review.py:DECISION_RULES. During Official, the same tool tracks our own
metrics (the organizer hides Official metrics until trading ends) from our
replica of the ledger, once Validation has shown the replica matches.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris import live, review  # noqa: E402
from eyris.config import UNIVERSE  # noqa: E402


def out_dir(phase):
    d = live.PRIVATE / "review" / phase
    d.mkdir(parents=True, exist_ok=True)
    return d


def paper_schedule():
    """Paper rounds have no organizer schedule: rebuild it from the logged paper days."""
    from eyris.config import EXEC_TIMES
    days = sorted({d.name.split("-", 1)[1].rsplit("-r", 1)[0] for d in live.PRIVATE.glob("paper-*-r*")})
    rounds = []
    for day in days:
        for n, t in enumerate(EXEC_TIMES, start=1):
            ex = pd.Timestamp(f"{day} {t}", tz=review.ET)
            rounds.append({"id": f"paper-{day}-r{n}", "phase": "paper", "status": "SCHEDULED",
                           "execution_time": ex.isoformat(),
                           "close_time": pd.Timestamp(f"{day} 16:00", tz=review.ET).isoformat()})
    return {"rounds": rounds}


def schedule(phase=None):
    if phase == "paper":
        return paper_schedule()
    try:
        return live.public_schedule()
    except Exception as e:
        print(f"live schedule unavailable ({e}); using the bundled starter-kit/schedule.json")
        return json.loads((ROOT / "starter-kit" / "schedule.json").read_text())


def cmd_fetch(phase):
    if phase == "paper":
        print("paper phase: nothing to fetch from the organizer")
        return
    d = out_dir(phase)
    (d / "schedule.json").write_text(json.dumps(schedule(), indent=1))
    with live._kit_session() as client:
        for name, call in (("portfolio", client.portfolio), ("ledger", client.ledger),
                           ("decisions", client.decisions), ("metrics", client.metrics)):
            try:
                body = call(phase)
                (d / f"api_{name}.json").write_text(json.dumps(body, indent=1, default=str))
                print(f"{name}: saved ({live._shape(body) if not isinstance(body, dict) else list(body)[:12]})")
            except Exception as e:
                print(f"{name}: {type(e).__name__}: {str(e)[:200]}")


def cmd_market(phase):
    import yfinance as yf
    d = out_dir(phase)
    sched = schedule("paper") if phase == "paper" else (
        json.loads((d / "schedule.json").read_text()) if (d / "schedule.json").exists() else schedule())
    rounds = review.phase_rounds(sched, phase, until=datetime.now(timezone.utc))
    if not rounds:
        print("no executed rounds yet")
        return
    first, last = rounds[0]["exec"].date(), rounds[-1]["exec"].date()
    frames = []
    day = first
    while day <= last:                      # Yahoo serves 1-minute bars in chunks of <= 7 days
        end = min(day + timedelta(days=7), last + timedelta(days=1))
        raw = yf.download(list(UNIVERSE), start=str(day), end=str(end), interval="1m", auto_adjust=False,
                          prepost=False, progress=False, group_by="column")
        if len(raw):
            long = raw[["Open", "High", "Low", "Close"]].stack(level=1, future_stack=True).reset_index()
            long.columns = ["timestamp", "ticker", "open", "high", "low", "close"]
            frames.append(long)
        day = end
    minute = pd.concat(frames, ignore_index=True).dropna(subset=["open"])
    minute.to_parquet(d / "minute.parquet")
    daily = yf.download(list(UNIVERSE), start=str(first), end=str(last + timedelta(days=1)), interval="1d",
                        auto_adjust=False, progress=False)["Close"].reindex(columns=list(UNIVERSE))
    daily.to_csv(d / "closes.csv")
    live.fetch_vix().to_csv(d / "vix.csv")
    print(f"saved {len(minute):,} 1-minute rows for {minute.ticker.nunique()} tickers, {len(daily)} daily closes")


def _receipts(decisions_payload):
    """{round_id: status} from the decisions API, whatever its exact shape."""
    items = decisions_payload
    if isinstance(items, dict):
        items = next((v for v in items.values() if isinstance(v, list)), [])
    out = {}
    for it in items or []:
        if isinstance(it, dict) and it.get("round_id"):
            out[it["round_id"]] = it.get("status") or it.get("selection_status") or it.get("execution_status")
    return out


def cmd_analyze(phase):
    d = out_dir(phase)
    sched = schedule("paper") if phase == "paper" else (
        json.loads((d / "schedule.json").read_text()) if (d / "schedule.json").exists() else schedule())
    rounds = review.phase_rounds(sched, phase, until=datetime.now(timezone.utc))
    logs = review.load_round_logs(live.PRIVATE, phase)
    replay = live.replay_all([r["id"] for r in rounds])
    api = {n: json.loads((d / f"api_{n}.json").read_text()) for n in ("metrics", "decisions", "portfolio")
           if (d / f"api_{n}.json").exists()}
    receipts = _receipts(api.get("decisions"))
    ops = review.operations(logs, rounds, replay, receipts)
    report = {"phase": phase, "generated": datetime.now(timezone.utc).isoformat(), "rules": review.DECISION_RULES,
              "rounds_scheduled": len(rounds), "rounds_logged": int(ops["logged"].sum()) if len(ops) else 0}
    lines = [f"# {phase.title()} review", "", f"Generated {report['generated']}. "
             f"{report['rounds_logged']}/{len(rounds)} executed rounds have a local decision log.", ""]

    # ---- plumbing
    checks = {}
    logged = ops[ops["logged"]] if len(ops) else ops
    if len(logged):
        checks["holdings from organizer API"] = bool(logged["holdings_ok"].all())
        checks["replay matches every round"] = bool(logged["replay"].fillna("").str.startswith(("MATCH", "OK_")).all())
        checks["no circuit-breaker holds"] = not bool(logged["breaker"].any())
    if receipts:
        bad = {k: v for k, v in receipts.items() if str(v).upper() in ("INVALID", "LATE", "REJECTED", "MISSED_DEADLINE")}
        checks["no invalid/late receipts"] = not bad
    lines += ["## 1. Plumbing", "", *[f"- [{'x' if ok else ' '}] {name}" for name, ok in checks.items()], "",
              review.md_table(ops.drop(columns=["logged"], errors="ignore")), ""]
    report["plumbing"] = checks
    report["operations"] = ops.to_dict("records")

    # ---- simulator reconciliation and our metrics
    minute_f, closes_f = d / "minute.parquet", d / "closes.csv"
    if minute_f.exists() and closes_f.exists() and rounds:
        minute = pd.read_parquet(minute_f)
        closes = pd.read_csv(closes_f, index_col=0, parse_dates=True)
        day_close = {ts.date(): row.to_numpy(dtype=float) for ts, row in closes.iterrows()}
        px = review.exec_prices(minute, rounds)
        weights = {}
        for r in rounds:
            f = live.PRIVATE / r["id"] / "decision.json"
            if f.exists() and r["id"] in logs and not logs[r["id"]]["hold"]:
                w = json.loads(f.read_text())["weights"]
                weights[r["id"]] = np.array([w[s] for s in UNIVERSE])
        ours, points = review.replicate(rounds, weights, px, day_close)
        m = review.metrics_of(ours, points)
        report["our_metrics_replica"] = m
        lines += ["## 2. Our metrics (replica of the organizer ledger at real 1-minute fills)", "",
                  *[f"- {k}: {v:.6g}" for k, v in m.items()], ""]
        if "metrics" in api:
            theirs = review.organizer_periods(api["metrics"])
            rec, ok = review.reconcile(ours, theirs)
            report["reconciliation_ok"] = ok
            report["reconciliation"] = rec.to_dict("records")
            lines += [f"Reconciliation vs the organizer metrics API: **{'PASS' if ok else 'FAIL'}** "
                      "(rule: NAV within 2 bps, notional within 1%).", "",
                      review.md_table(rec), ""]
        proxy = review.hourly_mid_proxy(minute, rounds)
        signed = (proxy / px - 1) * 1e4
        hh30 = [k for k, r in enumerate(rounds) if r["exec"].hour != 9]
        g = signed[hh30]
        report["fill_proxy_gap_bps"] = ({"median_abs": float(np.nanmedian(np.abs(g))),
                                         "p90_abs": float(np.nanpercentile(np.abs(g), 90)),
                                         "mean_signed": float(np.nanmean(g)),
                                         "t_signed": float(np.nanmean(g) / np.nanstd(g) * np.sqrt(np.isfinite(g).sum()))}
                                        if hh30 else {})
        fp = report["fill_proxy_gap_bps"]
        lines += ["## 3. Backtest fill proxy vs real 1-minute open (rounds 2-7)", "",
                  *((f"- |gap| median {fp['median_abs']:.1f} bps, 90th percentile {fp['p90_abs']:.1f} bps per name-round",
                     f"- signed mean {fp['mean_signed']:+.2f} bps (t = {fp['t_signed']:.1f}); near 0 means noise, "
                     "not a bias in the backtests") if fp else ("- no HH:30 rounds yet",)), ""]

        # ---- news veto evidence
        last_close = closes.iloc[-1].to_numpy(dtype=float)
        nv, nsum = review.news_veto_evidence(logs, rounds, px, last_close)
        report["news_veto"] = {**nsum, "trims": nv.to_dict("records")}
        lines += ["## 4. News veto", "", f"- rounds asked {nsum['rounds_asked']}, answer rate {nsum['answer_rate']}, "
                  f"trims {nsum['trims']} ({nsum['trims_per_day']:.2f}/day)", "",
                  review.md_table(nv), ""]
    else:
        lines += ["## 2-4. Market data missing", "", "Run `python scripts/validation_review.py market` first.", ""]

    # ---- VIX
    if (d / "vix.csv").exists() and rounds:
        from eyris import sentiment
        vix = pd.read_csv(d / "vix.csv", index_col=0, parse_dates=True)
        days = sorted({r["exec"].date() for r in rounds})
        sig = sentiment.signals(vix, [pd.Timestamp(x) for x in days])
        report["vix_by_day"] = {str(k.date()): float(v) for k, v in sig.vix.items()}
        lines += ["## 5. VIX (previous close per day; the overlay would trigger above 25)", "",
                  *[f"- {k}: {v:.2f}" for k, v in report["vix_by_day"].items()], ""]

    lines += ["## Pre-registered decision rules", "", *[f"- **{k}**: {v}" for k, v in review.DECISION_RULES.items()]]
    (d / "review.md").write_text("\n".join(lines), encoding="utf-8")
    (d / "review.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n".join(lines[:40]))
    print(f"\nfull report: {d / 'review.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "market", "analyze", "all"])
    ap.add_argument("--phase", default="validation", choices=["validation", "official", "paper"])
    a = ap.parse_args()
    if a.cmd in ("fetch", "all"):
        cmd_fetch(a.phase)
    if a.cmd in ("market", "all"):
        cmd_market(a.phase)
    if a.cmd in ("analyze", "all"):
        cmd_analyze(a.phase)


if __name__ == "__main__":
    main()
