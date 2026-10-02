# Eyris: ICAIF 2026 Trading Agent

A low-turnover, risk-based portfolio agent for the
[ACM ICAIF 2026 Trading Agent Competition](https://hackathon2.deepintomlf.ai/competitions/99/)
(Codabench competition 99).

The decision path is pure Python/NumPy with no LLM, no network and no randomness. It reads
only completed bars, and on any error it **holds**, which means it uploads nothing and the
backend keeps the existing portfolio with no fee.

**Live configuration** (`models/params.json`):
- inverse-volatility weights from 40 days of hourly returns, capped at 10% per stock
- **50% gross exposure** (the rest is cash)
- partial rebalancing with λ = 0.25, a 0.05 L1 no-trade band, and per-name trades under 0.2% dropped
- the alpha tilt is **off**

In practice the agent buys the inverse-vol portfolio and rarely trades after that.

| Rolling 15-day windows (start from cash) | Agent median rank | Best simple baseline | Agent worst 15d MDD | EW buy & hold worst 15d MDD |
|---|---:|---:|---:|---:|
| Train 2021–24 | **6.50** / 24 | 8.50 (EW B&H) | 6.2% | 13.2% |
| Validation 2025-01..09 | **6.50** / 24 | 8.12 (EW B&H) | 7.9% | 16.7% |
| Holdout 2025-Q4 (evaluated once) | **6.50** / 24 | 8.00 (cash) | 2.1% | 4.8% |

Full tables, charts and caveats are in [`reports/backtest_report.md`](reports/backtest_report.md).
The alpha experiment is in [`reports/alpha_experiment.md`](reports/alpha_experiment.md). Research-backed upgrades (vol targeting,
trend overlay, HRP, EWMA risk, the overnight effect) were tested and not adopted; see
[`reports/research_experiment.md`](reports/research_experiment.md). Recent LLM-agent research (2024-26) was applied through
AlphaAgent-style offline factor mining and an LLM-competitor field; see [`reports/agentic_research.md`](reports/agentic_research.md).

The trade-off is explicit: over all of 2025 the agent returned +8.6% (Sharpe 1.24, MDD 9.9%) against +19.0% (Sharpe 1.20,
MDD 20.9%) for equal-weight buy & hold. The competition ranks drawdown and turnover as heavily as return,
and on that score the low-exposure, low-turnover profile ranks better in every period. Gross 0.70–0.85 is
statistically tied with 0.50 on median rank. 0.50 won only on worst-case windows, so raising `gross` is a reasonable
judgment call if you expect most rival teams to run high-drawdown portfolios.

## Competition facts this design relies on

Taken from the Codabench pages and the official starter kit, verified 2026-10-01.

| Topic | Rule |
|---|---|
| Submission | **One `decision.json` upload per round** (not code): `submission_type`, `team_id`, `team_token`, `phase`, `round_id`, all 30 `weights`. First in-window upload is final, even if invalid. |
| Weights | Target weights, finite numbers in `[0, 0.30]` per stock, sum `<= 1`; the remainder is cash. Long-only; fractional shares allowed. |
| Rounds | 7 per day. Deadlines are 09:10, 10:25, 11:25, …, 15:25 ET. Execution is at 09:30 (open), then 10:30, …, 15:30. Valued at the 16:00 close with no trade. Positions carry overnight. |
| Missing or invalid round | No rebalance, **no fee, zero turnover**; the portfolio is held. |
| Costs | 0.1% of buy+sell notional, including the initial allocation from cash. |
| Phases | Live Validation Oct 8–9 (14 rounds, not ranked). Official Oct 12–30 (15 days, 105 rounds). Final materials due Nov 3. |
| Score | Teams are ranked separately on cumulative return (↑), Sharpe (↑, per-round returns × √1764), max drawdown (↓, includes 16:00 closes) and turnover (↓, mean traded notional / NAV). The **overall score is the mean of the four ranks**. |
| Data | Hourly OHLCV for the 30 stocks, 2021-01-04 → **2025-12-31** (no Jan 2026). Bars are clock-aligned: 09:30–10:00, then 10:00–11:00, …, 15:00–16:00. |
| Live data | Not provided. Teams source public data themselves (we use Yahoo via `yfinance`). |
| Runtime | No runtime limit applies to the agent itself: decisions are generated locally. |

Three consequences shaped the design:

1. Two of the four ranks (drawdown, turnover) reward doing little. Gross
   exposure (the cash sleeve) is therefore a tuned parameter.
2. Submitting the current weights still trades, because prices move between
   the deadline and execution. A no-trade decision is best expressed by **not uploading**.
3. The kit's `watch` loop aborts on a strategy exception. So we run one round
   at a time with our own runner (`python -m eyris.live run`), which simply
   skips the upload on HOLD.

## Layout

```
eyris/
  config.py      universe, official constants, Params
  data.py        load/clean bars, early closes, decision rounds + execution prices
  risk.py        inverse-vol, min-variance, HRP, EWMA risk; vol-target / trend exposure overlays
  alpha.py       optional LightGBM cross-sectional ranker + multiplicative tilt
  execution.py   partial rebalancing w += lam*(target - w), no-trade band, sanitize
  agent.py       Agent.decide(bars_until_cutoff, current_weights) -> weights | HOLD
  backtest.py    share-level simulator, official metrics, baselines, rank score
  experiment.py  periods, cached reference field, config evaluation
  live.py        yfinance pre-step, 30m -> historical grid, decision.json, runner
scripts/
  download_data.py   organizer dataset + official starter kit
  tune.py            grid search on validation windows
  alpha_experiment.py
  report.py          reports/backtest_report.md + charts
tests/               lookahead, constraints, failure modes, live resampling, metrics
models/params.json   the selected configuration used live
reports/             generated results
```

## Setup

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # data/hourly_market_data_2021_2026.parquet + starter-kit/
```

## Test

```sh
python -m pytest -q
```

- **Lookahead:** the history is corrupted from a cut-off onward and rebuilt.
  Every decision made before the cut-off must be bit-identical, on both the
  batch path and the single-decision path, with and without alpha.
- **Constraints:** every target and every submitted weight vector is finite,
  ≥ 0, ≤ the cap and sums to ≤ 1.
- **Failures:** NaN prices, NaN or wrong-shape holdings, short history, a
  missing symbol and model exceptions all return HOLD.
- **Other checks:** determinism, under 1s per decision, metric parity with the
  kit's worked example, early-close round cancellation, and that the live
  resampler never uses an in-progress bar.

## Reproduce the research

```sh
python scripts/tune.py              # ~30 min; reports/tuning_*.csv
python scripts/alpha_experiment.py  # reports/alpha_experiment.json
python scripts/report.py            # reports/backtest_report.md (+ holdout, evaluated once)
```

## Submit (Validation Oct 8–9, Official Oct 12–30)

1. **Register** with the kit before **Oct 8 00:00 ET** to enter Validation:
   `cd starter-kit && python tools/auto_submit.py register --file private/register.json`.
   Store `TEAM_ID` / `TEAM_TOKEN`. Keep `.env`, `.icaif/` and `private/` out of git (already ignored).
2. Configure `starter-kit/.env`: `CODABENCH_TOKEN`, `ICAIF_PROFILE=profiles/profile99-production.json`,
   `TEAM_ID`, `TEAM_TOKEN`.
3. **Each round**, about 3 minutes before the deadline (09:07, 10:22, 11:22, … ET):

   ```sh
   export TEAM_ID=... TEAM_TOKEN=...
   python -m eyris.live run --phase validation      # or --phase official
   ```

   This reads the live schedule and your portfolio from the organizer API, then
   fetches 30m bars from Yahoo. It keeps only bars that have **closed** and
   resamples them onto the historical grid. The decision is made offline, and
   `private/<round_id>/decision.json` is uploaded through the kit, unless the
   agent holds, in which case nothing is uploaded. Every round writes
   `private/<round_id>/decision_log.json`.

   For a manual flow, run
   `python -m eyris.live decide --phase official --round-id official-2026-10-12-r1`,
   then check the result with
   `python starter-kit/tools/validate_submission.py private/<round_id>/decision.json --schedule starter-kit/schedule.json`
   and upload the file on the Codabench "My Submissions" page.
4. Use **Validation (Oct 8–9)** to check what this repo could not test offline:
   - the organizer portfolio JSON parses (`holdings_source` in the log should read `portfolio_api`)
   - Yahoo bars are current at decision time (`last_bar` in the log)
   - the receipts read `VALID` / `EXECUTED`

   If the portfolio schema is not recognised, the runner falls back to local
   state, which is the last submitted weights drifted by price.
5. A cron schedule at the 7 run times on trading days is enough to automate it.
   Rounds that the agent holds simply upload nothing.

## Known limitations

- HH:30 fills are proxied by the hourly bar midpoint because the history has
  no HH:30 prints. `report.py` shows open/close sensitivity.
- Yahoo's 30m history is about 59 trading days, so all lookbacks are capped at
  40 days.
- The overall rank depends on the unknown field of competitors. The reference
  field is an assumption, and its makeup is stated in the report.
