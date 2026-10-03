# Eyris: ICAIF 2026 Trading Agent

Portfolio agent for the ACM ICAIF 2026 Trading Agent Competition (Codabench competition 99).
The README covers setup and submission. The `reports/` folder covers every experiment and its verdict.

## Commands

- Tests: `python -m pytest -q` (51 tests; must stay green)
- One live decision, no upload: `python -m eyris.live decide --phase validation --round-id validation-2026-10-08-r1`
- Decide and upload the open round: `python -m eyris.live run` (needs CODABENCH_TOKEN, TEAM_ID, TEAM_TOKEN; alerts via NTFY_TOPIC / DISCORD_WEBHOOK_URL)
- Verify every logged round reproduces exactly: `python -m eyris.live replay`
- Paper-trade a normal trading day (never uploads; own ledger private/paper_state.json): `python -m eyris.live paper --loop --reset`, review via `validation_review.py analyze --phase paper`
- After a phase: `python scripts/validation_review.py all --phase validation` (report in private/review/<phase>/review.md)
- Change settings: `python scripts/set_config.py key=value --note "why"` (validated; logged to models/config_history.json)
- Research: `scripts/tune.py`, `scripts/report.py`, `scripts/event_experiment.py`, `scripts/agentic_research.py`
- Data / kit: `python scripts/download_data.py`; earnings calendar: `python scripts/fetch_earnings.py`

## Competition rules that constrain the code

- Each round we upload `decision.json` (30 target weights), not code. Weights are in [0, 0.30], sum ≤ 1, the rest is cash, long-only.
- 7 rounds/day. Deadlines 09:10, 10:25 … 15:25 ET. Fills at the open of the 1-minute bar at 09:30, 10:30 … 15:30. The fee is 0.1% of traded value.
- A missing round = hold, no fee. The agent's HOLD means "upload nothing", never "re-submit current weights".
- Score = mean of 4 ranks vs other teams: return, Sharpe (per round ×√1764), max drawdown (incl. 16:00 closes), turnover.
- LLMs are allowed only from the approved list (Claude Opus 5 / Sonnet 5 / Haiku 4.5, GPT-5, Gemini 2.5 …), with disclosure of prompts.
  `eyris/news.py` therefore uses `claude-opus-5`, not newer models, and no fallbacks to unlisted models.
- Only public info published before each deadline.
- Organizer clarifications (Discord, Oct 2026):
  - Free sources and free tiers of paid platforms are allowed; paid or private data feeds are not.
    Our feeds (organizer dataset, Yahoo bars/RSS/earnings dates, SEC EDGAR) are all free.
  - Free and open-source models, including finance-tuned ones, are allowed.
  - Nodexi Agentics LLM API credits: apply Oct 4 00:00 – Oct 5 23:59 ET to tsg.icaif@gmail.com with team name, team token
    and members' names/emails (team ID too). Access is set up Oct 6. The credits may be for a non-Anthropic provider; if so,
    add an adapter in `eyris/news.py` (same prompt, schema and logging).

## Engineering rules

- No lookahead: features use bars ≤ `info_end = exec_bar - 1`. The lookahead tests corrupt future data and require identical past decisions.
- The decision path (`agent.py`, `risk.py`, `execution.py`, `sentiment.py`) is deterministic and offline. Network access is only
  in the `live.py` fetch pre-step (bars, VIX), the optional `news.py` veto, and the `alerts.py` post-step.
- Every input of a live decision is logged in `decision_log.json["inputs"]`; `replay` must reproduce it bit-for-bit.
  Anything new that influences a decision must be added to `inputs` and to `replay_round`.
- Circuit breaker (`breaker_l1` 0.08, `breaker_name` 0.03): HOLD on a target jump vs the previous round's logged base target.
- Never crash in a round: any error or NaN → HOLD.
- Adopt a change only if it beats the current config on 2025-01..09 validation windows AND does not lose on 2021-24 train windows.
  The scoring is in `scripts/agentic_research.py:select_score`, over three competitor fields: full, active, llm.
  The 2025-Q4 holdout has been viewed several times; treat it as indicative only.
- Windows is a supported platform (the user runs it there): no shell-only commands in Python scripts.

## Current configuration (`models/params.json`)

Inverse-vol weights, 40-day lookback, 10% per-stock cap, 50% gross (cash sleeve), λ=0.25, no-trade band 0.05.
Median rank in backtests: about the top quarter of every reference field. `news_veto: true` is a Live Validation trial.

## Tested and rejected (see reports/)

LightGBM alpha tilt, min-variance, HRP, EWMA risk, vol targeting, the trend overlay, 11 LLM-proposed factors (none passed),
and earnings-day trims (they cut drawdown but cost about as much turnover rank). The overnight effect was absent in this data.
Gross 0.5–0.85 is tied against an LLM-style field; 0.5 wins on worst-case windows.
Overnight search 2026-10-02 (`reports/overnight_research.md`): stronger low-vol tilt (`vol_power`), top-k lowest-vol,
caps 0.06–0.15, bands up to 0.20: nothing passed; the high-gross validation leaders lost on train and holdout.
Structural experiment (`reports/structural_experiment.md`): own-NAV drawdown brake, ERC risk parity, sector budgets,
entry ramp: none passed. Gross 0.02–0.7 scan: 0.5 is best in train, validation and holdout.
VIX sentiment (`eyris/sentiment.py`, `reports/sentiment_experiment.md`): de-risking on fear loses; contrarian
(gross ×1.4 when VIX > 25) passes the rule narrowly but is not significant and raised the 2022 drawdown. Built, `vix_mode: off`.
October check (`reports/october_check.md`, 31 Oct 8-16 windows 2021-25): nothing beats current clearly; full earnings trims
and gross 0.7 are worse; VIX overlay slightly better, entirely from Oct 2022 and 2025.

## Open items

1. Registered (done, before Oct 3). Still to do: set the secrets CODABENCH_TOKEN, TEAM_ID, TEAM_TOKEN,
   optionally ANTHROPIC_API_KEY, SEC_USER_AGENT, NTFY_TOPIC; rehearse with `paper --loop` on Oct 5–7 (no uploads).
2. On Validation (Oct 8–9), check `private/<round>/decision_log.json`:
   - `holdings_source` should be `portfolio_api`. The organizer portfolio schema is undocumented; if it shows `none`/`local_state`,
     fix `live.weights_from_portfolio` using the logged `portfolio_shape`.
   - `last_bar` should be current.
   - receipts should be VALID/EXECUTED.
3. After Oct 9: run `scripts/validation_review.py all`, then decide `news_veto`, `gross` and `vix_mode` for Official
   (Oct 12–30) using the rules pre-registered in `eyris/review.py:DECISION_RULES` (written 2026-10-02, before Validation):
   gross stays 0.5, vix_mode stays off, and news_veto stays on only if the API answered ≥ 90% of rounds, there were ≤ 2 trims/day and every trim
   names a concrete event. If the simulator reconciliation fails, fix the accounting convention and re-run the backtests.
   Don't change the strategy because of Validation P&L.
4. Final materials are due 2026-11-03: code, run instructions, video, and LLM disclosure (prompts and settings from `llm_log.json`).
   Run `python -m eyris.live replay` first and include `private/replay_report.json`; keep `data/live/` and `private/`.
