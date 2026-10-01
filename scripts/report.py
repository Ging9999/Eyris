"""Backtest report: metrics vs baselines, equity curve, 15-day window distribution.

Uses the selected configuration in models/params.json. Holdout (2025-Q4) is
evaluated here and nowhere else.
Writes reports/backtest_report.md, reports/*.png, reports/report_tables.json.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.alpha import AlphaModel  # noqa: E402
from eyris.backtest import METRICS, summarize, target_policy  # noqa: E402
from eyris.config import Params  # noqa: E402
from eyris.experiment import PERIODS, Study  # noqa: E402

OUT = ROOT / "reports"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
BASELINES = {"EW buy & hold": "ew_buy_hold", "EW rebalanced (every round)": "ew_rebal_every_round",
             "Inverse-vol, full rebalance": "invvol_full_every_round"}


def load_selected():
    cfg = json.loads((ROOT / "models" / "params.json").read_text())
    params = Params(**cfg["params"])
    model = AlphaModel.load(ROOT / cfg["model"]) if params.use_alpha else None
    return params, model, cfg


def period_table(study, params, model):
    rows = {"Eyris agent": study.evaluate(params, model)}
    if params.use_alpha:
        rows["Eyris without alpha"] = study.evaluate(replace(params, use_alpha=False, tilt=0.0))
    for label, name in BASELINES.items():
        rows[label] = study.evaluate_baseline(name)
    rows["Cash"] = study.evaluate_baseline("cash")
    return rows


def fmt_table(rows, n_field):
    lines = ["| Strategy | Median rank (1 = best of %d) | Worst rank | Median 15d return | Worst 15d return "
             "| Median Sharpe | Median MDD | Worst MDD | Median turnover |" % n_field,
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, df in rows.items():
        s = summarize(df)
        lines.append(
            f"| {name} | {s.loc['rank_score', 'median']:.2f} | {s.loc['rank_score', 'worst']:.2f} "
            f"| {s.loc['cumulative_return', 'median']:+.2%} | {s.loc['cumulative_return', 'worst']:+.2%} "
            f"| {s.loc['sharpe_ratio', 'median']:.2f} | {s.loc['maximum_drawdown', 'median']:.2%} "
            f"| {s.loc['maximum_drawdown', 'worst']:.2%} | {s.loc['turnover', 'median']:.4f} |")
    return "\n".join(lines)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def equity_chart(val, hold, params, model, path):
    """One continuous portfolio from cash, 2025-01-02 to 2025-12-31."""
    p, r = val.p, val.r
    k0, k1 = val.k_first, int(hold.ks[-1]) + 1
    ks = np.arange(k0, k1)
    from eyris.backtest import agent_targets, baseline_policies, simulate
    from eyris.agent import Agent
    from eyris.experiment import FIELD_REFERENCE
    tg = agent_targets(Agent(params, model), p, r, ks)
    ref = agent_targets(Agent(FIELD_REFERENCE), p, r, ks)
    field = baseline_policies(p, r, k0, ref)
    curves = {"Eyris agent": simulate(r, k0, k1, target_policy(tg, k0, params))}
    for label, name in BASELINES.items():
        curves[label] = simulate(r, k0, k1, field[name])
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    dates = p.days[r.day[ks]] + pd.to_timedelta(r.number[ks], unit="h")
    hold_start = pd.Timestamp(PERIODS["holdout"][0])
    ax.axvspan(hold_start, dates[-1], color=GRID, alpha=0.6, linewidth=0)
    ax.text(hold_start, 1.0, "  holdout (Q4)", transform=ax.get_xaxis_transform(), va="top",
            color=INK2, fontsize=9)
    stats = {}
    for (label, res), color in zip(curves.items(), SERIES):
        nav = res.nav_end / 1e6
        ax.plot(dates, nav, color=color, linewidth=2 if label == "Eyris agent" else 1.4, label=label)
        m = res.metrics()
        stats[label] = m
        ax.annotate(f"{nav[-1]:.3f}", (dates[-1], nav[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.set_ylabel("NAV (USD millions)", color=INK2, fontsize=9)
    ax.set_title("Continuous 2025 run from USD 1M cash (validation + holdout)", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return stats


def window_chart(rows, path, title):
    names = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.6), dpi=150,
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax, col, label in ((axes[0], "rank_score", "Overall rank score in field (lower is better)"),
                           (axes[1], "cumulative_return", "15-day cumulative return")):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        data = [rows[n][col].to_numpy() for n in names][::-1]
        bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.6),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                        markeredgecolor="none", alpha=0.6))
        for patch, n in zip(bp["boxes"], names[::-1]):
            patch.set_facecolor(SERIES[0] if n == "Eyris agent" else GRID)
            patch.set_edgecolor(SURFACE)
        ax.set_yticks(range(1, len(names) + 1))
        ax.set_yticklabels(names[::-1], color=INK2, fontsize=9)
        ax.set_xlabel(label, color=INK2, fontsize=9)
        if col == "cumulative_return":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.set_yticklabels([])
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    params, model, cfg = load_selected()
    studies = {name: Study(name) for name in ("train", "validation", "holdout")}
    tables, md_tables = {}, {}
    for name, st in studies.items():
        rows = period_table(st, params, model)
        n_field = len(st.field) + 1
        md_tables[name] = fmt_table(rows, n_field)
        tables[name] = {k: summarize(v).to_dict() for k, v in rows.items()}
        tables[name]["_windows"] = len(st.wins)
        if name != "train":
            window_chart(rows, OUT / f"windows_{name}.png",
                         f"Rolling 15-trading-day windows, {name} ({len(st.wins)} windows)")
    eq = equity_chart(studies["validation"], studies["holdout"], params, model, OUT / "equity_2025.png")
    # execution-price proxy sensitivity on validation
    sens = {}
    for fill in ("open", "mid", "close"):
        st = studies["validation"] if fill == "mid" else Study("validation", fill=fill)
        sens[fill] = summarize(st.evaluate(params, model)).to_dict()
    (OUT / "report_tables.json").write_text(json.dumps(
        {"params": params.to_dict(), "periods": tables, "equity_2025": eq, "fill_sensitivity": sens},
        indent=1, default=float))
    write_markdown(cfg, params, md_tables, studies, eq, sens)


def write_markdown(cfg, params, md, studies, eq, sens):
    alpha_note = ""
    ap = OUT / "alpha_experiment.json"
    if ap.exists():
        a = json.loads(ap.read_text())
        b = a["best_alpha"]
        alpha_note = (f"Best alpha variant: horizon {b['horizon']} rounds, tilt {b['tilt']}, band {b['band']} -> "
                      f"median rank {b['rank_score_median']:.2f} vs baseline "
                      f"{a['baseline_validation']['rank_score_median']:.2f}. "
                      f"Recommendation: **{a['recommendation']}** (details in `alpha_experiment.md`).")
    eq_lines = ["| Strategy | Return | Sharpe | Max drawdown | Avg turnover / round |", "|---|---:|---:|---:|---:|"]
    for k, m in eq.items():
        eq_lines.append(f"| {k} | {m['cumulative_return']:+.2%} | {m['sharpe_ratio']:.2f} | "
                        f"{m['maximum_drawdown']:.2%} | {m['turnover']:.4f} |")
    sens_lines = ["| Fill proxy for HH:30 executions | Median rank | Median 15d return | Median turnover |",
                  "|---|---:|---:|---:|"]
    for f, s in sens.items():
        sens_lines.append(f"| {f} | {s['median']['rank_score']:.2f} | {s['median']['cumulative_return']:+.2%} "
                          f"| {s['median']['turnover']:.4f} |")
    text = f"""# Backtest report

Generated by `python scripts/report.py`. Selected configuration (`models/params.json`):

```json
{json.dumps(params.to_dict(), indent=1)}
```

{cfg.get("note", "")}

## How to read this

- Every 15-trading-day window (the Official Competition length, 105 rounds) is
  simulated from USD 1,000,000 cash with 0.1% fees, at the competition's exact
  rounds: 09:30 open, then HH:30 fills proxied by the hourly bar's (open+close)/2.
- Metrics follow the official definitions (decision-period Sharpe x sqrt(1764),
  drawdown including 16:00 closes, turnover = mean traded notional / NAV
  including the initial allocation).
- **Rank** is the official overall rank score (mean of the four metric ranks)
  against a reference field of {len(studies['validation'].field)} strategies standing in for other
  teams: cash, equal-weight variants, inverse-vol, the starter-kit momentum
  agent, 20-day momentum, 10 random buy-and-hold portfolios and 5 random
  daily-reshuffled portfolios. Lower is better. The real field is unknown, so
  absolute ranks are indicative; comparisons between rows are what matter.
- Periods: train 2021-2024 ({studies['train'].wins.__len__()} windows), validation
  2025-01..09 ({len(studies['validation'].wins)} windows, used for tuning), holdout 2025-Q4
  ({len(studies['holdout'].wins)} windows, evaluated once). The dataset ends 2025-12-31, so there is no
  January 2026 data.

## Validation (2025-01-02 .. 2025-09-30)

{md['validation']}

![validation windows](windows_validation.png)

## Holdout (2025-Q4, untouched during tuning)

{md['holdout']}

![holdout windows](windows_holdout.png)

## Train (2021-2024, out-of-sample for the baseline parameters' selection)

{md['train']}

## Equity curve

![equity](equity_2025.png)

{chr(10).join(eq_lines)}

## Execution-price sensitivity (validation)

{chr(10).join(sens_lines)}

## Alpha tilt

{alpha_note}
"""
    (OUT / "backtest_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
