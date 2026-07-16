"""
10-year DAILY-bar backtest — the swing-timeframe validation.

Reuses this project's portfolio engine (backtest.py: next-open fills, intrabar
stops checked before the peak advances, gap-through-stop pessimism, friction
per side, PDT accounting) with signals rebuilt on daily bars. Multi-day holds
mean ~26x fewer decisions than 15m — friction stops being the dominant term.

Data: run download_daily10y.py first (Yahoo, 10y, adjusted).

Honesty notes baked into the report:
  * Survivorship bias is REAL at 10 years: today's S&P 500 list applied
    backwards excludes the delisted losers. Results are optimistic by some
    margin — treat "beats SPY narrowly" as "probably doesn't".
  * Parameters (SMA20/50, RSI thresholds, stop widths swept openly) were
    fixed before this run; the sweep is reported in full, not cherry-picked.

Usage:
    python backtest_daily.py --trail-pct 5
    python backtest_daily.py --trail-pct 5 --cost-bps 10 --runs momentum
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from strategies import StrategyParams, build_signals_daily, daily_flag_by_date

ROOT = Path(__file__).parent
DAILY_DIR = ROOT / "data" / "daily10y"
RESULTS = ROOT / "results"

MR_TIME_STOP_DAYS = 10         # meanrev: give a dip 2 weeks, then get out
MR_HARD_STOP_DAILY = 0.05      # 5% — daily bars move ~2%/day; 2% was 15m-scaled
MIN_ROWS = 260                 # at least ~1 trading year of history


def load_universe_daily():
    spy = pd.read_parquet(DAILY_DIR / "SPY.parquet")
    regime = daily_flag_by_date(spy["Close"], StrategyParams().regime_sma)
    timeline = spy.index
    ts_to_bar = {ts: i for i, ts in enumerate(timeline)}

    tickers: dict[str, bt.TickerData] = {}
    entries_by_bar: dict[int, list[tuple[float, str, str]]] = {}
    skipped = 0
    for p in sorted(DAILY_DIR.glob("*.parquet")):
        t = p.stem
        if t == "SPY":
            continue
        df = pd.read_parquet(p)
        if len(df) < MIN_ROWS:
            skipped += 1
            continue
        frame = build_signals_daily(df, regime)
        if frame.empty or frame["close"].iloc[-1] < bt.MIN_PRICE:
            skipped += 1
            continue
        tickers[t] = bt.TickerData(frame, timeline)
        for sleeve, col, rank_col in (("momentum", "entry_mom", "mom_rank"),
                                      ("meanrev", "entry_mr", "mr_rank")):
            sel = frame.index[frame[col].to_numpy()]
            for ts, rank in frame.loc[sel, rank_col].items():
                bar = ts_to_bar.get(ts)
                if bar is not None:
                    entries_by_bar.setdefault(bar, []).append(
                        (float(rank), t, sleeve))
    n_sig = sum(len(v) for v in entries_by_bar.values())
    print(f"Loaded {len(tickers)} tickers ({skipped} skipped), "
          f"{n_sig} raw entry signals")
    return tickers, entries_by_bar, timeline, spy["Close"]


def yearly_returns(curve: pd.Series) -> dict[str, float]:
    marks = curve.groupby(curve.index.year).last()
    prev = curve.iloc[0]
    out = {}
    for year, v in marks.items():
        out[str(year)] = round((v / prev - 1) * 100, 1)
        prev = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--trail-pct", type=float, default=5.0,
                    help="momentum trailing stop, percent (daily bars need "
                         "room: typical daily range is ~2%%)")
    ap.add_argument("--runs", default="momentum,meanrev,combined")
    args = ap.parse_args()
    cost = args.cost_bps / 10_000
    trail = args.trail_pct / 100
    tag = f"daily10y_{int(args.cost_bps)}bps_trail{args.trail_pct:g}"

    RESULTS.mkdir(exist_ok=True)
    tickers, entries_by_bar, timeline, spy_close = load_universe_daily()
    dates = [ts.date() for ts in timeline]
    print(f"Timeline: {len(timeline)} sessions "
          f"({dates[0]} .. {dates[-1]}), cost {args.cost_bps} bps/side, "
          f"trail {args.trail_pct:g}%\n")

    spy_curve = spy_close / spy_close.iloc[0] * bt.START_EQUITY
    bench = bt.metrics(spy_curve, [], dates)
    bench["yearly"] = yearly_returns(spy_curve)

    all_results = {"config": {
        "timeframe": "daily", "cost_bps_per_side": args.cost_bps,
        "trail_pct": trail, "mr_time_stop_days": MR_TIME_STOP_DAYS,
        "mr_hard_stop": MR_HARD_STOP_DAILY,
        "start_equity": bt.START_EQUITY, "max_positions": bt.MAX_POSITIONS,
        "window": f"{dates[0]}..{dates[-1]}", "sessions": len(dates),
        "survivorship_bias": "today's S&P list applied backwards — optimistic",
    }, "benchmark_spy": bench, "runs": {}}

    curves = {"SPY buy&hold": spy_curve}
    wanted = [r.strip() for r in args.runs.split(",")]
    run_defs = [(n, s) for n, s in (("momentum", ("momentum",)),
                                    ("meanrev", ("meanrev",)),
                                    ("combined", ("momentum", "meanrev")))
                if n in wanted]
    for name, sleeves in run_defs:
        curve, trades = bt.run_portfolio(tickers, entries_by_bar, timeline,
                                         sleeves, cost, trail_pct=trail,
                                         mr_time_stop=MR_TIME_STOP_DAYS,
                                         mr_hard_stop=MR_HARD_STOP_DAILY)
        m = bt.metrics(curve, trades, dates)
        m["avg_hold_days"] = round(float(np.mean(
            [(t.exit_ts - t.entry_ts).days for t in trades])), 1) if trades else 0
        del m["avg_bars_held"]
        m["split"] = bt.split_halves(curve, trades)
        m["yearly"] = yearly_returns(curve)
        m["exit_reasons"] = dict(pd.Series([t.reason for t in trades]).value_counts())
        all_results["runs"][name] = m
        curves[name] = curve
        pd.DataFrame([{
            "ticker": t.ticker, "sleeve": t.sleeve, "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts, "entry_fill": round(t.entry_fill, 4),
            "exit_fill": round(t.exit_fill, 4), "qty": round(t.qty, 4),
            "pnl_usd": round(t.pnl, 2), "ret_pct": round(t.ret_pct, 3),
            "reason": t.reason,
        } for t in trades]).to_csv(RESULTS / f"trades_{name}_{tag}.csv",
                                   index=False)
        print(f"[{name:9s}] ret {m['window_return_pct']:+8.1f}%  "
              f"CAGR {m['cagr_pct']:+5.1f}%  PF {m['profit_factor']}  "
              f"win {m['win_rate_pct']}%  trades {m['trades']}  "
              f"maxDD {m['max_drawdown_pct']}%  sharpe {m['sharpe_daily_ann']}  "
              f"hold {m['avg_hold_days']}d")

    print(f"[SPY B&H  ] ret {bench['window_return_pct']:+8.1f}%  "
          f"CAGR {bench['cagr_pct']:+5.1f}%  maxDD {bench['max_drawdown_pct']}%  "
          f"sharpe {bench['sharpe_daily_ann']}")

    out = RESULTS / f"summary_{tag}.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, c in curves.items():
            ax.plot(c.index, c / c.iloc[0] * 100, label=label, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_title(f"S&P 500 daily bars, {dates[0]}..{dates[-1]} "
                     f"({args.cost_bps} bps/side, trail {args.trail_pct:g}%)")
        ax.set_ylabel("Equity (start = 100, log scale)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"equity_curves_{tag}.png", dpi=130)
        print(f"Wrote {RESULTS / f'equity_curves_{tag}.png'}")
    except Exception as exc:
        print(f"(chart skipped: {exc})")


if __name__ == "__main__":
    main()
