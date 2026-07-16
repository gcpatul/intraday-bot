"""
Portfolio backtester — S&P 500 on 15-minute bars.

Modelled to match the discipline of chawani's proven daily backtester, adapted
to intraday:
  * Signals are computed on bar close t, filled at bar t+1 OPEN (no look-ahead).
  * Stops are checked intrabar against the bar's LOW, against the PRIOR peak,
    and only then is the peak advanced (a run-up-then-reverse can still stop out).
  * Gap through the stop => filled at the open, not the (better) stop level.
  * Friction on every fill (default 5 bps per side; use --cost-bps to stress).
  * PDT accounting: counts same-day round trips and the worst rolling
    5-trading-day total, because the real account is under the $25k line.

Portfolio rules (mirror the live bot): max 8 concurrent positions, 12.5% of
current equity per position, cash-capped (no margin), long only.

Data sources:
    --data-source yahoo    data/m15 + data/daily            (60d free window)
    --data-source alpaca   data/alpaca_m15 + data/alpaca_daily  (years; run
                           alpaca_data.py first)

Out-of-sample discipline: the 3% trail width was chosen by looking at the
Apr-Jul 2026 Yahoo window. When validating on Alpaca history, run with
    --end-date 2026-04-14
so the evaluation window contains only data that parameter choice never saw.

Usage:
    python backtest.py
    python backtest.py --data-source alpaca --trail-pct 3 --end-date 2026-04-14
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategies import StrategyParams, build_signals, daily_flag_by_date, session_filter

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

START_EQUITY = 12_500.0        # matches the real Alpaca account
MAX_POSITIONS = 8
POS_FRAC = 0.125
TRAIL_PCT = 0.01               # momentum sleeve: 1% trailing stop
MR_HARD_STOP = 0.02            # mean-reversion sleeve: 2% fixed stop
MR_TIME_STOP_BARS = 26         # ~1 trading session of 15m bars
MIN_PRICE = 5.0                # skip sub-$5 names (spreads are not 5bps there)


@dataclass
class Position:
    ticker: str
    sleeve: str
    qty: float
    entry_fill: float          # includes entry cost
    entry_ts: pd.Timestamp
    peak: float                # high-water mark for the trailing stop
    bars_held: int = 0
    exit_queued: str | None = None


@dataclass
class Trade:
    ticker: str
    sleeve: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_fill: float
    exit_fill: float
    qty: float
    reason: str

    @property
    def pnl(self) -> float:
        return self.qty * (self.exit_fill - self.entry_fill)

    @property
    def ret_pct(self) -> float:
        return (self.exit_fill / self.entry_fill - 1) * 100

    @property
    def same_day(self) -> bool:
        return self.entry_ts.date() == self.exit_ts.date()


class TickerData:
    """Master-timeline-aligned numpy arrays; NaN open == no bar this slot.
    float32 keeps ~500 tickers x 2 years of 15m bars around ~150 MB."""

    __slots__ = ("o", "h", "l", "c", "exit_mom", "exit_mr")

    def __init__(self, frame: pd.DataFrame, timeline: pd.DatetimeIndex):
        f = frame.reindex(timeline)
        self.o = f["open"].to_numpy(dtype=np.float32)
        self.h = f["high"].to_numpy(dtype=np.float32)
        self.l = f["low"].to_numpy(dtype=np.float32)
        self.c = f["close"].to_numpy(dtype=np.float32)
        self.exit_mom = f["exit_mom"].fillna(False).to_numpy(dtype=bool)
        self.exit_mr = f["exit_mr"].fillna(False).to_numpy(dtype=bool)

    def has_bar(self, i: int) -> bool:
        return not np.isnan(self.o[i])


def load_universe(m15_dir: Path, daily_dir: Path, end_date: str | None):
    """Returns (tickers, entries_by_bar, timeline, spy_15m_close)."""
    spy_daily = pd.read_parquet(daily_dir / "SPY.parquet")
    regime = daily_flag_by_date(spy_daily["Close"], StrategyParams().regime_sma)
    spy_15m = session_filter(pd.read_parquet(m15_dir / "SPY.parquet"))
    timeline = spy_15m.index
    if end_date:
        cutoff = pd.Timestamp(end_date, tz=timeline.tz) + pd.Timedelta(days=1)
        timeline = timeline[timeline < cutoff]
        spy_15m = spy_15m.loc[timeline]
    ts_to_bar = {ts: i for i, ts in enumerate(timeline)}

    tickers: dict[str, TickerData] = {}
    entries_by_bar: dict[int, list[tuple[float, str, str]]] = {}
    skipped = 0
    for p in sorted(m15_dir.glob("*.parquet")):
        t = p.stem
        if t == "SPY":
            continue
        daily_path = daily_dir / f"{t}.parquet"
        if not daily_path.exists():
            skipped += 1
            continue
        frame = build_signals(pd.read_parquet(p), pd.read_parquet(daily_path),
                              regime)
        if frame.empty or frame["close"].iloc[-1] < MIN_PRICE:
            skipped += 1
            continue
        tickers[t] = TickerData(frame, timeline)
        for sleeve, col, rank_col in (("momentum", "entry_mom", "mom_rank"),
                                      ("meanrev", "entry_mr", "mr_rank")):
            sel = frame.index[frame[col].to_numpy()]
            ranks = frame.loc[sel, rank_col]
            for ts, rank in ranks.items():
                bar = ts_to_bar.get(ts)
                if bar is not None:
                    entries_by_bar.setdefault(bar, []).append(
                        (float(rank), t, sleeve))
    n_sig = sum(len(v) for v in entries_by_bar.values())
    print(f"Loaded {len(tickers)} tickers ({skipped} skipped), "
          f"{n_sig} raw entry signals")
    return tickers, entries_by_bar, timeline, spy_15m["Close"]


def run_portfolio(tickers: dict[str, TickerData],
                  entries_by_bar: dict[int, list],
                  timeline: pd.DatetimeIndex,
                  sleeves: tuple[str, ...],
                  cost: float,
                  trail_pct: float = TRAIL_PCT,
                  mr_time_stop: int = MR_TIME_STOP_BARS,
                  mr_hard_stop: float = MR_HARD_STOP) -> tuple[pd.Series, list[Trade]]:
    cash = START_EQUITY
    positions: dict[str, Position] = {}
    pending_entries: list[tuple[float, str, str]] = []
    trades: list[Trade] = []
    last_close: dict[str, float] = {}
    equity_arr = np.empty(len(timeline))
    equity = START_EQUITY
    n_bars = len(timeline)

    for bar_i in range(n_bars):
        ts = timeline[bar_i]

        # (a) fill queued EXITS at this bar's open
        for tk in [tk for tk, pos in positions.items() if pos.exit_queued]:
            td = tickers[tk]
            if not td.has_bar(bar_i):
                continue                      # no bar; try again next bar
            pos = positions.pop(tk)
            fill = float(td.o[bar_i]) * (1 - cost)
            cash += pos.qty * fill
            trades.append(Trade(tk, pos.sleeve, pos.entry_ts, ts,
                                pos.entry_fill, fill, pos.qty, pos.exit_queued))

        # (b) fill queued ENTRIES at this bar's open (ranked, slot-capped)
        for rank, tk, sleeve in pending_entries:
            if len(positions) >= MAX_POSITIONS or tk in positions:
                continue
            td = tickers[tk]
            if not td.has_bar(bar_i):
                continue                      # signal went stale; drop it
            invest = min(equity * POS_FRAC, cash)
            if invest < 100:
                continue
            fill = float(td.o[bar_i]) * (1 + cost)
            qty = invest / fill
            cash -= invest
            positions[tk] = Position(tk, sleeve, qty, fill, ts, peak=fill)
        pending_entries = []

        # (c) intrabar stop check — prior peak first, THEN advance the peak
        for tk in list(positions):
            pos = positions[tk]
            td = tickers[tk]
            if not td.has_bar(bar_i):
                continue
            pos.bars_held += 1
            if pos.sleeve == "momentum":
                stop_level = pos.peak * (1 - trail_pct)
            else:
                stop_level = pos.entry_fill * (1 - mr_hard_stop)
            if td.l[bar_i] <= stop_level:
                raw_fill = min(stop_level, float(td.o[bar_i]))  # gap => worse
                fill = raw_fill * (1 - cost)
                cash += pos.qty * fill
                trades.append(Trade(tk, pos.sleeve, pos.entry_ts, ts,
                                    pos.entry_fill, fill, pos.qty,
                                    "trail_stop" if pos.sleeve == "momentum"
                                    else "hard_stop"))
                del positions[tk]
            elif pos.sleeve == "momentum":
                pos.peak = max(pos.peak, float(td.h[bar_i]))

        # (d) evaluate signals on this bar's close -> queue next-bar actions
        for tk, pos in positions.items():
            td = tickers[tk]
            if not td.has_bar(bar_i) or pos.exit_queued:
                continue
            if pos.sleeve == "momentum" and td.exit_mom[bar_i]:
                pos.exit_queued = "cross_down"
            elif pos.sleeve == "meanrev":
                if td.exit_mr[bar_i]:
                    pos.exit_queued = "rsi_exit"
                elif pos.bars_held >= mr_time_stop:
                    pos.exit_queued = "time_stop"

        if bar_i < n_bars - 1:
            live = len(positions) - sum(1 for p in positions.values()
                                        if p.exit_queued)
            slots = MAX_POSITIONS - live
            if slots > 0:
                cands = [e for e in entries_by_bar.get(bar_i, ())
                         if e[2] in sleeves and e[1] not in positions]
                cands.sort(reverse=True)      # strongest rank first
                seen: set[str] = set()
                for rank, tk, sleeve in cands:
                    if len(pending_entries) >= slots:
                        break
                    if tk in seen:
                        continue
                    seen.add(tk)
                    pending_entries.append((rank, tk, sleeve))

        # (e) mark to market
        for tk, pos in positions.items():
            td = tickers[tk]
            if td.has_bar(bar_i):
                last_close[tk] = float(td.c[bar_i])
        equity = cash + sum(pos.qty * last_close.get(tk, pos.entry_fill)
                            for tk, pos in positions.items())
        equity_arr[bar_i] = equity

    # liquidate whatever is still open at the final close (accounting only)
    final_ts = timeline[-1]
    for tk, pos in positions.items():
        fill = last_close.get(tk, pos.entry_fill) * (1 - cost)
        cash += pos.qty * fill
        trades.append(Trade(tk, pos.sleeve, pos.entry_ts, final_ts,
                            pos.entry_fill, fill, pos.qty, "end_of_test"))
    equity_arr[-1] = cash
    return pd.Series(equity_arr, index=timeline), trades


# ----------------------------- metrics ------------------------------------ #

def daily_marks(curve: pd.Series) -> pd.Series:
    return curve.groupby([ts.date() for ts in curve.index]).last()


def metrics(curve: pd.Series, trades: list[Trade],
            timeline_dates: list) -> dict:
    ret = (curve.iloc[-1] / curve.iloc[0] - 1) * 100
    n_sessions = len(timeline_dates)
    years = n_sessions / 252
    cagr = ((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1) * 100 \
        if years > 0.2 else None
    d = daily_marks(curve)
    rets = d.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 2 and rets.std() > 0 else 0.0
    peak = curve.cummax()
    max_dd = float(((curve - peak) / peak).min() * 100)

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    # PDT: same-day round trips, worst rolling 5-trading-day window
    dt_dates = [t.exit_ts.date() for t in trades if t.same_day]
    day_trades = len(dt_dates)
    counts = pd.Series(dt_dates).value_counts()
    per_day = pd.Series(0, index=pd.Index(sorted(set(timeline_dates))))
    per_day.update(counts)
    worst_5d = int(per_day.rolling(5).sum().max()) if len(per_day) >= 5 else day_trades

    return {
        "window_return_pct": round(float(ret), 2),
        "cagr_pct": round(float(cagr), 2) if cagr is not None else None,
        "sharpe_daily_ann": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
        "profit_factor": round(pf, 2) if pf != float("inf") else None,
        "avg_trade_pct": round(float(np.mean([t.ret_pct for t in trades])), 3) if trades else 0.0,
        "avg_bars_held": round(float(np.mean([max((t.exit_ts - t.entry_ts).total_seconds() / 900, 1) for t in trades])), 1) if trades else 0.0,
        "same_day_round_trips": day_trades,
        "worst_5day_daytrade_count": worst_5d,
        "final_equity": round(float(curve.iloc[-1]), 2),
    }


def split_halves(curve: pd.Series, trades: list[Trade]) -> dict:
    mid = curve.index[len(curve) // 2]
    first, second = curve[curve.index < mid], curve[curve.index >= mid]
    def _pf(ts):
        w = sum(t.pnl for t in ts if t.pnl > 0)
        l = abs(sum(t.pnl for t in ts if t.pnl <= 0))
        return round(w / l, 2) if l else None
    t1 = [t for t in trades if t.exit_ts < mid]
    t2 = [t for t in trades if t.exit_ts >= mid]
    return {
        "h1_return_pct": round(float(first.iloc[-1] / first.iloc[0] - 1) * 100, 2),
        "h2_return_pct": round(float(second.iloc[-1] / second.iloc[0] - 1) * 100, 2),
        "h1_profit_factor": _pf(t1), "h2_profit_factor": _pf(t2),
        "h1_trades": len(t1), "h2_trades": len(t2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=5.0,
                    help="friction per side, in basis points")
    ap.add_argument("--trail-pct", type=float, default=TRAIL_PCT * 100,
                    help="momentum trailing stop, percent (default 1.0)")
    ap.add_argument("--runs", default="momentum,meanrev,combined",
                    help="comma-separated subset of runs")
    ap.add_argument("--data-source", choices=("yahoo", "alpaca"),
                    default="yahoo")
    ap.add_argument("--end-date", default=None,
                    help="YYYY-MM-DD; evaluate only bars up to this date "
                         "(out-of-sample discipline)")
    args = ap.parse_args()
    cost = args.cost_bps / 10_000
    trail = args.trail_pct / 100
    if args.data_source == "alpaca":
        m15_dir, daily_dir = ROOT / "data" / "alpaca_m15", ROOT / "data" / "alpaca_daily"
    else:
        m15_dir, daily_dir = ROOT / "data" / "m15", ROOT / "data" / "daily"
    tag = (f"{args.data_source}_" if args.data_source != "yahoo" else "") \
        + f"{int(args.cost_bps)}bps" \
        + (f"_trail{args.trail_pct:g}" if trail != TRAIL_PCT else "") \
        + (f"_to{args.end_date}" if args.end_date else "")

    RESULTS.mkdir(exist_ok=True)
    tickers, entries_by_bar, timeline, spy_close = load_universe(
        m15_dir, daily_dir, args.end_date)
    dates = sorted({ts.date() for ts in timeline})
    print(f"Timeline: {len(timeline)} bars over {len(dates)} sessions "
          f"({dates[0]} .. {dates[-1]}), cost {args.cost_bps} bps/side\n")

    spy_curve = spy_close / spy_close.iloc[0] * START_EQUITY
    bench = metrics(spy_curve, [], dates)

    all_results = {"config": {
        "data_source": args.data_source,
        "cost_bps_per_side": args.cost_bps, "start_equity": START_EQUITY,
        "max_positions": MAX_POSITIONS, "pos_frac": POS_FRAC,
        "trail_pct": trail, "mr_hard_stop": MR_HARD_STOP,
        "mr_time_stop_bars": MR_TIME_STOP_BARS,
        "window": f"{dates[0]}..{dates[-1]}", "sessions": len(dates),
    }, "benchmark_spy": bench, "runs": {}}

    curves = {"SPY buy&hold": spy_curve}
    wanted = [r.strip() for r in args.runs.split(",")]
    run_defs = [(n, s) for n, s in (("momentum", ("momentum",)),
                                    ("meanrev", ("meanrev",)),
                                    ("combined", ("momentum", "meanrev")))
                if n in wanted]
    for name, sleeves in run_defs:
        curve, trades = run_portfolio(tickers, entries_by_bar, timeline,
                                      sleeves, cost, trail_pct=trail)
        m = metrics(curve, trades, dates)
        m["split"] = split_halves(curve, trades)
        m["exit_reasons"] = dict(pd.Series([t.reason for t in trades]).value_counts())
        all_results["runs"][name] = m
        curves[name] = curve
        pd.DataFrame([{
            "ticker": t.ticker, "sleeve": t.sleeve, "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts, "entry_fill": round(t.entry_fill, 4),
            "exit_fill": round(t.exit_fill, 4), "qty": round(t.qty, 4),
            "pnl_usd": round(t.pnl, 2), "ret_pct": round(t.ret_pct, 3),
            "reason": t.reason, "same_day": t.same_day,
        } for t in trades]).to_csv(RESULTS / f"trades_{name}_{tag}.csv",
                                   index=False)
        print(f"[{name:9s}] ret {m['window_return_pct']:+7.2f}%  "
              f"cagr {m['cagr_pct']}%  PF {m['profit_factor']}  "
              f"win {m['win_rate_pct']}%  trades {m['trades']}  "
              f"maxDD {m['max_drawdown_pct']}%  "
              f"dayTrades {m['same_day_round_trips']} "
              f"(worst5d {m['worst_5day_daytrade_count']})")

    print(f"[SPY B&H  ] ret {bench['window_return_pct']:+7.2f}%  "
          f"cagr {bench['cagr_pct']}%  maxDD {bench['max_drawdown_pct']}%  "
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
            ax.plot(range(len(c)), c / c.iloc[0] * 100, label=label, linewidth=1.2)
        ax.set_title(f"S&P 500, 15-min bars [{args.data_source}], "
                     f"{dates[0]}..{dates[-1]} ({args.cost_bps} bps/side, "
                     f"trail {args.trail_pct:g}%)")
        ax.set_ylabel("Equity (start = 100)")
        ax.set_xlabel("15-min bars")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"equity_curves_{tag}.png", dpi=130)
        print(f"Wrote {RESULTS / f'equity_curves_{tag}.png'}")
    except Exception as exc:
        print(f"(chart skipped: {exc})")


if __name__ == "__main__":
    main()
