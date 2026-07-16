"""
Rotation strategies — 10-year daily backtest: DUAL MOMENTUM and
CROSS-SECTIONAL MOMENTUM. Uses data/daily10y (run download_daily10y.py).

=============================================================================
PRE-REGISTERED PASS CRITERIA — written BEFORE the first run of this script.
Do not move these goalposts after seeing results.

DUAL MOMENTUM (SPY when its own 12-month return > 0, else cash):
  PASS iff ALL of:
   D1. CAGR >= SPY CAGR - 2 percentage points
   D2. max drawdown <= 60% of SPY's max drawdown
   D3. both halves of the window positive
   D4. D1-D3 still hold at 10 bps/side

CROSS-SECTIONAL MOMENTUM (monthly top-20 by 12-1 momentum, equal weight;
also run with the SPY 200-SMA regime filter):
  PASS iff ALL of:
   X1. total return > SPY after costs
   X2. max drawdown <= SPY's
   X3. both halves positive
   X4. X1-X3 still hold at 10 bps/side
  MANDATORY CAVEAT: survivorship bias inflates cross-sectional momentum more
  than any strategy we've tested (we are ranking TODAY'S winners list by past
  returns). A pass here is PROVISIONAL until reproduced on bias-free data.

POINT-IN-TIME MODE (--pit) — decision rule pre-registered 2026-07-12,
BEFORE its first run:
  Rank only stocks that were actual index members on each ranking date
  (fja05680/sp500 membership history; clean renames aliased, e.g. FB->META).
  Residual bias: members whose price history Yahoo no longer serves
  (acquired/bankrupt) are absent — monthly coverage is measured and printed.
  RULE: if xs_momentum_regime still passes X1-X3 at BOTH 5 and 10 bps under
  --pit, it upgrades to VALIDATED (free-data grade). If it fails, the edge
  is attributed to survivorship bias and the strategy is DROPPED. No
  parameter changes are permitted in response to --pit results.
=============================================================================

Modelling notes:
  * Signals at month-end close; positions apply from the NEXT trading day.
  * Cash yields 0% (conservative: T-bills paid ~4-5% for much of the window,
    which would flatter both strategies' cash periods).
  * All curves (including the SPY benchmark) start at the first month-end
    where a 12-month signal exists, so nobody gets a warmup head start.
  * Costs: --cost-bps per side on every switched sleeve.
  * Delisted/missing daily returns average out of the basket (weight
    implicitly redistributes); noted, not perfect.

Usage:
    python backtest_rotation.py                # 5 bps/side
    python backtest_rotation.py --cost-bps 10  # stress
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from backtest_daily import yearly_returns

ROOT = Path(__file__).parent
DAILY_DIR = ROOT / "data" / "daily10y"
RESULTS = ROOT / "results"

TOP_N = 20
MOM_SKIP = 21          # 12-1 momentum: skip the most recent month
MOM_LOOKBACK = 252
MIN_PRICE = 5.0

# clean renames where the new ticker carries continuous price history
ALIASES = {"FB": "META", "ANTM": "ELV", "CTL": "LUMN", "KORS": "CPRI",
           "VIAC": "PARA", "WLTW": "WTW", "FISV": "FI", "PKI": "RVTY",
           "RE": "EG", "FLT": "CPAY"}


def load_membership() -> list[tuple[pd.Timestamp, frozenset]]:
    rows = list(csv.reader(open(ROOT / "data" / "sp500_pit_membership.csv")))[1:]
    out = []
    for d, ticks in rows:
        s = frozenset(ALIASES.get(x, x) for x in
                      (t.strip().replace(".", "-") for t in ticks.split(",")))
        out.append((pd.Timestamp(d), s))
    return out


def members_asof(membership: list, ts: pd.Timestamp) -> frozenset:
    last = frozenset()
    for d, s in membership:
        if d <= ts:
            last = s
        else:
            break
    return last


def load_closes() -> pd.DataFrame:
    cols = {}
    for p in sorted(DAILY_DIR.glob("*.parquet")):
        s = pd.read_parquet(p)["Close"]
        cols[p.stem] = s
    df = pd.DataFrame(cols).sort_index()
    if "SPY" not in df.columns:
        raise SystemExit("SPY missing from daily10y — cannot benchmark.")
    return df


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    per = idx.to_period("M")
    keep = ~pd.Series(per).duplicated(keep="last").values
    return idx[keep]


def halves(curve: pd.Series) -> dict:
    mid = curve.index[len(curve) // 2]
    h1 = curve[curve.index < mid]
    h2 = curve[curve.index >= mid]
    return {"h1_return_pct": round(float(h1.iloc[-1] / h1.iloc[0] - 1) * 100, 2),
            "h2_return_pct": round(float(h2.iloc[-1] / h2.iloc[0] - 1) * 100, 2)}


def dual_momentum(spy: pd.Series, mes: pd.DatetimeIndex, cost: float) -> tuple[pd.Series, int]:
    r12 = (spy / spy.shift(MOM_LOOKBACK) - 1).reindex(mes)
    flag = pd.Series(np.nan, index=spy.index)
    flag.loc[mes] = (r12 > 0).astype(float)
    flag = flag.shift(1).ffill()          # signal applies from the next day
    ret = spy.pct_change().fillna(0.0) * flag.fillna(0.0)
    switches = flag.fillna(0.0).diff().abs() > 0
    ret[switches] -= cost                 # one side per switch
    curve = (1 + ret).cumprod() * bt.START_EQUITY
    return curve, int(switches.sum())


def xs_momentum(closes: pd.DataFrame, mes: pd.DatetimeIndex, cost: float,
                regime: bool,
                membership: list | None = None) -> tuple[pd.Series, int, list]:
    spy = closes["SPY"]
    uni = closes.drop(columns=["SPY"])
    mom = uni.shift(MOM_SKIP) / uni.shift(MOM_LOOKBACK) - 1
    spy_sma = spy.rolling(200).mean()
    rets = uni.pct_change()

    daily_ret = pd.Series(0.0, index=closes.index)
    prev: set[str] = set()
    n_rebalances = 0
    coverage: list[float] = []
    for i, me in enumerate(mes[:-1]):
        row = mom.loc[me].dropna()
        row = row[uni.loc[me].reindex(row.index) > MIN_PRICE]
        if membership is not None:
            elig = members_asof(membership, me)
            if elig:
                coverage.append(
                    len(elig & set(uni.columns)) / len(elig))
            row = row[row.index.isin(elig)]
        if regime and not (spy.loc[me] > spy_sma.loc[me]):
            top: set[str] = set()
        elif len(row) >= 100:
            top = set(row.nlargest(TOP_N).index)
        else:
            top = set()
        start, end = mes[i], mes[i + 1]
        days = closes.index[(closes.index > start) & (closes.index <= end)]
        if len(days) == 0:
            continue
        if top:
            daily_ret.loc[days] = rets.loc[days, sorted(top)].mean(axis=1).fillna(0.0)
        turn = len(prev - top) + len(top - prev)
        if turn:
            daily_ret.loc[days[0]] -= (turn / TOP_N) * cost
            n_rebalances += 1
        prev = top
    curve = (1 + daily_ret).cumprod() * bt.START_EQUITY
    return curve, n_rebalances, coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--pit", action="store_true",
                    help="point-in-time membership mode (see docstring rule)")
    args = ap.parse_args()
    cost = args.cost_bps / 10_000
    membership = load_membership() if args.pit else None

    closes = load_closes()
    spy = closes["SPY"]
    mes = month_ends(closes.index)

    # common start: first month-end with a valid 12m signal; no warmup edge
    first_valid = next(me for me in mes if not np.isnan(
        spy.loc[me] / spy.shift(MOM_LOOKBACK).loc[me] - 1))
    window = closes.index[closes.index > first_valid]
    print(f"Window: {window[0].date()} .. {window[-1].date()} "
          f"({len(window)} sessions), cost {args.cost_bps} bps/side\n")

    def clip(c: pd.Series) -> pd.Series:
        c = c.loc[window]
        return c / c.iloc[0] * bt.START_EQUITY

    dates = [ts.date() for ts in window]
    spy_curve = clip(spy)
    bench = bt.metrics(spy_curve, [], dates)
    bench["yearly"] = yearly_returns(spy_curve)

    runs = {}
    curves = {"SPY buy&hold": spy_curve}

    if not args.pit:
        c, n = dual_momentum(spy, mes, cost)
        curves["dual momentum"] = clip(c)
        m = bt.metrics(curves["dual momentum"], [], dates)
        m.update(halves(curves["dual momentum"]))
        m["yearly"] = yearly_returns(curves["dual momentum"])
        m["switches"] = n
        runs["dual_momentum"] = m

    suffix = "_pit" if args.pit else ""
    for regime, base in ((False, "xs_momentum"), (True, "xs_momentum_regime")):
        name = base + suffix
        c, n, cov = xs_momentum(closes, mes, cost, regime, membership)
        curves[name] = clip(c)
        m = bt.metrics(curves[name], [], dates)
        m.update(halves(curves[name]))
        m["yearly"] = yearly_returns(curves[name])
        m["rebalances"] = n
        if cov:
            m["membership_coverage"] = {
                "mean": round(float(np.mean(cov)), 3),
                "min": round(float(np.min(cov)), 3)}
        runs[name] = m

    # ---- pre-registered verdicts ---- #
    if "dual_momentum" in runs:
        d = runs["dual_momentum"]
        d["criteria"] = {
            "D1_cagr_within_2pp_of_spy": d["cagr_pct"] >= bench["cagr_pct"] - 2,
            "D2_maxdd_leq_60pct_of_spy": abs(d["max_drawdown_pct"]) <= 0.60 * abs(bench["max_drawdown_pct"]),
            "D3_both_halves_positive": d["h1_return_pct"] > 0 and d["h2_return_pct"] > 0,
        }
    for base in ("xs_momentum", "xs_momentum_regime"):
        name = base + suffix
        if name not in runs:
            continue
        x = runs[name]
        x["criteria"] = {
            "X1_beats_spy_total_return": x["window_return_pct"] > bench["window_return_pct"],
            "X2_maxdd_leq_spy": abs(x["max_drawdown_pct"]) <= abs(bench["max_drawdown_pct"]),
            "X3_both_halves_positive": x["h1_return_pct"] > 0 and x["h2_return_pct"] > 0,
        }

    for name, m in runs.items():
        crits = m["criteria"]
        verdict = "PASS(provisional)" if all(crits.values()) else "FAIL"
        print(f"[{name:18s}] ret {m['window_return_pct']:+8.1f}%  "
              f"CAGR {m['cagr_pct']:+5.1f}%  maxDD {m['max_drawdown_pct']}%  "
              f"sharpe {m['sharpe_daily_ann']}  "
              f"halves {m['h1_return_pct']:+.1f}/{m['h2_return_pct']:+.1f}  "
              f"-> {verdict}")
        for k, v in crits.items():
            print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"[SPY B&H          ] ret {bench['window_return_pct']:+8.1f}%  "
          f"CAGR {bench['cagr_pct']:+5.1f}%  maxDD {bench['max_drawdown_pct']}%  "
          f"sharpe {bench['sharpe_daily_ann']}")

    RESULTS.mkdir(exist_ok=True)
    tag = f"rotation{'_pit' if args.pit else ''}_{int(args.cost_bps)}bps"
    out = {"config": {
        "timeframe": "daily-monthly-rotation",
        "cost_bps_per_side": args.cost_bps,
        "window": f"{dates[0]}..{dates[-1]}", "sessions": len(dates),
        "survivorship_bias": "today's S&P list backwards — inflates XS momentum badly",
    }, "benchmark_spy": bench, "runs": runs}
    (RESULTS / f"summary_{tag}.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\nWrote results/summary_{tag}.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, c in curves.items():
            ax.plot(c.index, c / c.iloc[0] * 100, label=label, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_title(f"Rotation strategies vs SPY, {dates[0]}..{dates[-1]} "
                     f"({args.cost_bps} bps/side)")
        ax.set_ylabel("Equity (start = 100, log)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"equity_curves_{tag}.png", dpi=130)
        print(f"Wrote results/equity_curves_{tag}.png")
    except Exception as exc:
        print(f"(chart skipped: {exc})")


if __name__ == "__main__":
    main()
