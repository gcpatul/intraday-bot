"""
Signal generation — pure functions of price history, no side effects.

Two sleeves, designed to complement each other:

  MOMENTUM (trend-following on 15m):
    enter  when SMA20 crosses above SMA50, price > SMA50, RSI14 < 75,
           and the SPY daily regime is risk-on (SPY > 200-day SMA as of
           the PREVIOUS close — no look-ahead).
    exit   via 1% trailing stop (handled by the backtester/bot), or
           SMA20 crossing back below SMA50.

  MEAN REVERSION (buy-the-dip on 15m, Connors-style RSI(2)):
    enter  when RSI(2) < 10 on 15m, the stock is above its own 200-day
           SMA (prior close), and the SPY regime is risk-on.
    exit   when RSI(2) > 60, or a 26-bar (~1 trading day) time stop,
           or a 2% hard stop from entry.

Every daily-derived filter is shifted by one day so a bar on day D only
ever sees daily data through D-1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class StrategyParams:
    fast_sma: int = 20
    slow_sma: int = 50
    rsi_slow: int = 14
    rsi_slow_max: float = 75.0
    rsi_fast: int = 2
    rsi_fast_entry: float = 10.0
    rsi_fast_exit: float = 60.0
    daily_trend_sma: int = 200
    regime_sma: int = 200


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def session_filter(m15: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-session bars only (starts 09:30–15:45 ET)."""
    idx = m15.index
    if idx.tz is None:
        idx = idx.tz_localize(NY)
    else:
        idx = idx.tz_convert(NY)
    m15 = m15.copy()
    m15.index = idx
    minutes = idx.hour * 60 + idx.minute
    return m15[(minutes >= 9 * 60 + 30) & (minutes <= 15 * 60 + 45)]


def daily_flag_by_date(daily_close: pd.Series, sma_period: int) -> pd.Series:
    """Boolean 'above SMA' per calendar date, lagged one day (no look-ahead).
    Indexed by date; consumers map each 15m bar's date onto it with ffill."""
    sma = daily_close.rolling(sma_period).mean()
    flag = (daily_close > sma).shift(1)
    out = flag.copy()
    out.index = pd.Index([ts.date() for ts in flag.index], name="date")
    return out


def _map_dates(flag_by_date: pd.Series, bar_dates: pd.Index) -> np.ndarray:
    """Map per-date flags onto every 15m bar date, forward-filling weekends/
    holidays. Unknown (warmup NaN) => False: no signal without evidence."""
    all_dates = flag_by_date.index.union(pd.Index(sorted(set(bar_dates))))
    filled = flag_by_date.reindex(all_dates).ffill()
    return filled.reindex(pd.Index(bar_dates)).fillna(False).to_numpy(dtype=bool)


def build_signals_daily(daily: pd.DataFrame, regime_by_date: pd.Series,
                        p: StrategyParams = StrategyParams()) -> pd.DataFrame:
    """Same signal logic and output schema as build_signals, but on DAILY bars
    with multi-day holds (swing timeframe). Signals evaluate on day t's close;
    the backtester fills at day t+1's open. The SPY regime flag is lagged one
    day (daily_flag_by_date already shifts); own-trend uses day t's close —
    both known at decision time."""
    close = daily["Close"]
    fast = close.rolling(p.fast_sma).mean()
    slow = close.rolling(p.slow_sma).mean()
    sma200 = close.rolling(p.daily_trend_sma).mean()
    rsi_slow = wilder_rsi(close, p.rsi_slow)
    rsi_fast = wilder_rsi(close, p.rsi_fast)

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    bar_dates = pd.Index([ts.date() for ts in daily.index])
    regime_on = _map_dates(regime_by_date, bar_dates)
    uptrend = (close > sma200).fillna(False).to_numpy()

    entry_mom = (cross_up & (close > slow) & (rsi_slow < p.rsi_slow_max)
                 ).to_numpy() & regime_on
    entry_mr = (rsi_fast < p.rsi_fast_entry).to_numpy() & uptrend & regime_on

    out = pd.DataFrame({
        "open": daily["Open"], "high": daily["High"],
        "low": daily["Low"], "close": close,
        "entry_mom": entry_mom,
        "exit_mom": cross_dn.to_numpy(),
        "entry_mr": entry_mr,
        "exit_mr": (rsi_fast > p.rsi_fast_exit).to_numpy(),
        "mom_rank": (close / slow - 1).to_numpy(),
        "mr_rank": (-rsi_fast).to_numpy(),
    }, index=daily.index)
    return out.dropna(subset=["open", "high", "low", "close"])


def build_signals(m15: pd.DataFrame, daily: pd.DataFrame,
                  regime_by_date: pd.Series,
                  p: StrategyParams = StrategyParams()) -> pd.DataFrame:
    """Per-ticker signal frame on the 15m timeline.

    Returns columns: open, high, low, close, entry_mom, exit_mom, entry_mr,
    exit_mr, mom_rank, mr_rank. Signals are evaluated on bar close; the
    backtester fills at the NEXT bar's open.
    """
    m15 = session_filter(m15)
    close = m15["Close"]

    fast = close.rolling(p.fast_sma).mean()
    slow = close.rolling(p.slow_sma).mean()
    rsi_slow = wilder_rsi(close, p.rsi_slow)
    rsi_fast = wilder_rsi(close, p.rsi_fast)

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    bar_dates = pd.Index([ts.date() for ts in m15.index])
    regime_on = _map_dates(regime_by_date, bar_dates)
    uptrend = _map_dates(
        daily_flag_by_date(daily["Close"], p.daily_trend_sma), bar_dates)

    entry_mom = (cross_up & (close > slow) & (rsi_slow < p.rsi_slow_max)
                 ).to_numpy() & regime_on
    entry_mr = ((rsi_fast < p.rsi_fast_entry).to_numpy()
                & uptrend & regime_on)

    out = pd.DataFrame({
        "open": m15["Open"], "high": m15["High"],
        "low": m15["Low"], "close": close,
        "entry_mom": entry_mom,
        "exit_mom": cross_dn.to_numpy(),
        "entry_mr": entry_mr,
        "exit_mr": (rsi_fast > p.rsi_fast_exit).to_numpy(),
        # rank: when more signals fire than free slots, take the strongest
        "mom_rank": (close / slow - 1).to_numpy(),   # most extended trend
        "mr_rank": (-rsi_fast).to_numpy(),           # most oversold
    }, index=m15.index)
    return out.dropna(subset=["open", "high", "low", "close"])
