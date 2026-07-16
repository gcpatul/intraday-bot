"""
Alpaca historical data downloader — the long-history upgrade over Yahoo's
60-day limit (DAY_TRADING_BACKTEST_PLAN Phase 0).

Downloads and caches (one parquet per ticker, Yahoo-style column names so
strategies.py works unchanged):
  * 15-minute bars, ~2 years  -> data/alpaca_m15/
  * daily bars,    ~3 years   -> data/alpaca_daily/   (200-SMA warmup + regime)

Free (Basic) plan notes, baked in:
  * Tries the server-default feed first; on a subscription error falls back
    to feed=IEX explicitly. IEX bars = IEX-only trades — sparser than
    consolidated tape for less-liquid names; MIN_* thresholds filter those out.
  * ~200 requests/min rate limit: batched symbols, backoff on 429.
  * adjustment=ALL (splits + dividends), matching Yahoo auto_adjust=True.

Resumable: cached tickers are skipped; just re-run after any failure.

Credentials: ALPACA_API_KEY / ALPACA_SECRET_KEY from the environment, or from
a local .env file next to this script (KEY=value lines). Values are never
printed.

Usage:
    python alpaca_data.py             # download everything missing
    python alpaca_data.py --status    # show cache coverage
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

ROOT = Path(__file__).parent
M15_DIR = ROOT / "data" / "alpaca_m15"
DAILY_DIR = ROOT / "data" / "alpaca_daily"

YEARS_15M = 2
YEARS_DAILY = 3
BATCH = 25
SLICE_DAYS_15M = 90     # request 15m data in ~3-month windows: short-lived
                        # requests survive flaky connections; a failed slice
                        # retries alone instead of restarting a giant batch
MIN_15M_BARS = 2000     # ~2y of 15m for a liquid name is ~13k; <2k = too sparse
MIN_DAILY_BARS = 400


def load_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def universe() -> list[str]:
    u = json.loads((ROOT / "universe.json").read_text())
    tickers = list(dict.fromkeys(u["sp500"]))
    if "SPY" not in tickers:
        tickers.append("SPY")
    # Alpaca uses dots for share classes where Yahoo uses dashes
    return [t.replace("-", ".") for t in tickers]


def client() -> StockHistoricalDataClient:
    load_env()
    key = os.environ.get("ALPACA_API_KEY")
    # chawani's .env convention is ALPACA_API_SECRET; accept both spellings
    sec = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")
    if not key or not sec:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set "
                         "(env or .env file). Refusing to start.")
    return StockHistoricalDataClient(key, sec)


def _fetch_slice(cli, symbols: list[str], timeframe, start, end) -> pd.DataFrame | None:
    """One time-sliced batched request with feed fallback + backoff."""
    for feed in (None, DataFeed.IEX):
        kwargs = {"feed": feed} if feed else {}
        req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe,
                               start=start, end=end,
                               adjustment=Adjustment.ALL, **kwargs)
        for attempt in range(4):
            try:
                return cli.get_stock_bars(req).df
            except Exception as exc:
                msg = str(exc).lower()
                if "subscription" in msg or "forbidden" in msg:
                    break               # retry outer loop with explicit IEX
                if "too many" in msg or "429" in msg or "rate" in msg:
                    wait = 15 * (attempt + 1)
                    print(f"    rate limited, sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                print(f"    slice error ({type(exc).__name__}), "
                      f"retry {attempt + 1}/4", flush=True)
                time.sleep(5)
        else:
            continue
    return None


def _fetch_batch(cli, symbols: list[str], timeframe, start,
                 slice_days: int | None) -> pd.DataFrame | None:
    """Fetch [start..now] for a symbol batch, in time slices if requested."""
    end = datetime.now(timezone.utc) - timedelta(minutes=16)  # free-tier embargo
    if not slice_days:
        return _fetch_slice(cli, symbols, timeframe, start, end)
    parts = []
    s = start
    while s < end:
        e = min(s + timedelta(days=slice_days), end)
        df = _fetch_slice(cli, symbols, timeframe, s, e)
        if df is not None and not df.empty:
            parts.append(df)
        s = e
    return pd.concat(parts) if parts else None


def _download(cli, tickers: list[str], out_dir: Path, timeframe, years: int,
              min_bars: int, label: str,
              slice_days: int | None = None) -> tuple[int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [t for t in tickers
               if not (out_dir / f"{t.replace('.', '-')}.parquet").exists()]
    print(f"[{label}] {len(tickers) - len(missing)} cached, "
          f"{len(missing)} to download", flush=True)
    start = datetime.now(timezone.utc) - timedelta(days=int(365.25 * years))

    saved, failed = 0, []
    for i in range(0, len(missing), BATCH):
        chunk = missing[i:i + BATCH]
        raw = _fetch_batch(cli, chunk, timeframe, start, slice_days)
        if raw is None or raw.empty:
            failed.extend(chunk)
            print(f"  batch {i // BATCH + 1}: EMPTY/failed", flush=True)
            continue
        for t in chunk:
            try:
                df = raw.xs(t, level="symbol")
            except KeyError:
                failed.append(t)
                continue
            df = df.rename(columns={"open": "Open", "high": "High",
                                    "low": "Low", "close": "Close",
                                    "volume": "Volume"})
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(
                subset=["Close"]).sort_index()
            df = df[~df.index.duplicated(keep="first")]  # slice-boundary dupes
            df = df[df["Close"] > 0]
            if len(df) < min_bars:
                failed.append(t)
                continue
            # store under the Yahoo-style name so the rest of the stack matches
            df.to_parquet(out_dir / f"{t.replace('.', '-')}.parquet")
            saved += 1
        print(f"  batch {i // BATCH + 1}/"
              f"{(len(missing) + BATCH - 1) // BATCH}: "
              f"saved {saved}, failed {len(failed)}", flush=True)
    return saved, failed


def status() -> None:
    n = len(universe())
    for name, d in (("15m", M15_DIR), ("daily", DAILY_DIR)):
        have = len(list(d.glob("*.parquet"))) if d.exists() else 0
        print(f"alpaca {name}: {have}/{n} cached")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        status()
        return

    cli = client()
    tickers = universe()
    print(f"Universe: {len(tickers)} tickers | 15m x {YEARS_15M}y, "
          f"daily x {YEARS_DAILY}y", flush=True)
    t0 = time.time()
    n15, f15 = _download(cli, tickers, M15_DIR,
                         TimeFrame(15, TimeFrameUnit.Minute), YEARS_15M,
                         MIN_15M_BARS, "15m", slice_days=SLICE_DAYS_15M)
    nd, fd = _download(cli, tickers, DAILY_DIR, TimeFrame.Day, YEARS_DAILY,
                       MIN_DAILY_BARS, "daily")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(f"15m:   +{n15}, failed/sparse {len(f15)}: {sorted(f15)[:15]}")
    print(f"daily: +{nd}, failed/sparse {len(fd)}: {sorted(fd)[:15]}")
    if not (M15_DIR / "SPY.parquet").exists() or not (DAILY_DIR / "SPY.parquet").exists():
        print("FATAL: SPY missing — no regime/benchmark possible.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
