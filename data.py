"""
Data layer for the intraday bot.

Downloads and caches (as parquet, one file per ticker):
  * 15-minute bars, last 60 days  -> data/m15/     (Yahoo's free intraday limit)
  * daily bars, last 2 years      -> data/daily/   (for 200-day trend + SPY regime)

Resumable: tickers with an existing cache file are skipped, so a crashed or
rate-limited run can simply be re-run. Delete data/ to force a full refresh.

Usage:
    python data.py            # download everything missing
    python data.py --status   # show cache coverage only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
M15_DIR = ROOT / "data" / "m15"
DAILY_DIR = ROOT / "data" / "daily"

CHUNK_SIZE = 40
CHUNK_PAUSE_S = 2       # be polite to Yahoo; avoids rate-limit bans
MIN_15M_BARS = 300      # ~12 sessions; anything less is too sparse to test
MIN_DAILY_BARS = 250    # need a 200-day SMA plus headroom


def universe() -> list[str]:
    u = json.loads((ROOT / "universe.json").read_text())
    tickers = list(dict.fromkeys(u["sp500"]))  # dedupe, keep order
    if "SPY" not in tickers:
        tickers.append("SPY")                  # regime symbol + benchmark
    return tickers


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    return df


def _download_chunked(tickers: list[str], out_dir: Path, *, period: str,
                      interval: str, min_bars: int) -> tuple[int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [t for t in tickers if not (out_dir / f"{t}.parquet").exists()]
    print(f"[{interval}] {len(tickers) - len(missing)} cached, "
          f"{len(missing)} to download", flush=True)

    saved, failed = 0, []
    for i in range(0, len(missing), CHUNK_SIZE):
        chunk = missing[i:i + CHUNK_SIZE]
        for attempt in (1, 2):
            try:
                raw = yf.download(
                    tickers=chunk, period=period, interval=interval,
                    group_by="ticker", auto_adjust=True,
                    progress=False, threads=True,
                )
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"  chunk {i // CHUNK_SIZE}: download failed twice: {exc}",
                          flush=True)
                    raw = None
                else:
                    time.sleep(10)
        if raw is None or raw.empty:
            failed.extend(chunk)
            continue

        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = _clean(df)
            except KeyError:
                failed.append(t)
                continue
            if len(df) < min_bars:
                failed.append(t)
                continue
            df.to_parquet(out_dir / f"{t}.parquet")
            saved += 1
        print(f"  chunk {i // CHUNK_SIZE + 1}/"
              f"{(len(missing) + CHUNK_SIZE - 1) // CHUNK_SIZE}: "
              f"saved so far {saved}, failed {len(failed)}", flush=True)
        time.sleep(CHUNK_PAUSE_S)
    return saved, failed


def status() -> None:
    tickers = universe()
    for name, d in (("15m", M15_DIR), ("daily", DAILY_DIR)):
        have = {p.stem for p in d.glob("*.parquet")} if d.exists() else set()
        print(f"{name}: {len(have)}/{len(tickers)} cached")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        status()
        return

    tickers = universe()
    print(f"Universe: {len(tickers)} tickers (incl. SPY)", flush=True)

    t0 = time.time()
    n15, fail15 = _download_chunked(tickers, M15_DIR, period="60d",
                                    interval="15m", min_bars=MIN_15M_BARS)
    nd, faild = _download_chunked(tickers, DAILY_DIR, period="2y",
                                  interval="1d", min_bars=MIN_DAILY_BARS)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"15m:   +{n15} saved, {len(fail15)} failed/sparse: {fail15[:15]}")
    print(f"daily: +{nd} saved, {len(faild)} failed/sparse: {faild[:15]}")
    if "SPY" in fail15 or "SPY" in faild:
        print("FATAL: SPY data missing — regime filter and benchmark impossible.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
