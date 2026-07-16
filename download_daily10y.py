"""
10-year daily bars (Yahoo, split/dividend-adjusted) -> data/daily10y/
Reuses data.py's resumable chunked downloader. Re-run to fill gaps.
"""
from pathlib import Path

from data import _download_chunked, universe

OUT = Path(__file__).parent / "data" / "daily10y"

if __name__ == "__main__":
    tickers = universe()
    print(f"Universe: {len(tickers)} tickers, 10y daily", flush=True)
    saved, failed = _download_chunked(tickers, OUT, period="10y",
                                      interval="1d", min_bars=400)
    print(f"daily10y: +{saved} saved, {len(failed)} failed/sparse: "
          f"{sorted(failed)[:20]}")
