"""
Download 10y daily bars for FORMER S&P members missing from data/daily10y —
the survivorship-bias fix for the point-in-time rotation backtest.

Names that were dropped from the index but still trade will download fine;
truly delisted names (acquired, bankrupt) fail — that residual gap is
reported and must be disclosed with any result. min_bars is low on purpose:
even a member with only ~6 months of history before delisting belongs in
the point-in-time ranking pool.
"""
import csv
from pathlib import Path

from data import _download_chunked

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "daily10y"

ALIASES = {"FB": "META", "ANTM": "ELV", "CTL": "LUMN", "KORS": "CPRI",
           "VIAC": "PARA", "WLTW": "WTW", "FISV": "FI", "PKI": "RVTY",
           "RE": "EG", "FLT": "CPAY"}

if __name__ == "__main__":
    rows = list(csv.reader(open(ROOT / "data" / "sp500_pit_membership.csv")))[1:]
    window_rows = [r for r in rows if r[0] >= "2016-07-01"]
    before = [r for r in rows if r[0] < "2016-07-01"]
    if before:
        window_rows.insert(0, before[-1])
    needed = set()
    for d, ticks in window_rows:
        needed |= {t.strip().replace(".", "-") for t in ticks.split(",")}
    have = {p.stem for p in OUT.glob("*.parquet")}
    missing = sorted(t for t in needed
                     if ALIASES.get(t, t) not in have)
    print(f"{len(missing)} former members to attempt", flush=True)
    saved, failed = _download_chunked(missing, OUT, period="10y",
                                      interval="1d", min_bars=100)
    print(f"+{saved} saved; {len(failed)} unavailable (delisted residual): "
          f"{sorted(failed)}")
