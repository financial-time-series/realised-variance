"""Download daily OHLCV for the EQ universe. LAPTOP ONLY.

Yahoo returns 429 to the cluster (verified this session, see cluster.local.md),
so this runs here and the result is rsync'd out. That is the normal case for
financial data and not worth fighting.

    python scripts/fetch_eq.py                 # incremental; skips cached
    python scripts/fetch_eq.py --force         # re-download everything
    python scripts/fetch_eq.py --tickers SPY AAPL

One parquet per ticker under data/raw/eq/, so an interrupted run resumes and a
single bad ticker never costs the other forty-four. The combined panel is
rebuilt from those files at the end and is the only artefact downstream code
reads.

AUTO-ADJUSTMENT. Prices are split- and dividend-adjusted. For range-based
volatility this matters in one direction only: an unadjusted split day shows a
50% "range" that is pure corporate action. Adjustment scales O/H/L/C by the same
daily factor, so the within-day log ratios the estimators use are untouched.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finlab.universe import EQ_END, EQ_START, EQ_TICKERS, eq_group  # noqa: E402

RAW_DIR = Path("data/raw/eq")
PANEL = Path("data/raw/eq_ohlcv.parquet")


def fetch_one(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    """One ticker, with backoff. Returns None if Yahoo will not serve it."""
    import yfinance as yf

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
            )
            if df is None or df.empty:
                raise ValueError("empty frame")
            # yfinance returns MultiIndex columns even for a single ticker in
            # recent versions. Flatten to the level that holds OHLCV names.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df.index.name = "ds"
            return df.reset_index()
        except Exception as exc:  # noqa: BLE001 - one ticker must not kill the run
            wait = 5 * attempt
            print(f"  {ticker}: attempt {attempt}/{retries} failed "
                  f"({type(exc).__name__}: {exc}) -- retrying in {wait}s")
            time.sleep(wait)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=EQ_TICKERS)
    ap.add_argument("--start", default=EQ_START)
    ap.add_argument("--end", default=EQ_END)
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="pause between tickers; Yahoo rate-limits hard")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for i, ticker in enumerate(args.tickers, 1):
        path = RAW_DIR / f"{ticker}.parquet"
        if path.exists() and not args.force:
            print(f"[{i:>2}/{len(args.tickers)}] {ticker:<6} cached")
            continue
        print(f"[{i:>2}/{len(args.tickers)}] {ticker:<6} downloading...", flush=True)
        df = fetch_one(ticker, args.start, args.end)
        if df is None:
            failed.append(ticker)
            print(f"  {ticker}: GIVING UP")
            continue
        df.to_parquet(path, index=False)
        print(f"  {ticker}: {len(df):,} rows  {df['ds'].min().date()} -> {df['ds'].max().date()}")
        time.sleep(args.sleep)

    # --- rebuild the combined panel from whatever is cached ------------------
    parts = []
    for path in sorted(RAW_DIR.glob("*.parquet")):
        ticker = path.stem
        df = pd.read_parquet(path)
        df["unique_id"] = ticker
        df["group"] = eq_group(ticker)
        parts.append(df)

    if not parts:
        print("nothing downloaded")
        return 1

    panel = pd.concat(parts, ignore_index=True)
    panel["ds"] = pd.to_datetime(panel["ds"]).dt.tz_localize(None).dt.normalize()
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL, index=False)

    print(f"\nwrote {PANEL}: {len(panel):,} rows, "
          f"{panel['unique_id'].nunique()} tickers, "
          f"{panel['ds'].min().date()} -> {panel['ds'].max().date()}")
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
