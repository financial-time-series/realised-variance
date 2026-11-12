"""Download 5-minute klines for the CX universe from the Binance bulk archive.

    python scripts/fetch_cx.py
    python scripts/fetch_cx.py --tickers BTCUSDT --start 2024-01

`data.binance.vision` serves monthly zips and answers 200 from BOTH the laptop
and the cluster login node (verified this session), so this one could run
either side. It runs here for the same reason as fetch_eq.py: one place owns
`data/raw/`, and rsync moves it out.

WHY 5-MINUTE. The realised-variance literature settles around 5 minutes as the
sampling frequency where microstructure noise is tolerable and the estimator is
still efficient. Going to 1-minute would triple the download and bias RV upward
through bid-ask bounce; going to 30-minute would throw away most of the signal.

MISSING BARS ARE NOT ZERO. A UTC day with fewer than the expected 288 bars is
an exchange outage or a listing boundary, not a calm day. Days are counted here
and incomplete ones are dropped downstream in build_panel.py rather than
silently summed over whatever arrived.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finlab.universe import (  # noqa: E402
    CX_END, CX_INTERVAL, CX_START, CX_TICKERS,
)

BASE = "https://data.binance.vision/data/spot/monthly/klines"
RAW_DIR = Path("data/raw/cx")

# Binance kline columns, in the archive's fixed order.
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]


def months(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m") for d in
            pd.date_range(f"{start}-01", f"{end}-01", freq="MS")]


def fetch_month(session: requests.Session, ticker: str, month: str) -> pd.DataFrame | None:
    """One monthly zip. 404 means the pair did not trade yet -- not an error."""
    url = f"{BASE}/{ticker}/{CX_INTERVAL}/{ticker}-{CX_INTERVAL}-{month}.zip"
    try:
        r = session.get(url, timeout=90)
    except Exception as exc:  # noqa: BLE001
        print(f"  {ticker} {month}: {type(exc).__name__} -- skipped")
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        print(f"  {ticker} {month}: HTTP {r.status_code} -- skipped")
        return None

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            head = fh.read(64)
        # Newer archives ship a CSV header row; older ones do not. Detect
        # rather than assume -- guessing wrong silently drops a bar or turns
        # the whole month into NaN.
        has_header = head.lstrip()[:9].lower().startswith(b"open_time")
        with z.open(name) as fh:
            df = pd.read_csv(fh, header=0 if has_header else None,
                             names=None if has_header else KLINE_COLS)

    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    # open_time is epoch. Binance switched these archives from milliseconds to
    # microseconds during 2025; discriminate on magnitude instead of trusting
    # a fixed unit, or every bar after the switch lands in the year 57000.
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df["ts"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    return df.drop(columns=["open_time"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=CX_TICKERS)
    ap.add_argument("--start", default=CX_START)
    ap.add_argument("--end", default=CX_END)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    all_months = months(args.start, args.end)

    for ticker in args.tickers:
        out = RAW_DIR / f"{ticker}.parquet"
        if out.exists() and not args.force:
            n = len(pd.read_parquet(out, columns=["ts"]))
            print(f"{ticker:<9} cached ({n:,} bars)")
            continue

        print(f"{ticker:<9} downloading {len(all_months)} months...", flush=True)
        parts, first, last = [], None, None
        for month in all_months:
            df = fetch_month(session, ticker, month)
            if df is None:
                continue
            parts.append(df)
            first = first or month
            last = month
            time.sleep(0.05)

        if not parts:
            print(f"  {ticker}: no data in range -- skipped")
            continue

        panel = pd.concat(parts, ignore_index=True).sort_values("ts")
        panel = panel.drop_duplicates("ts").reset_index(drop=True)
        panel.to_parquet(out, index=False)
        print(f"  {ticker}: {len(panel):,} bars  {first} -> {last}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
