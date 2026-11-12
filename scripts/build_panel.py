"""Turn raw downloads into the two tidy realised-variance panels.

    python scripts/build_panel.py --dataset eq
    python scripts/build_panel.py --dataset cx

Writes data/processed/<dataset>_rv.parquet and a filter-chain report to
results/tables/filters_<dataset>.csv. The report is not bookkeeping: a referee
cannot judge selection effects from a final observation count alone, so the
paper prints how many series and days survive each step (README section 6).

THE TARGET. Daily realised variance, in squared-log-return units. Every model
forecasts the same object and is scored on the same object; whether a model
works internally in RV, log RV or sqrt RV is its own business, and its
predictions are converted back to the RV level before they are written.

    EQ  realised variance is a RANGE-BASED PROXY from daily OHLC.
        Garman-Klass is the primary estimator; Parkinson and the squared
        close-to-close return are carried alongside for robustness.
        This is a proxy, not a measurement, and the paper says so. What makes
        it defensible is (a) Patton's result that QLIKE and MSE rank forecasts
        consistently even under a noisy but conditionally unbiased proxy, and
        (b) the CX panel, where RV is measured from actual intraday returns.

    CX  realised variance MEASURED as the sum of 288 squared 5-minute log
        returns over a complete UTC day. No proxy assumption.

WHY VARIANCE AND NOT VOLATILITY. QLIKE is defined on variance. Scoring sqrt(RV)
under a loss designed for RV silently changes which forecast is optimal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finlab.universe import CX_BARS_PER_DAY  # noqa: E402

PROCESSED = Path("data/processed")
TABLES = Path("results/tables")

MIN_OBS_EQ = 2000     # ~8 years of trading days
MIN_OBS_CX = 1000     # ~2.7 years of calendar days


class FilterLog:
    """Records what each filter removed, in order, for the paper's sample table."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.rows: list[dict] = []

    def record(self, step: str, df: pd.DataFrame) -> None:
        self.rows.append({
            "step": step,
            "series": df["unique_id"].nunique(),
            "observations": len(df),
        })
        print(f"  {step:<44} {df['unique_id'].nunique():>4} series  {len(df):>9,} obs")

    def write(self) -> None:
        TABLES.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame(self.rows)
        out["dropped"] = out["observations"].diff().fillna(0).astype(int) * -1
        path = TABLES / f"filters_{self.dataset}.csv"
        out.to_csv(path, index=False)
        print(f"\nwrote {path}")


# --- EQ ----------------------------------------------------------------------

def build_eq(log: FilterLog) -> pd.DataFrame:
    src = Path("data/raw/eq_ohlcv.parquet")
    if not src.exists():
        raise FileNotFoundError(f"{src} -- run scripts/fetch_eq.py first")
    df = pd.read_parquet(src)
    log.record("downloaded", df)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    log.record("drop rows with missing OHLC", df)

    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    log.record("drop non-positive prices", df)

    # Internal consistency. A bar where the close sits outside [low, high], or
    # the high is below the low, is a vendor error; the range estimators would
    # happily return a negative variance from it.
    ok = (df["high"] >= df["low"]) & \
         (df["close"].between(df["low"], df["high"])) & \
         (df["open"].between(df["low"], df["high"]))
    df = df[ok]
    log.record("drop internally inconsistent bars", df)

    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    ho, lo = np.log(df["high"] / df["open"]), np.log(df["low"] / df["open"])
    hc, lc = np.log(df["high"] / df["close"]), np.log(df["low"] / df["close"])

    # Garman-Klass (1980): the efficient OHLC estimator, ~7x the efficiency of
    # squared close-to-close. Primary target.
    df["rv"] = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    # Parkinson (1980): range only. Robustness -- it ignores the open-close
    # move, so it reacts differently to trending days.
    df["rv_parkinson"] = hl**2 / (4 * np.log(2))
    # Rogers-Satchell (1991): drift-independent. Second robustness.
    df["rv_rs"] = ho * hc + lo * lc

    df = df.sort_values(["unique_id", "ds"])
    # Squared close-to-close return: the crudest proxy, and the one whose
    # conditional unbiasedness is least disputable. Third robustness.
    df["ret"] = df.groupby("unique_id")["close"].transform(
        lambda s: np.log(s).diff())
    df["rv_sqret"] = df["ret"] ** 2

    df = df.dropna(subset=["rv", "ret"])
    log.record("drop first day (no return) and NaN estimates", df)

    # Garman-Klass can go non-positive on a day where the close-open move
    # dominates a tiny range. QLIKE takes a log, so zero and negative variance
    # cannot be scored and must not be quietly floored.
    df = df[(df["rv"] > 0) & (df["rv_parkinson"] > 0)]
    log.record("drop non-positive variance estimates", df)

    counts = df.groupby("unique_id")["ds"].count()
    keep = counts[counts >= MIN_OBS_EQ].index
    df = df[df["unique_id"].isin(keep)]
    log.record(f"drop series with < {MIN_OBS_EQ} observations", df)

    return df[["unique_id", "ds", "group", "close", "ret",
               "rv", "rv_parkinson", "rv_rs", "rv_sqret"]]


# --- CX ----------------------------------------------------------------------

def build_cx(log: FilterLog) -> pd.DataFrame:
    raw_dir = Path("data/raw/cx")
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{raw_dir}/*.parquet -- run scripts/fetch_cx.py first")

    daily_parts, bar_rows = [], 0
    for path in files:
        bars = pd.read_parquet(path).sort_values("ts")
        bar_rows += len(bars)
        bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
        bars = bars.dropna(subset=["close"])
        bars = bars[bars["close"] > 0]

        # Log returns on the continuous close series, so the return spanning
        # midnight belongs to the day it ends in and a complete UTC day carries
        # exactly 288 of them -- full 24-hour coverage, no gap at the boundary.
        bars["r"] = np.log(bars["close"]).diff()
        bars["date"] = bars["ts"].dt.floor("D")

        g = bars.groupby("date")
        daily = pd.DataFrame({
            "rv": g["r"].apply(lambda s: np.sum(s.dropna() ** 2)),
            "n_bars": g["r"].apply(lambda s: s.notna().sum()),
            "close": g["close"].last(),
            # Realised quarticity: needed for the HARQ-style robustness and
            # nearly free to compute here.
            "rq": g["r"].apply(
                lambda s: len(s.dropna()) / 3.0 * np.sum(s.dropna() ** 4)),
        }).reset_index()
        daily["unique_id"] = path.stem
        daily_parts.append(daily)

    df = pd.concat(daily_parts, ignore_index=True).rename(columns={"date": "ds"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["group"] = "crypto"
    print(f"  ({bar_rows:,} 5-minute bars aggregated to daily)")
    log.record("aggregated to daily", df)

    # An incomplete day is an exchange outage or a listing boundary, not a calm
    # day. Summing over whatever arrived would report a spuriously low variance
    # on exactly the days most likely to be turbulent.
    df = df[df["n_bars"] == CX_BARS_PER_DAY]
    log.record(f"keep only complete UTC days ({CX_BARS_PER_DAY} bars)", df)

    df = df[df["rv"] > 0]
    log.record("drop zero-variance days", df)

    df = df.sort_values(["unique_id", "ds"])
    df["ret"] = df.groupby("unique_id")["close"].transform(lambda s: np.log(s).diff())

    counts = df.groupby("unique_id")["ds"].count()
    keep = counts[counts >= MIN_OBS_CX].index
    df = df[df["unique_id"].isin(keep)]
    log.record(f"drop series with < {MIN_OBS_CX} observations", df)

    # The CX panel measures RV directly, so the EQ robustness estimators do not
    # apply. rv_sqret is kept because it is the one proxy both panels share.
    df["rv_sqret"] = df["ret"] ** 2
    return df[["unique_id", "ds", "group", "close", "ret", "rv", "rq",
               "rv_sqret", "n_bars"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["eq", "cx"], required=True)
    args = ap.parse_args()

    print(f"=== building {args.dataset} panel ===")
    log = FilterLog(args.dataset)
    df = build_eq(log) if args.dataset == "eq" else build_cx(log)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"{args.dataset}_rv.parquet"
    df.to_parquet(out, index=False)
    log.write()

    print(f"\nwrote {out}")
    print(f"  {df['unique_id'].nunique()} series, {len(df):,} observations")
    print(f"  {df['ds'].min().date()} -> {df['ds'].max().date()}")
    print(f"  median annualised vol: "
          f"{np.sqrt(df['rv'].median() * 252) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
