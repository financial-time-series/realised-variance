"""Auxiliary inputs for the benchmarks added in revision. LAPTOP.

    python scripts/build_extra.py                    # downloads VIX from FRED
    python scripts/build_extra.py --vix-csv VIXCLS.csv

Writes two files beside the panels, both committed:

  data/processed/vix.parquet         ds, vix, iv_daily
      The CBOE VIX close (FRED series VIXCLS), and the implied DAILY variance it
      represents, (VIX/100)^2 / 252, in the same units as the realised-variance
      target. The input of HAR-log-IV and of the hybrid's equity variant.

  data/processed/eq_returns.parquet  unique_id, ds, ret_oc, ret_on
      Open-to-close and overnight (previous close to open) log returns, for
      exactly the rows of eq_rv.parquet. The Garman-Klass target measures
      trading-hours variance and so excludes the overnight move; a GARCH fitted
      to close-to-close returns forecasts a different, larger quantity. GARCH
      on open-to-close returns is the like-for-like variant.

It also adds ONE column to data/processed/eq_rv.parquet, leaving every
existing column bit-identical (checked before writing):

  rv_rsf   Rogers-Satchell with its degenerate values floored. The estimator is
           exactly zero on 0.34% of equity days -- whenever the open or the close
           coincides with the day's high or low -- and below 1% of its series
           median on a further 0.02%, values as small as 3e-19; neither
           Garman-Klass nor Parkinson ever falls below 1% of its median. No
           log-based input or QLIKE target can take a zero, so values below 1% of
           the series median are set to that level, and the robustness runs keep
           the same calendar as every other run.
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
            "?id=VIXCLS&cosd=1999-01-01&coed=2026-07-31")


def build_vix(csv: Path | None) -> pd.DataFrame:
    if csv is None:
        import requests
        r = requests.get(FRED_URL, timeout=60, headers={"User-Agent": "data-fetch/0.1"})
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text))
        print(f"  downloaded VIXCLS from FRED on {date.today()}")
    else:
        raw = pd.read_csv(csv)
    raw.columns = ["ds", "vix"]
    raw["ds"] = pd.to_datetime(raw["ds"])
    # FRED marks holidays with "." or leaves them empty: no quote that day.
    raw["vix"] = pd.to_numeric(raw["vix"], errors="coerce")
    vix = raw.dropna().reset_index(drop=True)
    vix = vix[(vix.ds <= "2026-07-31")]
    vix["iv_daily"] = (vix["vix"] / 100.0) ** 2 / 252.0
    return vix


def build_eq_returns() -> pd.DataFrame:
    raw = pd.read_parquet("data/raw/eq_ohlcv.parquet")[
        ["unique_id", "ds", "open", "close"]].copy()
    raw["ds"] = pd.to_datetime(raw["ds"])
    raw = raw.sort_values(["unique_id", "ds"])
    raw["ret_oc"] = np.log(raw["close"] / raw["open"])
    raw["ret_on"] = np.log(raw["open"] / raw.groupby("unique_id")["close"].shift(1))
    panel = pd.read_parquet(PROCESSED / "eq_rv.parquet")[["unique_id", "ds"]]
    panel["ds"] = pd.to_datetime(panel["ds"])
    out = panel.merge(raw[["unique_id", "ds", "ret_oc", "ret_on"]],
                      on=["unique_id", "ds"], how="left")
    missing = int(out["ret_oc"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} panel rows have no open-to-close return")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vix-csv", type=Path, default=None)
    args = ap.parse_args()

    vix = build_vix(args.vix_csv)
    vix.to_parquet(PROCESSED / "vix.parquet", index=False)
    print(f"wrote {PROCESSED / 'vix.parquet'}: {len(vix):,} days, "
          f"{vix.ds.min().date()} -> {vix.ds.max().date()}")

    eq_dates = pd.to_datetime(pd.read_parquet(PROCESSED / "eq_rv.parquet")["ds"]).unique()
    uncovered = np.setdiff1d(eq_dates, vix["ds"].to_numpy())
    print(f"  equity trading days without a VIX close: {len(uncovered)} "
          f"(forward-filled by the models)")

    panel_path = PROCESSED / "eq_rv.parquet"
    panel = pd.read_parquet(panel_path)
    before = panel.drop(columns=["rv_rsf"], errors="ignore").copy()
    med = panel[panel.rv_rs > 0].groupby("unique_id")["rv_rs"].median()
    floor = 0.01 * panel["unique_id"].map(med)
    panel["rv_rsf"] = np.maximum(panel["rv_rs"], floor)
    n_zero = int((panel["rv_rs"] <= 0).sum())
    n_tiny = int(((panel["rv_rs"] > 0) & (panel["rv_rs"] < floor)).sum())
    pd.testing.assert_frame_equal(before, panel.drop(columns=["rv_rsf"]))
    panel.to_parquet(panel_path, index=False)
    print(f"added rv_rsf to {panel_path}: {n_zero} zero ({n_zero / len(panel):.2%}) and "
          f"{n_tiny} near-zero ({n_tiny / len(panel):.2%}) Rogers-Satchell days floored "
          f"at 1% of the series median")

    ret = build_eq_returns()
    ret.to_parquet(PROCESSED / "eq_returns.parquet", index=False)
    print(f"wrote {PROCESSED / 'eq_returns.parquet'}: {len(ret):,} rows")
    m = pd.read_parquet(PROCESSED / "eq_rv.parquet").merge(ret, on=["unique_id", "ds"])
    g = m.groupby("unique_id").agg(cc=("ret", lambda x: (x ** 2).mean()),
                                   on=("ret_on", lambda x: (x.dropna() ** 2).mean()),
                                   gk=("rv", "mean"))
    print(f"  median series: Garman-Klass mean / close-to-close mean square = "
          f"{(g.gk / g.cc).median():.3f}; overnight share = {(g.on / g.cc).median():.3f}")

    # The facts the Data section quotes, written where check_limits.py looks.
    raw = pd.read_parquet("data/raw/eq_ohlcv.parquet")
    per = raw.groupby("unique_id")["ds"].count()
    eqp = pd.read_parquet(PROCESSED / "eq_rv.parquet")
    facts = {
        "eq_overnight_share_median": (g.on / g.cc).median(),
        "eq_gk_over_cc_median": (g.gk / g.cc).median(),
        "eq_raw_rows": len(raw), "eq_complete_history_rows": int(per.max()),
        "eq_shortfall_per_series": float(per.max() - len(raw) / per.size),
        "eq_rs_zero_days": n_zero, "eq_rs_zero_share": n_zero / len(eqp),
        "eq_rs_tiny_days": n_tiny, "eq_rs_tiny_share": n_tiny / len(eqp),
        "vix_days": len(vix),
    }
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"fact": k, "value": v} for k, v in facts.items()]).to_csv(
        "results/tables/data_facts.csv", index=False)
    print("wrote results/tables/data_facts.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
