"""Why a per-date median can still fail to reject: failures shared across assets.

    .venv/bin/python scripts/daily_medians.py --dataset cx

For HAR-log and each zero-shot variant, on the leakage-controlled window (the
2024 generation, from 2025-01-01) and at h=1, the QLIKE of every asset is
summarised by its median across assets on each date. Writes
results/tables/daily_<dataset>.csv with, per variant: the pooled median (over
all assets and dates), the average over dates of the per-date median, the share
of dates on which the variant's per-date median is below HAR-log's, the share of
its positive per-date differential carried by the worst 5% of dates, and the
worst date. A model that fails on the same dates for most assets at once has a
lower pooled median and a higher average per-date median at the same time, and
then no test built on per-date summaries can find in its favour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finlab.evaluation import qlike  # noqa: E402
from finlab.io import load_campaign  # noqa: E402
from finlab.zoo import GENERATION_WINDOWS, ZOO  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--out-root", default="out/runs")
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    args = ap.parse_args()

    from aggregate import label, restrict_to_common_sample
    df = load_campaign(args.out_root, dataset=args.dataset, rv_col="rv")
    df["label"] = label(df)
    df["origin"] = pd.to_datetime(df["origin"])
    df = restrict_to_common_sample(df, args.min_coverage)
    df = df[(df.h == args.h) & (df.origin >= np.datetime64(GENERATION_WINDOWS["g2024"]))]
    keep = df.label.eq("har-log") | df.label.map(
        lambda l: l.split("[")[0] in ZOO and ZOO[l.split("[")[0]].generation == "g2024")
    df = df[keep].assign(loss=lambda d: qlike(d.y.to_numpy(), d.y_hat.to_numpy()))
    daily = df.groupby(["label", "origin"])["loss"].median().unstack("label")
    bench = daily["har-log"]
    rows = []
    for lab in daily.columns:
        diff = daily[lab] - bench
        pos = diff[diff > 0].sort_values(ascending=False)
        top = pos.head(max(int(len(diff) * 0.05), 1)).sum() / pos.sum() if pos.sum() > 0 else np.nan
        rows.append({"dataset": args.dataset, "h": args.h, "model": lab,
                     "pooled_median": float(df[df.label == lab]["loss"].median()),
                     "mean_daily_median": float(daily[lab].mean()),
                     "share_dates_below_bench": float((diff < 0).mean()) if lab != "har-log" else np.nan,
                     "share_top5pct_dates": float(top) if lab != "har-log" else np.nan,
                     "worst_date": str(diff.idxmax())[:10] if lab != "har-log" else "",
                     "worst_daily_median": float(daily[lab].max()),
                     "n_dates": int(len(diff))})
    out = pd.DataFrame(rows)
    path = Path("results/tables") / f"daily_{args.dataset}.csv"
    out.to_csv(path, index=False)
    print(out.round(4).to_string(index=False))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
