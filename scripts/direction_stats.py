"""How often each model forecasts a rise in variance, against how often one occurs.

    .venv/bin/python scripts/direction_stats.py --dataset eq

Behind the directional loss (MADL). For every model on the leakage-controlled
window (2024 generation and baselines, from 2025-01-01) and each horizon: the
share of forecasts above the "no change" reference b (the realised variance
averaged over the h days up to the origin), the share of outcomes above it, and
the share of origins on which the forecast called the direction right. A
forecast of the conditional MEAN of a right-skewed variable -- HAR-log with its
lognormal correction -- sits above the typical outcome and so calls rises more
often than they occur; a forecast of the typical outcome does not. Writes
results/tables/direction_<dataset>.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finlab.io import load_campaign  # noqa: E402
from finlab.zoo import GENERATION_WINDOWS, ZOO  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--out-root", default="out/runs")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    args = ap.parse_args()

    from aggregate import label, panel_targets, restrict_to_common_sample
    df = load_campaign(args.out_root, dataset=args.dataset, rv_col="rv")
    df["label"] = label(df)
    df["origin"] = pd.to_datetime(df["origin"])
    df = restrict_to_common_sample(df, args.min_coverage)
    df = df[df.origin >= np.datetime64(GENERATION_WINDOWS["g2024"])]
    adm = df.label.map(lambda l: l.split("[")[0] not in ZOO
                       or ZOO[l.split("[")[0]].generation == "g2024")
    df = df[adm]
    tg = panel_targets(Path(f"data/processed/{args.dataset}_rv.parquet"), "rv")
    df = df.merge(tg[["unique_id", "origin", "h", "base"]], on=["unique_id", "origin", "h"])
    df["f_up"] = df.y_hat > df.base
    df["y_up"] = df.y > df.base
    df["right"] = np.sign(np.log(df.y_hat / df.base)) == np.sign(np.log(df.y / df.base))
    out = (df.groupby(["label", "h"])[["f_up", "y_up", "right"]].mean().reset_index()
             .rename(columns={"f_up": "share_forecast_rise", "y_up": "share_rise",
                              "right": "share_direction_right"}))
    out.insert(0, "dataset", args.dataset)
    path = Path("results/tables") / f"direction_{args.dataset}.csv"
    out.to_csv(path, index=False)
    print(out[out.h == 1].round(3).to_string(index=False))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
