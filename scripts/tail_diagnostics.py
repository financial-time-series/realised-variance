"""How often does a model forecast a variance that is essentially zero?

    uv run --no-sync python scripts/tail_diagnostics.py --dataset cx

Separate from aggregate.py on purpose: this reads the raw predictions and is
cheap, whereas aggregate.py's bootstrap tests are not, and this question came
up after those had already been run.

WHY IT MATTERS. QLIKE contains y/f, so a forecast approaching zero costs an
unbounded amount. Fed RAW realised variance, several zero-shot checkpoints
occasionally emit a near-zero (or non-positive, hence floored) variance: on the
crypto panel TTM-r2 produced forecasts five orders of magnitude below the
smallest variance ever observed. A handful of those days dominate its mean
QLIKE completely.

That is a real deployment failure and belongs in the paper -- but the *size* of
the resulting mean depends on the floor constant, which is our choice, not the
model's. The honest summary is therefore not the mean alone but the FREQUENCY
of the event, which is floor-independent and is what this script measures. The
paper quotes the frequency and uses the median for central tendency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finlab.io import load_campaign  # noqa: E402
from finlab.zoo import GENERATION_WINDOWS  # noqa: E402

TABLES = Path("results/tables")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--out-root", default="out/runs")
    ap.add_argument("--rv-col", default="rv",
                    help="the main study's measure; the robustness runs share model "
                         "names and must not be mixed in")
    args = ap.parse_args()

    df = load_campaign(args.out_root, dataset=args.dataset, rv_col=args.rv_col)
    v = df["variant"].fillna("") if "variant" in df.columns else ""
    df["label"] = np.where(pd.Series(v).eq(""), df["model"],
                           df["model"] + "[" + pd.Series(v) + "]")
    df["origin"] = pd.to_datetime(df["origin"])

    windows = {
        "pre": df.origin < np.datetime64(GENERATION_WINDOWS["g2024"]),
        "clean": df.origin >= np.datetime64(GENERATION_WINDOWS["g2024"]),
        "frontier": df.origin >= np.datetime64(GENERATION_WINDOWS["g2025"]),
    }

    rows = []
    for wname, mask in windows.items():
        sub = df[mask]
        for (lab, h), g in sub.groupby(["label", "h"]):
            y, f = g["y"].to_numpy(), g["y_hat"].to_numpy()
            ratio = f / y
            rows.append({
                "dataset": args.dataset, "window": wname, "h": int(h),
                "model": lab, "n": len(g),
                # The event that breaks QLIKE: a forecast far below the truth.
                "share_below_1pct": float((ratio < 0.01).mean()),
                "share_below_10pct": float((ratio < 0.10).mean()),
                # And its mirror, which QLIKE punishes far more gently.
                "share_above_10x": float((ratio > 10.0).mean()),
                "min_ratio": float(ratio.min()),
                "median_ratio": float(np.median(ratio)),
                "min_forecast": float(f.min()),
                "min_actual": float(y.min()),
            })

    out = pd.DataFrame(rows).sort_values(
        ["window", "h", "share_below_1pct"], ascending=[True, True, False])
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"tails_{args.dataset}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out)} rows)")

    worst = out[(out.window == "clean") & (out.h == 1)].head(10)
    print("\nworst offenders, clean window, h=1:")
    print(worst[["model", "share_below_1pct", "min_ratio",
                 "median_ratio"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
