"""SPA on the crypto common window without one erratic candidate.

    .venv/bin/python scripts/spa_drop_one.py

The SPA p-value of the crypto common (frontier) window at h=1 under QLIKE was
0.187 in the first version and is 0.467 in the re-run. The difference is one
forecast: re-sampling Moirai-MoE-B on raw input produced a single non-positive
forecast (16 January 2026), which by itself carries the whole of that
candidate's loss differential and widens the bootstrap distribution of the SPA
statistic. This script recomputes the test on the first version's candidate set
with and without Moirai-MoE-B (raw and calibrated), exactly as aggregate.py
builds it, and writes results/tables/spa_drop_one_cx.csv. Quoted in the
manuscript (section "Does the newer generation change the answer?").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from finlab.evaluation import per_origin_loss, spa_test  # noqa: E402
from finlab.io import load_campaign  # noqa: E402
import aggregate as A  # noqa: E402

DROP = "moirai-moe-base[raw]"


def main() -> int:
    df = load_campaign("out/runs", dataset="cx", rv_col="rv",
                       extra_runs=lambda cfg: set(cfg.get("models", [cfg.get("model")]))
                       <= A.MEASURE_FREE)
    df["label"] = A.label(df)
    df["origin"] = pd.to_datetime(df["origin"])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df[df.h == 1]
    tg = A.panel_targets(Path("data/processed/cx_rv.parquet"), "rv")
    df = df.merge(tg, on=["unique_id", "origin", "h"], how="left")
    df["y"] = df["y_target"]
    df = df.drop(columns=["y_target"])
    df = A.restrict_to_common_sample(df, 0.80)
    df = A.add_calibrated(df, A.LEAKAGE_FREE_START)
    s = df[(df.origin >= np.datetime64(A.GENERATION_WINDOWS["g2025"]))
           & ~df.label.str.endswith("+rtcal")].copy()
    s["qlike"] = A.LOSSES["qlike"](s.y.to_numpy(), s.y_hat.to_numpy())
    s = s.groupby(["label", "unique_id", "origin"], sort=False)["qlike"].mean().reset_index()
    wide = per_origin_loss(s.rename(columns={"label": "model"}), "qlike", agg="mean",
                           loss_col="qlike")
    orig = wide.loc[:, ~wide.columns.map(A.base_name).isin(A.REVISION_MODELS)]
    drop = [c for c in orig.columns if c.startswith(DROP)]
    d = orig[DROP] - orig[A.BENCHMARK]
    rows = []
    for name, cand in (("original", orig), (f"original without {DROP}", orig.drop(columns=drop))):
        r = spa_test(cand, A.BENCHMARK, reps=1000)
        rows.append({"dataset": "cx", "window": "frontier", "h": 1, "loss": "qlike",
                     "set": name, "n_origins": len(cand), **r})
    out = pd.DataFrame(rows)
    out["worst_day_of_dropped"] = str(d.idxmax().date())
    out["worst_day_share_of_sum"] = float(d.max() / d.sum())
    path = ROOT / "results/tables/spa_drop_one_cx.csv"
    out.to_csv(path, index=False)
    print(out[["set", "n_alternatives", "pvalue_consistent", "best_alternative"]].to_string(index=False))
    print(f"wrote {path}")
    # aggregate.py's helpers leave intermediate files that only its own main()
    # renames; this script is not that main(), so it removes them.
    for tmp in ("_cal_factors_tmp.csv", "_common_tmp.csv", "_rtcal_counts_tmp.csv"):
        (ROOT / "results/tables" / tmp).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
