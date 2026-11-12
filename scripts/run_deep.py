"""Deep forecasters trained from scratch on this data. GPU.

    uv run --no-sync python scripts/run_deep.py --dataset eq --model dlinear \
        --seed 1 --run-dir out/runs/eq-dlinear-s1

These are the control that separates two very different explanations for a
zero-shot result. If the foundation models beat HAR, is that because
pretraining transferred something, or merely because a flexible neural network
fitted to this panel would have done as well? Only a like-for-like architecture
trained here can tell them apart, so DLinear, PatchTST and NHITS are trained on
exactly the same panel, over exactly the same origins, and scored identically.

DLinear earns its place specifically: Zeng et al. (2023) showed a single linear
layer matching elaborate transformers on long-horizon benchmarks. If it matches
the foundation models here too, the story is about the difficulty of the task,
not the sophistication of the model.

REFIT ANNUALLY, NOT MONTHLY. The statistical models re-estimate every 22
origins; retraining a neural network that often across 4,000 origins is not
merely expensive, it is not what anyone does. `--refit-every 252` is the
honest setting and the cost table reports what it bought.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finlab.evaluation import floor_forecasts  # noqa: E402
from finlab.io import write_run  # noqa: E402
from finlab.panel import Panel  # noqa: E402

HORIZONS = (1, 5, 22)
MIN_HISTORY = 252
INPUT_SIZE = 128        # ~6 months of trading days of context


def make_model(name: str, h: int, seed: int, max_steps: int):
    from neuralforecast.models import NHITS, DLinear, PatchTST

    common = dict(h=h, input_size=INPUT_SIZE, max_steps=max_steps,
                  random_seed=seed, scaler_type="standard",
                  enable_progress_bar=False, logger=False)
    if name == "dlinear":
        return DLinear(**common, learning_rate=1e-3, batch_size=256)
    if name == "patchtst":
        return PatchTST(**common, learning_rate=1e-3, batch_size=128,
                        patch_len=16, stride=8, hidden_size=128, n_heads=4)
    if name == "nhits":
        return NHITS(**common, learning_rate=1e-3, batch_size=256,
                     n_freq_downsample=[4, 2, 1])
    raise SystemExit(f"unknown model {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--model", required=True,
                    choices=["dlinear", "patchtst", "nhits"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rv-col", default="rv")
    ap.add_argument("--origin-start", default=None)
    ap.add_argument("--refit-every", type=int, default=252)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--origin-step", type=int, default=1)
    ap.add_argument("--n-windows", type=int, default=None,
                    help="cap the number of rolling origins -- smoke tests")
    args = ap.parse_args()

    import torch
    from neuralforecast import NeuralForecast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"device: {device} ({gpu})")

    panel = Panel.load(f"data/processed/{args.dataset}_rv.parquet", rv_col=args.rv_col)
    uids = panel.uids[: args.limit] if args.limit else panel.uids
    origin_start = args.origin_start or ("2010-01-01" if args.dataset == "eq"
                                         else "2020-01-01")
    max_h = max(args.horizons)

    # neuralforecast wants a long frame. Train on log RV: the level is
    # right-skewed over orders of magnitude and a network fitted to it spends
    # its capacity on the tail. Scaling is per series and fitted on train only.
    #
    # TRADING TIME, NOT CALENDAR TIME. `ds` is the observation's INDEX within
    # its own series, not its date. A calendar frequency ("B") makes
    # neuralforecast reindex onto a regular business-day grid, which invents
    # rows for every market holiday -- forecasts then come back stamped with
    # dates the exchange never traded on, and joining them to the panel fails
    # (or worse, silently matches the wrong day). Trading time is also exactly
    # what panel.py already defines a horizon in, so this keeps one convention
    # across the whole campaign. Positions are mapped back to real dates below.
    rows = []
    for uid in uids:
        s = panel.series[uid]
        rows.append(pd.DataFrame({"unique_id": uid,
                                  "ds": np.arange(len(s.rv)),
                                  "y": np.log(s.rv)}))
    frame = pd.concat(rows, ignore_index=True)

    # The evaluation window, expressed in positions: how many origins of each
    # series fall on or after origin_start. Counted on the shortest so every
    # origin exists for every asset.
    per_uid = {}
    for uid in uids:
        s = panel.series[uid]
        per_uid[uid] = int((s.dates >= np.datetime64(origin_start)).sum())
    n_windows = max(min(per_uid.values()) - max_h, 1)
    if args.n_windows:
        n_windows = min(n_windows, args.n_windows)
    print(f"{len(uids)} series, {n_windows} rolling origins, refit every "
          f"{args.refit_every}")

    t0 = time.perf_counter()
    nf = NeuralForecast(
        models=[make_model(args.model, max_h, args.seed, args.max_steps)],
        freq=1,   # integer trading-time index -- see the frame construction above
    )
    cv = nf.cross_validation(
        df=frame, n_windows=n_windows, step_size=args.origin_step,
        refit=args.refit_every,
    )
    elapsed = round(time.perf_counter() - t0, 2)

    cv = cv.reset_index() if "unique_id" not in cv.columns else cv
    col = [c for c in cv.columns
           if c.lower() == args.model or c in ("DLinear", "PatchTST", "NHITS")][0]
    cv = cv.rename(columns={col: "pred", "cutoff": "origin_pos"})

    # cross_validation returns one row per (origin, step). Exponentiate each
    # step and only then average: the target is the mean of the variances.
    cv["pred_rv"] = np.exp(np.clip(cv["pred"].to_numpy(), -700, 700))
    cv = cv.sort_values(["unique_id", "origin_pos", "ds"])
    cv["step"] = cv.groupby(["unique_id", "origin_pos"]).cumcount() + 1

    parts = []
    for h in args.horizons:
        sub = cv[cv.step <= h]
        agg = (sub.groupby(["unique_id", "origin_pos"])
                  .agg(y_hat=("pred_rv", "mean")).reset_index())

        # Positions back to dates, per series. Everything downstream keys on
        # real origin dates, so this is where trading time is left behind.
        origins, targets, dss, keep = [], [], [], []
        for uid, opos in zip(agg.unique_id, agg.origin_pos):
            s = panel.series[uid]
            i = int(opos)
            ok = 0 <= i < len(s.dates) and i + h < len(s.rv)
            keep.append(ok)
            if not ok:
                origins.append(np.datetime64("NaT")); targets.append(np.nan)
                dss.append(np.datetime64("NaT"))
                continue
            origins.append(s.dates[i])
            dss.append(s.dates[min(i + h, len(s.dates) - 1)])
            # Score against the panel's own target, not the frame's `y`: one
            # definition of the target, in one place (panel.py).
            targets.append(float(np.mean(s.rv[i + 1: i + 1 + h])))

        agg["origin"] = origins
        agg["ds"] = dss
        agg["y"] = targets
        agg = agg[np.array(keep) & np.isfinite(agg["y"].to_numpy())]
        agg = agg[agg["origin"].to_numpy() >= np.datetime64(origin_start)]

        y_hat, _ = floor_forecasts(agg["y_hat"].to_numpy(),
                                   float(np.exp(frame["y"].min())))
        parts.append(pd.DataFrame({
            "unique_id": agg.unique_id.to_numpy(), "origin": agg.origin.to_numpy(),
            "ds": agg.ds.to_numpy(), "h": h, "y": agg.y.to_numpy(),
            "y_hat": y_hat, "model": args.model, "seed": args.seed,
        }))
        print(f"  h={h:>2}: {len(agg):,} forecasts")

    print(f"{args.model} seed {args.seed}: {elapsed}s")
    write_run(
        args.run_dir, pd.concat(parts, ignore_index=True),
        config={"dataset": args.dataset, "family": "deep", "model": args.model,
                "seed": args.seed, "rv_col": args.rv_col,
                "origin_start": origin_start, "refit_every": args.refit_every,
                "input_size": INPUT_SIZE, "max_steps": args.max_steps,
                "horizons": args.horizons, "n_windows": n_windows,
                "target": "log(RV)", "global_model": True,
                "n_series": len(uids), "origin_step": args.origin_step,
                "limit": args.limit},
        timing={"fit_predict_seconds": elapsed, "device": device, "gpu": gpu,
                "cpu": platform.processor() or platform.machine()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
