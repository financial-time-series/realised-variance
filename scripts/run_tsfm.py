"""Zero-shot foundation-model forecasts. GPU.

    uv run --no-sync python scripts/run_tsfm.py --dataset eq \
        --model chronos-bolt-base --input-transform log \
        --run-dir out/runs/eq-chronos-bolt-base-log

ONE PASS, THREE HORIZONS. These models emit a whole path, so a single call from
each origin produces h=1, 5 and 22 by averaging the first 1, 5 and 22 steps.
That is both cheaper and more faithful than three separate calls -- and it is
the iterated/path convention, whereas the HAR family is estimated directly per
horizon. Both are standard practice for their family; the paper says which is
which rather than pretending the protocols are identical.

THE TRANSFORM IS PART OF THE MODEL, NOT A DETAIL. Realised variance is
strongly right-skewed, and every one of these checkpoints was pretrained on a
corpus with its own scaling conventions. Feeding raw RV and feeding log RV are
different experiments, so both are run and both are reported. If "works
zero-shot, no tuning required" were true, the two would agree; the claim is
falsifiable precisely here.

Under the log transform the path is exponentiated STEP BY STEP and then
averaged, because the target is the mean of the variances and not the variance
of the mean. Plain exp(.) returns a conditional median rather than a mean, and
that bias is left in on purpose: correcting it needs a residual variance
estimated from data, which is exactly the fitting that "zero-shot" excludes.
The consequence -- log-input models under-forecast, and QLIKE punishes them --
is a finding about the deployment recipe, not an artefact to be patched out.
"""

from __future__ import annotations

import argparse
import json
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
from finlab.zoo import ZOO, cached_revision, load  # noqa: E402

HORIZONS = (1, 5, 22)
MIN_HISTORY = 252


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--model", required=True, choices=sorted(ZOO))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--input-transform", default="log", choices=["log", "raw"])
    ap.add_argument("--rv-col", default="rv")
    ap.add_argument("--origin-start", default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--context-len", type=int, default=None,
                    help="override the checkpoint's own maximum, for the "
                         "equal-context robustness run")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default comes from the checkpoint's own Spec")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--origin-step", type=int, default=1)
    ap.add_argument("--shard", default=None,
                    help="i/n: forecast only the series whose position modulo n is i, "
                         "into the shared parts/ directory, and stop. A final run "
                         "without --shard assembles the parts; its timing is the sum of "
                         "the per-series compute times recorded beside each part, so "
                         "the reported cost does not depend on how the work was split")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"device: {device} ({gpu})")

    spec = ZOO[args.model]
    panel = Panel.load(f"data/processed/{args.dataset}_rv.parquet", rv_col=args.rv_col)
    uids = panel.uids[: args.limit] if args.limit else panel.uids
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        uids = [u for j, u in enumerate(uids) if j % n == k]
    origin_start = args.origin_start or ("2010-01-01" if args.dataset == "eq"
                                         else "2020-01-01")
    max_h = max(args.horizons)
    batch_size = args.batch_size or spec.batch_size

    # Per-series checkpointing. The slowest checkpoints here run for hours, and
    # a wall-time kill loses everything not on disk. Each series is written as it
    # finishes, with its own compute time, context count and floor count in a
    # sidecar, and is skipped on a resubmit, so requeuing a job that died at hour
    # seven costs only the series it had not reached.
    parts_dir = Path(args.run_dir) / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    todo = [u for u in uids if not (parts_dir / f"{u}.parquet").exists()]

    fc, ctx_len, load_seconds = None, None, 0.0
    if todo:
        t_load = time.perf_counter()
        fc = load(args.model, device=device, context_len=args.context_len)
        load_seconds = round(time.perf_counter() - t_load, 2)
        # The forecaster may shrink the context itself -- Moirai rounds to a
        # whole number of patches, TTM is fixed-shape. Read it back.
        ctx_len = fc.context_len
        print(f"loaded {spec.repo} in {load_seconds}s (released {spec.released}, "
              f"generation {spec.generation}, context {ctx_len}, batch {batch_size})")
    else:
        print("every series is cached: assembling the parts")

    t0 = time.perf_counter()
    parts, n_floored, n_ctx = [], 0, 0
    cached_seconds, cached_ctx, cached_floored, missing_meta = 0.0, 0, 0, 0
    computed_seconds = 0.0
    meta_gpu = None        # the GPU the cached series were forecast on

    for si, uid in enumerate(uids, 1):
        part_path = parts_dir / f"{uid}.parquet"
        if part_path.exists():
            parts.append(pd.read_parquet(part_path))
            meta = parts_dir / f"{uid}.json"
            if meta.exists():
                m = json.loads(meta.read_text())
                cached_seconds += m["seconds"]
                cached_ctx += m["n_ctx"]
                cached_floored += m["n_floored"]
                ctx_len = ctx_len or m.get("context_len")
                meta_gpu = meta_gpu or m.get("gpu")
            else:
                missing_meta += 1
            print(f"  [{si:>3}/{len(uids)}] {uid:<9} cached", flush=True)
            continue
        t_series = time.perf_counter()
        ctx_series, floored_series = n_ctx, n_floored

        s = panel.series[uid]
        origins = panel.origins(uid, start=origin_start, min_history=MIN_HISTORY,
                                max_h=max_h, step=args.origin_step)
        if len(origins) == 0:
            continue
        pos = np.array([panel._index[uid][d] for d in origins])

        paths = np.empty((len(pos), max_h), dtype=np.float64)
        for b in range(0, len(pos), batch_size):
            idx = pos[b: b + batch_size]
            contexts = []
            for i in idx:
                c = s.rv[max(0, i + 1 - ctx_len): i + 1]
                contexts.append(np.log(c) if args.input_transform == "log" else c)
            paths[b: b + len(idx)] = fc.predict(contexts, max_h)
            n_ctx += len(idx)

        if args.input_transform == "log":
            # step-by-step, then average -- see the module docstring
            paths = np.exp(np.clip(paths, -700, 700))

        per_h = []
        for h in args.horizons:
            y_hat = paths[:, :h].mean(axis=1)
            y_hat, bad = floor_forecasts(y_hat, float(s.rv[:MIN_HISTORY].min()))
            n_floored += bad
            y_true = np.array([np.mean(s.rv[i + 1: i + 1 + h]) for i in pos])
            per_h.append(pd.DataFrame({
                "unique_id": uid,
                "origin": origins,
                "ds": [s.dates[min(i + h, len(s.dates) - 1)] for i in pos],
                "h": h,
                "y": y_true,
                "y_hat": y_hat,
                "model": args.model,
                # The input transform is part of the specification, not a
                # setting: raw-RV and log-RV runs of the same checkpoint are
                # different experiments and must not collapse into one row
                # during aggregation.
                "variant": args.input_transform,
            }))
        block = pd.concat(per_h, ignore_index=True)
        block.to_parquet(part_path, index=False)
        series_seconds = time.perf_counter() - t_series
        computed_seconds += series_seconds
        (parts_dir / f"{uid}.json").write_text(json.dumps({
            "seconds": round(series_seconds, 3),
            "n_ctx": int(n_ctx - ctx_series), "n_floored": int(n_floored - floored_series),
            "context_len": int(ctx_len), "gpu": gpu}))
        parts.append(block)
        if si % 10 == 0 or si == len(uids):
            print(f"  [{si:>3}/{len(uids)}] {uid:<9} "
                  f"{n_ctx:,} contexts  {time.perf_counter() - t0:.0f}s", flush=True)

    if args.shard:
        print(f"shard {args.shard}: {len(todo)} series forecast, "
              f"{time.perf_counter() - t0:.0f}s; assemble with a run without --shard")
        return 0
    if missing_meta:
        print(f"WARNING: {missing_meta} cached series have no timing sidecar; their "
              f"compute time is missing from the reported cost")
    # Compute time is the sum over series, however the work was split: the
    # series forecast in this run plus those recorded beside cached parts.
    elapsed = round(computed_seconds + cached_seconds, 2)
    n_ctx += cached_ctx
    n_floored += cached_floored
    preds = pd.concat(parts, ignore_index=True)
    print(f"{args.model}: {elapsed}s for {n_ctx:,} contexts "
          f"({1000 * elapsed / max(n_ctx, 1):.2f} ms each)"
          + (f", {n_floored} floored" if n_floored else ""))

    write_run(
        args.run_dir, preds,
        config={"dataset": args.dataset, "family": "tsfm", "model": args.model,
                "repo": spec.repo, "released": spec.released,
                "files_changed": spec.files_changed,
                "effective_release": spec.effective_release,
                "revision_pinned": spec.revision,
                "revision_loaded": cached_revision(spec.repo),
                # Non-positive or non-finite point forecasts are floored at
                # FLOOR_FRACTION times the series' smallest variance in its
                # first MIN_HISTORY observations (finlab.evaluation); counted.
                "n_floored": int(n_floored),
                "n_forecasts": int(len(preds)),
                "num_samples": spec.extra.get("num_samples"),
                "generation": spec.generation, "batch_size": batch_size,
                "input_transform": args.input_transform,
                "context_len": ctx_len, "checkpoint_max_context": spec.max_context,
                "rv_col": args.rv_col, "origin_start": origin_start,
                "horizons": args.horizons, "zero_shot": True,
                "n_series": len(uids), "n_contexts": n_ctx,
                "origin_step": args.origin_step, "limit": args.limit},
        timing={"fit_predict_seconds": elapsed, "predict_seconds": elapsed,
                "load_seconds": load_seconds, "n_contexts": n_ctx,
                "ms_per_context": round(1000 * elapsed / max(n_ctx, 1), 3),
                # The device the forecasts were computed on: for a pure
                # assembly run, the one recorded beside the cached parts.
                "device": "cuda" if (gpu or meta_gpu) else device, "gpu": gpu or meta_gpu,
                "cpu": platform.processor() or platform.machine()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
