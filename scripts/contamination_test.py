"""Before and after each checkpoint's own release: the contrast, its tests, placebos.

    .venv/bin/python scripts/contamination_test.py --dataset eq
    .venv/bin/python scripts/contamination_test.py --dataset eq --rv-col rv_parkinson

THE CONTRAST. For a zero-shot checkpoint m with effective release date r_m (the
later of its Hub creation date and the date its weight files last changed; see
finlab.zoo), and a horizon h,

    ratio(w) = median QLIKE of m over window w / median QLIKE of the benchmark over w
    Delta    = ratio(post) - ratio(pre)

with pre = [first origin, r_m) and post = [r_m, last origin]. The split is the
checkpoint's OWN date. The first version of the paper split every 2024
checkpoint at 2025-01-01, the date its whole generation had been released;
that left Chronos-T5-B's ten post-release months of 2024 counted as
"pre-release". The generation windows remain the right design for comparing
checkpoints jointly (MCS, SPA: aggregate.py), which needs a common sample.

Ratios to a benchmark refitted from the data rather than raw losses, because
the two windows differ in how hard they are to forecast and the benchmark
absorbs that. The benchmark is HAR-log, fitted on an expanding window; the
contrast is repeated against HAR-log-roll, fitted on a rolling window of 1,000
observations, so that a benchmark slow to adapt in 2025-26 cannot be what moves
the ratio.

THE TESTS.
  per variant  moving-block bootstrap (blocks of 22 whole origins, carrying the
               cross-section), independently within pre and post; two-sided p
               from the recentred draws.
  pooled       one bootstrap for all variants at once: blocks of 22 origins are
               drawn over the whole timeline and every variant's Delta is
               recomputed on the SAME draw, which preserves the dependence
               between checkpoints that share corpora and lineage. Statistic:
               the median (and the mean) of Delta across the variants of a
               group. This replaces the sign count, which treated dependent
               checkpoints as independent.
  placebo      the release date moved back by the length of the post window,
               c_m = r_m - (end - r_m): pre = [first origin, c_m), post =
               [c_m, r_m). Both windows then precede the release, so a real
               release effect should be absent. Also every 1 January from two
               years after the first origin to the last date that leaves a
               full-length placebo window: the share of those placebo Deltas at
               least as large as the actual one is an empirical p-value.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finlab.evaluation import LOSSES, moving_blocks  # noqa: E402
from finlab.io import load_campaign  # noqa: E402
from finlab.zoo import ZOO  # noqa: E402

TABLES = Path("results/tables")
BLOCK = 22


class Mat:
    """Per-origin loss matrix for one label: rows = origins (time order),
    columns = assets, NaN where an asset has no observation at that origin."""

    def __init__(self, wide: pd.DataFrame):
        self.dates = wide.index.to_numpy()
        self.x = wide.to_numpy(dtype=np.float64)

    def med(self, rows: np.ndarray) -> float:
        v = self.x[rows].ravel()
        v = v[~np.isnan(v)]
        return float(np.median(v)) if len(v) else np.nan

    def mean(self, rows: np.ndarray) -> float:
        v = self.x[rows].ravel()
        return float(np.nanmean(v)) if len(v) else np.nan


def ratio(m: Mat, b: Mat, rows: np.ndarray) -> float:
    return m.med(rows) / b.med(rows)


def delta_on(m: Mat, b: Mat, rows: np.ndarray, cut: np.datetime64,
             lo: np.datetime64 | None = None, hi: np.datetime64 | None = None) -> float:
    """Delta = ratio(post) - ratio(pre), on the origins `rows` (which may repeat,
    as in a bootstrap draw), split at `cut` and restricted to [lo, hi)."""
    d = m.dates[rows]
    keep = np.ones(len(rows), bool)
    if lo is not None:
        keep &= d >= lo
    if hi is not None:
        keep &= d < hi
    pre, post = rows[keep & (d < cut)], rows[keep & (d >= cut)]
    if len(pre) < 60 or len(post) < 40:
        return np.nan
    return ratio(m, b, post) - ratio(m, b, pre)


def per_variant(m: Mat, b: Mat, cut, lo, hi, reps, rng) -> tuple[float, float, float, float]:
    """Delta and its bootstrap p and 95% interval, blocks drawn within each side."""
    allrows = np.arange(len(m.dates))
    d = m.dates
    inwin = (d >= lo) if lo is not None else np.ones(len(d), bool)
    if hi is not None:
        inwin &= d < hi
    pre, post = allrows[inwin & (d < cut)], allrows[inwin & (d >= cut)]
    if len(pre) < 60 or len(post) < 40:
        return np.nan, np.nan, np.nan, np.nan
    obs = ratio(m, b, post) - ratio(m, b, pre)
    draws = np.empty(reps)
    for k in range(reps):
        i = pre[moving_blocks(len(pre), rng, BLOCK)]
        j = post[moving_blocks(len(post), rng, BLOCK)]
        draws[k] = ratio(m, b, j) - ratio(m, b, i)
    null = draws - np.nanmean(draws)
    p = float(np.nanmean(np.abs(null) >= abs(obs)))
    lo_ci, hi_ci = np.nanpercentile(draws, [2.5, 97.5])
    return float(obs), p, float(lo_ci), float(hi_ci)


def work(job: dict) -> tuple[list[dict], list[dict]]:
    """Everything for one (loss, horizon, benchmark): per-variant rows, pooled rows."""
    rng = np.random.default_rng(job["seed"])
    mats, bench = job["mats"], job["bench"]
    b = bench
    first, last = b.dates[0], b.dates[-1]
    rows_out, pool_out = [], []
    deltas, placebo_deltas = {}, {}
    for lab, m in mats.items():
        spec = ZOO[lab.split("[")[0]]
        r = np.datetime64(spec.effective_release)
        span = last - r
        obs, p, lo_ci, hi_ci = per_variant(m, b, r, None, None, job["reps"], rng)
        allrows = np.arange(len(m.dates))
        pre_rows, post_rows = allrows[m.dates < r], allrows[m.dates >= r]
        # Placebo: the same calendar length as the post window, ending at r.
        c = r - span
        if c > first + np.timedelta64(730, "D"):
            pl, pl_p, _, _ = per_variant(m, b, c, None, r, job["reps"], rng)
        else:
            c, pl, pl_p = np.datetime64("NaT"), np.nan, np.nan
        # Yearly placebo cutoffs, each with a post window of the same length.
        yearly = []
        y0 = pd.Timestamp(first).year + 2
        for year in range(y0, pd.Timestamp(r).year + 1):
            cy = np.datetime64(f"{year}-01-01")
            if cy + span > r:
                break
            yearly.append(delta_on(m, b, allrows, cy, None, cy + span))
        yearly = np.array([v for v in yearly if np.isfinite(v)])
        deltas[lab], placebo_deltas[lab] = obs, pl
        rows_out.append({
            "loss": job["loss"], "h": job["h"], "benchmark": job["bench_name"],
            "model": lab, "generation": spec.generation,
            "release": str(r), "pre_start": str(first)[:10],
            "pre_end": str(m.dates[pre_rows[-1]])[:10] if len(pre_rows) else "",
            "n_pre": int(len(pre_rows)),
            "post_start": str(m.dates[post_rows[0]])[:10] if len(post_rows) else "",
            "post_end": str(last)[:10], "n_post": int(len(post_rows)),
            "pre_ratio": ratio(m, b, pre_rows) if len(pre_rows) else np.nan,
            "post_ratio": ratio(m, b, post_rows) if len(post_rows) else np.nan,
            "delta": obs, "p_value": p, "ci_lo": lo_ci, "ci_hi": hi_ci,
            "pre_mean_ratio": (m.mean(pre_rows) / b.mean(pre_rows)) if len(pre_rows) else np.nan,
            "post_mean_ratio": (m.mean(post_rows) / b.mean(post_rows)) if len(post_rows) else np.nan,
            "placebo_cut": str(c)[:10] if not pd.isna(c) else "",
            "placebo_delta": pl, "placebo_p": pl_p,
            "n_placebo_yearly": int(len(yearly)),
            "placebo_yearly_median": float(np.median(yearly)) if len(yearly) else np.nan,
            "placebo_share_ge": float(np.mean(yearly >= obs)) if len(yearly) and np.isfinite(obs) else np.nan,
        })

    # --- pooled: every variant's Delta on the same bootstrap draw -------------
    labs = list(mats)
    groups = {
        "g2024": [l for l in labs if ZOO[l.split("[")[0]].generation == "g2024"],
        "g2024-log": [l for l in labs if ZOO[l.split("[")[0]].generation == "g2024" and "[log]" in l],
        "g2024-raw": [l for l in labs if ZOO[l.split("[")[0]].generation == "g2024" and "[raw]" in l],
        "g2025": [l for l in labs if ZOO[l.split("[")[0]].generation == "g2025"],
    }
    n = len(b.dates)
    cuts = {l: np.datetime64(ZOO[l.split("[")[0]].effective_release) for l in labs}
    pcuts = {l: cuts[l] - (last - cuts[l]) for l in labs}
    draws = {l: np.empty(job["reps_pool"]) for l in labs}
    pdraws = {l: np.empty(job["reps_pool"]) for l in labs}
    for k in range(job["reps_pool"]):
        rows = moving_blocks(n, rng, BLOCK)
        for l in labs:
            draws[l][k] = delta_on(mats[l], b, rows, cuts[l])
            pdraws[l][k] = (delta_on(mats[l], b, rows, pcuts[l], None, cuts[l])
                            if np.isfinite(placebo_deltas[l]) else np.nan)
    for gname, members in groups.items():
        members = [l for l in members if np.isfinite(deltas[l])]
        if len(members) < 2:
            continue
        for kind, obs_d, drw in (("release", deltas, draws), ("placebo", placebo_deltas, pdraws)):
            mm = [l for l in members if np.isfinite(obs_d[l])]
            if len(mm) < 2:
                continue
            v = np.array([obs_d[l] for l in mm])
            D = np.vstack([drw[l] for l in mm])            # (variants, reps)
            med_draw, mean_draw = np.nanmedian(D, axis=0), np.nanmean(D, axis=0)
            p_med = float(np.nanmean(np.abs(med_draw - np.nanmean(med_draw)) >= abs(np.median(v))))
            p_mean = float(np.nanmean(np.abs(mean_draw - np.nanmean(mean_draw)) >= abs(v.mean())))
            worst = int(np.argmax(np.abs(v)))
            pool_out.append({
                "loss": job["loss"], "h": job["h"], "benchmark": job["bench_name"],
                "group": gname, "kind": kind, "n_variants": len(mm),
                "n_positive": int((v > 0).sum()), "mean_delta": float(v.mean()),
                "median_delta": float(np.median(v)),
                "mean_delta_excl_largest": float(np.delete(v, worst).mean()),
                "largest": mm[worst], "largest_delta": float(v[worst]),
                "share_of_sum_largest": float(v[worst] / v.sum()) if v.sum() != 0 else np.nan,
                "p_median": p_med, "p_mean": p_mean,
                "ci_median_lo": float(np.nanpercentile(med_draw, 2.5)),
                "ci_median_hi": float(np.nanpercentile(med_draw, 97.5)),
            })
    print(f"  {job['loss']:<6} h={job['h']:<3} vs {job['bench_name']:<13} done", flush=True)
    return rows_out, pool_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--out-root", default="out/runs")
    ap.add_argument("--rv-col", default="rv")
    ap.add_argument("--reps", type=int, default=2000, help="per-variant bootstrap")
    ap.add_argument("--reps-pool", type=int, default=1000, help="pooled bootstrap")
    ap.add_argument("--losses", nargs="+", default=["qlike", "mse"])
    ap.add_argument("--benchmarks", nargs="+", default=["har-log", "har-log-roll"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="matches aggregate.py; CX needs 0.80")
    args = ap.parse_args()

    from aggregate import (MEASURE_FREE, label, panel_targets,  # noqa: E402
                           restrict_to_common_sample)

    tag = args.dataset if args.rv_col == "rv" else f"{args.dataset}_{args.rv_col}"
    df = load_campaign(args.out_root, dataset=args.dataset, rv_col=args.rv_col,
                       extra_runs=lambda cfg: set(cfg.get("models", [cfg.get("model")]))
                       <= MEASURE_FREE)
    df["label"] = label(df)
    df["origin"] = pd.to_datetime(df["origin"])
    tg = panel_targets(Path(f"data/processed/{args.dataset}_rv.parquet"), args.rv_col)
    df = df.merge(tg[["unique_id", "origin", "h", "y_target"]],
                  on=["unique_id", "origin", "h"], how="left")
    df["y"] = df["y_target"]
    # The SAME common-sample restriction the aggregation applies, so the two
    # scripts compute the same quantity on the same sample.
    df = restrict_to_common_sample(df, args.min_coverage)

    keep = df.label.map(lambda l: l.split("[")[0] in ZOO) | df.label.isin(args.benchmarks)
    df = df[keep]
    jobs = []
    for lname in args.losses:
        fn = LOSSES[lname]
        d = df.assign(loss=fn(df.y.to_numpy(), df.y_hat.to_numpy()))
        for h in sorted(d.h.unique()):
            dh = d[d.h == h]
            mats = {}
            for lab, g in dh.groupby("label"):
                w = g.pivot_table(index="origin", columns="unique_id", values="loss")
                mats[lab] = w
            for bname in args.benchmarks:
                if bname not in mats:
                    continue
                bw = mats[bname]
                zs = {l: Mat(w.reindex(index=bw.index, columns=bw.columns))
                      for l, w in mats.items() if l.split("[")[0] in ZOO}
                jobs.append({"loss": lname, "h": int(h), "bench_name": bname,
                             "bench": Mat(bw), "mats": zs, "reps": args.reps,
                             "reps_pool": args.reps_pool,
                             "seed": args.seed + 1000 * int(h) + (0 if lname == "qlike" else 7)})
    print(f"{len(jobs)} (loss, horizon, benchmark) jobs on {args.workers} workers", flush=True)
    rows, pools = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r, pl in ex.map(work, jobs):
            rows += r
            pools += pl

    TABLES.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.insert(0, "dataset", args.dataset)
    out.insert(1, "rv_col", args.rv_col)
    out.to_csv(TABLES / f"contamtest_{tag}.csv", index=False)
    pool = pd.DataFrame(pools)
    pool.insert(0, "dataset", args.dataset)
    pool.insert(1, "rv_col", args.rv_col)
    pool.to_csv(TABLES / f"contampool_{tag}.csv", index=False)
    print(f"\nwrote contamtest_{tag}.csv ({len(out)} rows) and contampool_{tag}.csv ({len(pool)} rows)")

    for (lname, bname), s in pool[pool.kind == "release"].groupby(["loss", "benchmark"]):
        for _, r in s[s.group == "g2024"].sort_values("h").iterrows():
            print(f"  {lname:<6} vs {bname:<13} h={int(r.h):>2}: median Delta {r.median_delta:+.3f} "
                  f"(p {r.p_median:.3f}), mean {r.mean_delta:+.3f}, "
                  f"{int(r.n_positive)}/{int(r.n_variants)} positive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
