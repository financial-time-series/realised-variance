"""Turn a campaign into the paper's tables. RUN THIS ON THE CLUSTER, in a job.

    .venv/bin/python scripts/aggregate.py --dataset eq --out-root out/runs
    .venv/bin/python scripts/aggregate.py --dataset eq --rv-col rv_parkinson   # robustness

`out/` lives on the cluster; the CSVs written here are kilobytes and are what
comes back (`slurm/sync.sh pull`). Iterating on presentation is then a local
loop with no cluster round-trip.

THE EVALUATION WINDOWS. A window here is a date range PLUS the set of models
that may honestly be scored on it, because the checkpoints were not all
published at the same time. A zero-shot model is admissible on a window only if
the window opens after every member of its generation was published.

  pre        origins before 2025-01-01: what a study without the release-date
             control would have measured. Every model is scored here.
  clean      origins from 2025-01-01, after every 2024 checkpoint. The 2025
             generation is excluded.
  frontier   origins from 2025-11-01, after every checkpoint in the study.
  stress     the most volatile fifth of `clean` origins, by each asset's own
  /calm      trailing variance as visible at the origin.
  pre_g2025  2025-01-01 to 2025-10-31: post-release for the 2024 generation,
             pre-release for the 2025 one.
  blockA     the same dates as pre_g2025, 2024 generation and baselines only:
             the FIRST of two consecutive out-of-sample blocks. A specification
             chosen on blockA and scored on `frontier` (the second block) is
             evaluated on data that played no part in choosing it -- the
             double out-of-sample design of Mroziewicz and Slepaczuk (2026).
  b2025h1,   three consecutive, non-overlapping half-year blocks of the
  b2025h2,   post-release period (the third runs to the end of the sample), so
  b2026      that a result can be checked block by block rather than once.

The per-checkpoint before/after contrast, where each checkpoint's own release
date splits the sample, is computed by scripts/contamination_test.py.

THE TESTS. Diebold-Mariano (HLN-corrected), the model confidence set and
Hansen's SPA, each run on per-origin losses that summarise the cross-section
three ways (`--aggregators`): the mean (the conventional test), the median and
the 10% trimmed mean. The first version of the paper tested only the mean,
while its tables also reported medians that carried no test; a block-bootstrap
test of the difference in pooled medians now accompanies every median.

THE CALIBRATED VARIANTS.
  <model>+cal    one multiplicative constant per series and horizon: the
                 geometric mean of y / y_hat over every origin before
                 2025-01-01 (at least 252), applied unchanged afterwards.
  <model>+rtcal  zero-shot models only. The same estimator, updated in real
                 time from POST-RELEASE origins alone: at origin t it averages
                 log(y/y_hat) over origins from the checkpoint's own release
                 date whose target window had closed by t (at least 20; before
                 that the forecast is left unscaled). No observation that could
                 sit in a pretraining corpus enters it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finlab.evaluation import (  # noqa: E402
    AGGREGATORS, DIRECTIONAL_LOSSES, LOSSES, ROBUST_LOSSES, TRIM, dm_hln,
    median_difference_test, model_confidence_set, per_origin_loss, spa_test,
)
from finlab.io import load_campaign, load_costs  # noqa: E402
from finlab.zoo import GENERATION_WINDOWS, LEAKAGE_FREE_START, ZOO  # noqa: E402

TABLES = Path("results/tables")
BENCHMARK = "har-log"          # the model everything is tested against
MIN_CAL_ORIGINS = 252
MIN_RT_ORIGINS = 20
HORIZONS = (1, 5, 22)
# Forecasts that do not depend on the realised-variance measure: GARCH reads
# returns. In a robustness run on another measure their forecasts are reused
# and only their target is re-scored.
MEASURE_FREE = {"garch", "egarch", "garch-oc"}
# Added in revision: kept apart so the SPA can also be run on the model set of
# the first version, whose candidate count the paper quotes.
REVISION_MODELS = {"har-log-roll", "har-log-iv", "garch-oc", "lstm-hybrid"}
# Windows that get the robust (median, trimmed) MCS and SPA as well as the mean.
ROBUST_TEST_WINDOWS = ("clean", "frontier", "stress")
MEDIAN_TEST_WINDOWS = ("clean", "frontier", "stress", "calm", "blockA",
                       "b2025h1", "b2025h2", "b2026")


def base_name(label: str) -> str:
    """Strip the [transform] and calibration decorations to find the architecture."""
    return label.split("[")[0].replace("+rtcal", "").replace("+cal", "")


def label(df: pd.DataFrame) -> pd.Series:
    """Display name: architecture, plus the variant when one is recorded."""
    if "variant" not in df.columns:
        return df["model"]
    v = df["variant"].fillna("")
    return np.where(v.eq("") | v.isna(), df["model"], df["model"] + "[" + v + "]")


def add_calibrated(df: pd.DataFrame, window_start: str) -> pd.DataFrame:
    """Append `<model>+cal` rows, scaled by a pre-window constant per series."""
    cutoff = np.datetime64(window_start)
    past = df[df.origin < cutoff]
    if past.empty:
        return df

    # Geometric mean ratio: robust to the right skew of variance, and it is the
    # correction a lognormal median-to-mean adjustment implies.
    ratio = (past.assign(lr=np.log(past.y) - np.log(past.y_hat))
                 .groupby(["label", "unique_id", "h"])["lr"]
                 .agg(["mean", "count"]))
    ratio = ratio[ratio["count"] >= MIN_CAL_ORIGINS]
    if ratio.empty:
        return df
    factor = np.exp(ratio["mean"]).rename("factor")
    factor.reset_index().to_csv(TABLES / "_cal_factors_tmp.csv", index=False)

    fut = df[df.origin >= cutoff].merge(
        factor, left_on=["label", "unique_id", "h"], right_index=True, how="inner")
    if fut.empty:
        return df
    fut = fut.assign(y_hat=fut.y_hat * fut.factor, label=fut.label + "+cal")
    return pd.concat([df, fut.drop(columns=["factor"])], ignore_index=True)


def add_rt_calibrated(df: pd.DataFrame) -> pd.DataFrame:
    """Append `<model>+rtcal` rows: a constant estimated in real time from
    post-release origins only (zero-shot models). See the module docstring."""
    zs = df[df.label.map(base_name).isin(ZOO) & ~df.label.str.contains(r"\+")]
    parts = []
    for (lab, uid, h), g in zs.groupby(["label", "unique_id", "h"], sort=False):
        spec = ZOO[base_name(lab)]
        release = np.datetime64(spec.effective_release)
        start = np.datetime64(GENERATION_WINDOWS[spec.generation])
        g = g.sort_values("origin")
        est = g[g.origin >= release]
        tgt = g[g.origin >= start]
        if est.empty or tgt.empty:
            continue
        lr = np.log(est.y.to_numpy() / est.y_hat.to_numpy())
        csum = np.concatenate([[0.0], np.cumsum(lr)])
        # Origins s whose target window (ending on ds_s) had closed by origin t.
        k = np.searchsorted(est.ds.to_numpy(), tgt.origin.to_numpy(), side="right")
        mean_lr = np.where(k >= MIN_RT_ORIGINS, csum[k] / np.maximum(k, 1), 0.0)
        parts.append(tgt.assign(y_hat=tgt.y_hat.to_numpy() * np.exp(mean_lr),
                                label=lab + "+rtcal", rt_n=k))
    if not parts:
        return df
    rt = pd.concat(parts, ignore_index=True)
    (rt.groupby("label")["rt_n"].agg(["min", "median", "max"]).reset_index()
       .to_csv(TABLES / "_rtcal_counts_tmp.csv", index=False))
    return pd.concat([df, rt.drop(columns=["rt_n"])], ignore_index=True)


def restrict_to_common_sample(df: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """Score every model on the identical set of (asset, origin, horizon) keys.

    The runners do not naturally agree on their origin sets (the pooled ML
    models drop missing rows per horizon, the deep models take their rolling
    windows from the end of the shortest series). Left alone that is several
    different samples, and the losses would not be comparable. Intersecting
    costs a fraction of a percent of the observations. The shrinkage is checked
    against `min_coverage` so that a run which died part-way cannot silently
    define the sample for every model.
    """
    key = ["unique_id", "origin", "h"]
    per_run = {r: set(map(tuple, g[key].drop_duplicates()
                          .itertuples(index=False, name=None)))
               for r, g in df.groupby("run")}
    sizes = {r: len(k) for r, k in per_run.items()}
    biggest = max(sizes.values())

    common: set = set.intersection(*per_run.values())
    coverage = len(common) / biggest
    thin = sorted((r for r, n in sizes.items() if n / biggest < min_coverage),
                  key=lambda r: sizes[r])

    print(f"common sample: {len(common):,} (asset, origin, horizon) keys "
          f"= {coverage:.2%} of the largest run ({biggest:,})")
    if thin:
        raise SystemExit(
            "REFUSING TO AGGREGATE: these runs cover far less than the rest, "
            "which usually means they died part-way and their partial output "
            "would silently define the sample for every model:\n  "
            + "\n  ".join(f"{r}: {sizes[r]:,} keys ({sizes[r]/biggest:.1%})"
                          for r in thin)
            + f"\nRe-run them, or drop them from out/runs, or lower "
              f"--min-coverage (currently {min_coverage:.0%}) deliberately."
        )

    idx = pd.MultiIndex.from_frame(df[key])
    keep = idx.isin(common)
    dropped = len(df) - int(keep.sum())
    print(f"  dropped {dropped:,} rows ({dropped / len(df):.2%}) outside it")
    pd.DataFrame([{"keys": len(common), "largest_run": biggest,
                   "coverage": coverage, "rows_dropped": dropped,
                   "rows_dropped_share": dropped / len(df)}]).to_csv(
        TABLES / "_common_tmp.csv", index=False)
    return df[keep].reset_index(drop=True)


def check_models_are_distinct(df: pd.DataFrame, tol: float = 1e-12) -> None:
    """Fail if two models produce the same forecasts. They should not.

    This exists because one already did: the EGARCH runs of the first campaign
    were bit-for-bit the random walk at every multi-step horizon, after the
    runner caught an error and fell back. Distinct model names must yield
    distinct forecasts.
    """
    wide = (df.pivot_table(index=["unique_id", "origin", "h"], columns="label",
                           values="y_hat", aggfunc="mean")
              .dropna(axis=0, how="any"))
    if wide.shape[1] < 2 or wide.empty:
        return
    cols = list(wide.columns)
    arr = wide.to_numpy()
    twins = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if base_name(cols[i]) == base_name(cols[j]):
                continue
            if np.allclose(arr[:, i], arr[:, j], rtol=tol, atol=0.0):
                twins.append((cols[i], cols[j]))
    if twins:
        raise SystemExit(
            "REFUSING TO AGGREGATE: these model pairs produce identical "
            "forecasts, which means one of them is not the model it claims "
            "to be:\n  " + "\n  ".join(f"{a}  ==  {b}" for a, b in twins))
    print(f"  distinctness: all {len(cols)} model variants differ")


def panel_targets(panel_path: Path, rv_col: str) -> pd.DataFrame:
    """Per (asset, origin, h): the target mean of `rv_col` over the next h
    observations, and the MADL reference -- its mean over the h observations up
    to and including the origin."""
    panel = pd.read_parquet(panel_path)[["unique_id", "ds", rv_col]]
    panel["ds"] = pd.to_datetime(panel["ds"])
    from numpy.lib.stride_tricks import sliding_window_view
    out = []
    for uid, g in panel.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        x = g[rv_col].to_numpy(dtype=np.float64)
        n = len(x)
        for h in HORIZONS:
            # Exact window means (not differences of a cumulative sum, which
            # lose relative precision on the smallest variances).
            wm = sliding_window_view(x, h).mean(axis=1)       # wm[j] = mean x[j:j+h]
            y_t = np.full(n, np.nan)
            y_t[: n - h] = wm[1:]                             # mean x[t+1 : t+1+h]
            base = np.full(n, np.nan)
            base[h - 1:] = wm                                 # mean x[t-h+1 : t+1]
            out.append(pd.DataFrame({"unique_id": uid, "origin": g["ds"].to_numpy(),
                                     "h": h, "y_target": y_t, "base": base}))
    return pd.concat(out, ignore_index=True)


def stress_flag(df: pd.DataFrame, panel_path: Path, rv_col: str) -> np.ndarray:
    """True where the origin sits in the top quintile of the asset's own
    trailing 22-day variance. Computed from data visible AT the origin."""
    panel = pd.read_parquet(panel_path)[["unique_id", "ds", rv_col]]
    panel["ds"] = pd.to_datetime(panel["ds"])
    panel = panel.sort_values(["unique_id", "ds"])
    panel["trail22"] = (panel.groupby("unique_id")[rv_col]
                        .transform(lambda s: s.rolling(22, min_periods=22).mean()))
    # Expanding quintile so the threshold itself uses no future information.
    panel["thresh"] = (panel.groupby("unique_id")["trail22"]
                       .transform(lambda s: s.expanding(252).quantile(0.8)))
    panel["stress"] = panel["trail22"] > panel["thresh"]
    key = panel.set_index(["unique_id", "ds"])["stress"]
    idx = pd.MultiIndex.from_arrays([df.unique_id, df.origin])
    return key.reindex(idx).fillna(False).to_numpy()


def admissible_mask(labels: pd.Series, window_start: str | None) -> np.ndarray:
    """A zero-shot model may appear on a window only if the window opens after
    every member of its generation was published; baselines always may."""
    if window_start is None:
        return np.ones(len(labels), dtype=bool)
    ws = np.datetime64(window_start)
    gen_start = labels.map(lambda l: GENERATION_WINDOWS[ZOO[base_name(l)].generation]
                           if base_name(l) in ZOO else "1900-01-01")
    return (pd.to_datetime(gen_start).to_numpy() <= ws)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--out-root", default="out/runs")
    ap.add_argument("--benchmark", default=BENCHMARK)
    ap.add_argument("--rv-col", default="rv",
                    help="the realised-variance measure: rv (the main study), "
                         "rv_parkinson or rv_rs (equity robustness)")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--aggregators", nargs="+", default=list(AGGREGATORS))
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="refuse to aggregate if any run covers less than this "
                         "fraction of the largest run's keys")
    ap.add_argument("--no-mcs", action="store_true", help="skip the MCS (slow)")
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    tag = args.dataset if args.rv_col == "rv" else f"{args.dataset}_{args.rv_col}"
    panel_path = Path(f"data/processed/{args.dataset}_rv.parquet")

    df = load_campaign(args.out_root, dataset=args.dataset, rv_col=args.rv_col,
                       extra_runs=lambda cfg: set(cfg.get("models", [cfg.get("model")]))
                       <= MEASURE_FREE)
    df["label"] = label(df)
    df["origin"] = pd.to_datetime(df["origin"])
    df["ds"] = pd.to_datetime(df["ds"])
    print(f"loaded {len(df):,} forecasts, {df.label.nunique()} model variants, "
          f"{df.unique_id.nunique()} series, measure {args.rv_col}")

    # One definition of the target, from the panel. For runs fed this measure
    # it must reproduce the stored target exactly; GARCH runs, which read
    # returns, are re-scored against it.
    tg = panel_targets(panel_path, args.rv_col)
    df = df.merge(tg, on=["unique_id", "origin", "h"], how="left")
    same = df["rv_col"].eq(args.rv_col).to_numpy()
    gap = np.nanmax(np.abs(df.loc[same, "y"].to_numpy() / df.loc[same, "y_target"].to_numpy() - 1))
    print(f"  target check: max relative gap {gap:.2e} on runs fed {args.rv_col}")
    if not gap <= 1e-6:
        raise SystemExit("REFUSING TO AGGREGATE: stored targets disagree with the panel")
    df["y"] = df["y_target"]
    df = df.drop(columns=["y_target"])

    df = restrict_to_common_sample(df, args.min_coverage)
    check_models_are_distinct(df)

    if "seed" in df.columns and df["seed"].notna().any():
        seed_spread = (df[df.seed.notna()]
                       .assign(err=lambda d: np.abs(np.log(d.y) - np.log(d.y_hat)))
                       .groupby(["label", "h", "seed"])["err"].mean()
                       .groupby(["label", "h"]).std().rename("seed_std_mae_log"))
        seed_spread.reset_index().to_csv(TABLES / f"seeds_{tag}.csv", index=False)

    df = add_calibrated(df, LEAKAGE_FREE_START)
    df = add_rt_calibrated(df)
    df["is_stress"] = stress_flag(df, panel_path, args.rv_col)

    o = df.origin
    g24, g25 = (np.datetime64(GENERATION_WINDOWS["g2024"]),
                np.datetime64(GENERATION_WINDOWS["g2025"]))
    is_rt = df.label.str.endswith("+rtcal").to_numpy()
    stress = df["is_stress"].to_numpy()
    adm24 = admissible_mask(df.label, GENERATION_WINDOWS["g2024"])
    windows = {
        "pre": ((o < g24).to_numpy() & ~is_rt, None),
        "clean": ((o >= g24).to_numpy() & adm24, "g2024"),
        "frontier": ((o >= g25).to_numpy(), "g2025"),
        "stress": ((o >= g24).to_numpy() & adm24 & stress, "g2024"),
        "calm": ((o >= g24).to_numpy() & adm24 & ~stress, "g2024"),
        "pre_g2025": (((o >= g24) & (o < g25)).to_numpy() & ~is_rt, None),
        "blockA": (((o >= g24) & (o < g25)).to_numpy() & adm24, "g2024"),
        "b2025h1": (((o >= g24) & (o < "2025-07-01")).to_numpy() & adm24, "g2024"),
        "b2025h2": (((o >= "2025-07-01") & (o < "2026-01-01")).to_numpy() & adm24, "g2024"),
        "b2026": ((o >= "2026-01-01").to_numpy(), "g2025"),
    }

    loss_rows, dm_rows, mcs_rows, spa_rows, med_rows, win_rows = [], [], [], [], [], []
    loss_names = list(LOSSES) + list(DIRECTIONAL_LOSSES)

    for wname, (wmask, _gen) in windows.items():
        sub = df[wmask]
        if sub.empty:
            continue
        # Calibrated variants exist only after 2025-01-01; on the pre-release
        # windows they would mix calibrated and uncalibrated origins.
        if wname in ("pre", "pre_g2025"):
            sub = sub[~sub.label.str.contains(r"\+")]
        win_rows.append({"window": wname, "first_origin": sub.origin.min(),
                         "last_origin": sub.origin.max(),
                         "n_origins": int(sub.origin.nunique()),
                         "n_series": int(sub.unique_id.nunique())})
        for h in sorted(sub.h.unique()):
            s = sub[sub.h == h].copy()
            if s.empty:
                continue
            yv, fv = s.y.to_numpy(), s.y_hat.to_numpy()
            for lname, fn in LOSSES.items():
                s[lname] = fn(yv, fv)
            s["madl"] = DIRECTIONAL_LOSSES["madl"](yv, fv, s["base"].to_numpy())
            # Seeds are replicates: average each key's loss over them, so a
            # seeded model has one loss per key like every other model. The
            # mean is unchanged by this; the median and the tests need it.
            s = (s.groupby(["label", "unique_id", "origin"], sort=False)[loss_names]
                  .mean().reset_index())

            for lname in loss_names:
                grp = s.groupby("label")[lname]
                stats_ = pd.DataFrame({
                    "value": grp.mean(), "median": grp.median(),
                    "trimmed": grp.agg(lambda x: trim_mean(x.to_numpy(), TRIM)),
                    "n": grp.size()})
                stats_["n_series"] = s.groupby("label")["unique_id"].nunique()
                # How much of the mean the worst 0.1% of observations carry.
                tail = grp.apply(lambda x: float(
                    x.nlargest(max(int(len(x) * 0.001), 1)).sum() / x.sum())
                    if x.sum() > 0 else np.nan)
                for lab, row in stats_.iterrows():
                    loss_rows.append({"dataset": args.dataset, "rv_col": args.rv_col,
                                      "window": wname, "h": int(h), "model": lab,
                                      "loss": lname, "value": float(row.value),
                                      "median": float(row["median"]),
                                      "trimmed": float(row.trimmed),
                                      "tail_share_top0.1pct": float(tail.get(lab, np.nan)),
                                      "n_obs": int(row.n), "n_series": int(row.n_series)})

            if args.benchmark not in set(s.label):
                continue

            # --- pooled-median tests: what the median columns report --------
            if wname in MEDIAN_TEST_WINDOWS:
                for lname in ("qlike", "mse"):
                    wide = s.pivot_table(index=["origin", "unique_id"], columns="label",
                                         values=lname).dropna(axis=0, how="any")
                    wide = wide.sort_index()
                    origins = wide.index.get_level_values(0).to_numpy()
                    starts = np.r_[0, np.flatnonzero(origins[1:] != origins[:-1]) + 1, len(origins)]
                    rows_by_origin = [np.arange(a, b) for a, b in zip(starts[:-1], starts[1:])]
                    cols = list(wide.columns)
                    diff, p = median_difference_test(
                        wide.to_numpy(), rows_by_origin, cols.index(args.benchmark),
                        reps=args.reps)
                    med = np.median(wide.to_numpy(), axis=0)
                    for j, lab in enumerate(cols):
                        if lab == args.benchmark:
                            continue
                        med_rows.append({"dataset": args.dataset, "rv_col": args.rv_col,
                                         "window": wname, "h": int(h), "loss": lname,
                                         "model": lab, "benchmark": args.benchmark,
                                         "median_model": float(med[j]),
                                         "median_bench": float(med[cols.index(args.benchmark)]),
                                         "diff": float(diff[j]), "p_value": float(p[j]),
                                         "n_origins": len(rows_by_origin)})

            # --- DM, MCS and SPA on per-origin summaries ---------------------
            for lname in ("qlike", "mse", "madl"):
                aggs = args.aggregators if lname != "madl" else ["mean"]
                for agg in aggs:
                    wide = per_origin_loss(s.rename(columns={"label": "model"}), lname,
                                           agg=agg, loss_col=lname)
                    if args.benchmark not in wide.columns or wide.shape[1] < 2:
                        continue
                    for model in wide.columns:
                        if model == args.benchmark:
                            continue
                        stat, p = dm_hln(wide[model].to_numpy(),
                                         wide[args.benchmark].to_numpy(), h)
                        dm_rows.append({"dataset": args.dataset, "rv_col": args.rv_col,
                                        "window": wname, "h": int(h), "loss": lname,
                                        "aggregator": agg, "model": model,
                                        "benchmark": args.benchmark, "dm_stat": stat,
                                        "p_value": p, "n_origins": int(len(wide))})

                    robust_ok = agg == "mean" or wname in ROBUST_TEST_WINDOWS
                    if len(wide) < 60 or not robust_ok:
                        continue
                    # The real-time calibrated variants are a robustness check on
                    # calibration, not candidate models: kept out of MCS and SPA.
                    cand = wide.loc[:, ~wide.columns.str.endswith("+rtcal")]
                    if not args.no_mcs:
                        try:
                            m = model_confidence_set(cand, alpha=args.alpha, reps=args.reps)
                            for lab, row in m.iterrows():
                                mcs_rows.append({"dataset": args.dataset,
                                                 "rv_col": args.rv_col, "window": wname,
                                                 "h": int(h), "loss": lname,
                                                 "aggregator": agg, "model": lab,
                                                 "mcs_pvalue": float(row.mcs_pvalue),
                                                 "in_mcs": bool(row.in_mcs)})
                        except Exception as exc:  # noqa: BLE001
                            print(f"  MCS failed ({wname}, h={h}, {lname}, {agg}): "
                                  f"{type(exc).__name__}: {exc}")
                    sets = {"full": cand}
                    if agg == "mean" and wname in ("clean", "frontier", "stress"):
                        sets["original"] = cand.loc[:, ~cand.columns.map(base_name)
                                                    .isin(REVISION_MODELS)]
                    for set_name, c in sets.items():
                        try:
                            spa_rows.append({"dataset": args.dataset, "rv_col": args.rv_col,
                                             "window": wname, "h": int(h), "loss": lname,
                                             "aggregator": agg, "set": set_name,
                                             "n_origins": int(len(c)),
                                             **spa_test(c, args.benchmark, reps=args.reps)})
                        except Exception as exc:  # noqa: BLE001
                            print(f"  SPA failed ({wname}, h={h}, {lname}, {agg}): "
                                  f"{type(exc).__name__}: {exc}")
            print(f"  {wname:<9} h={h:<3} {s.label.nunique()} models, "
                  f"{s.origin.nunique()} origins", flush=True)

    pd.DataFrame(loss_rows).to_csv(TABLES / f"losses_{tag}.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(TABLES / f"dm_{tag}.csv", index=False)
    pd.DataFrame(mcs_rows).to_csv(TABLES / f"mcs_{tag}.csv", index=False)
    pd.DataFrame(spa_rows).to_csv(TABLES / f"spa_{tag}.csv", index=False)
    pd.DataFrame(med_rows).to_csv(TABLES / f"medtest_{tag}.csv", index=False)
    pd.DataFrame(win_rows).to_csv(TABLES / f"windows_{tag}.csv", index=False)
    for tmp, name in (("_cal_factors_tmp.csv", f"calfactors_{tag}.csv"),
                      ("_rtcal_counts_tmp.csv", f"rtcal_{tag}.csv"),
                      ("_common_tmp.csv", f"common_{tag}.csv")):
        if (TABLES / tmp).exists():
            (TABLES / tmp).rename(TABLES / name)

    if args.rv_col == "rv":
        costs = load_costs(args.out_root, dataset=args.dataset)
        # The robustness runs share model names; the cost table is the main study's.
        costs = costs[costs["rv_col"].fillna("rv") == "rv"]
        costs.to_csv(TABLES / f"cost_{args.dataset}.csv", index=False)

        panel = pd.read_parquet(panel_path)
        gmap = panel.drop_duplicates("unique_id").set_index("unique_id")["group"]
        grp = df[windows["clean"][0]].assign(
            group=lambda d: d["unique_id"].map(gmap),
            qlike=lambda d: LOSSES["qlike"](d.y.to_numpy(), d.y_hat.to_numpy()))
        (grp.groupby(["group", "h", "label"])["qlike"].agg(["mean", "median"])
            .reset_index().rename(columns={"mean": "qlike", "median": "qlike_median"})
            .to_csv(TABLES / f"bygroup_{args.dataset}.csv", index=False))
        # Per asset, for the comparison with single-index studies (SPY and QQQ
        # track the S&P 500 and the Nasdaq-100). Seeds averaged per key first.
        (grp.groupby(["unique_id", "h", "label", "origin"])["qlike"].mean()
            .groupby(["unique_id", "h", "label"]).agg(["mean", "median"]).reset_index()
            .rename(columns={"mean": "qlike", "median": "qlike_median"})
            .to_csv(TABLES / f"byasset_{args.dataset}.csv", index=False))

    print(f"\nwrote {len(loss_rows)} loss rows, {len(dm_rows)} DM tests, "
          f"{len(mcs_rows)} MCS rows, {len(spa_rows)} SPA tests, "
          f"{len(med_rows)} median tests to {TABLES}/ (tag {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
