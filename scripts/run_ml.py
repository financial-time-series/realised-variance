"""Ridge, LightGBM and random forest on HAR-style features. CPU only.

    uv run --no-sync python scripts/run_ml.py --dataset eq --model lightgbm \
        --run-dir out/runs/eq-lightgbm

GLOBAL, NOT PER-SERIES. One model is fitted across the whole panel, unlike the
HAR family which is fitted per asset. That is the standard modern setup and it
is also the only way the tree models get enough data, but it forces a scale
decision: realised variance differs by an order of magnitude between a utility
ETF and a small-cap, and a pooled tree would spend its capacity learning which
asset it is looking at.

The fix is to learn in RELATIVE LOG SPACE. Every feature and the target are
expressed as log deviations from the asset's own monthly cascade RV^(22)_t,
which is observable at the origin. The model therefore predicts "how much
higher than its own recent average will this asset's variance be", the level is
restored afterwards, and nothing about the normalisation looks forward.

TRAINING ROWS ARE SELECTED BY TARGET END DATE, not by origin date. A row may be
trained on only once its target window has entirely closed on or before the
refit date. Filtering on the origin instead would admit rows whose targets
extend past the refit point -- the same leak features.py warns about, and the
one that is easy to reintroduce here because the panel is pooled.
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
CASCADES = (1, 5, 22, 66)     # daily, weekly, monthly, quarterly


def build_frame(panel: Panel, uids, horizons, origin_start, max_h):
    """One tidy frame of features and targets for the whole panel."""
    parts = []
    for uid in uids:
        s = panel.series[uid]
        rv = s.rv
        T = len(rv)
        log_rv = np.log(rv)
        csum = np.concatenate([[0.0], np.cumsum(rv)])

        casc = {}
        for lag in CASCADES:
            m = np.full(T, np.nan)
            idx = np.arange(lag - 1, T)
            m[idx] = (csum[idx + 1] - csum[idx + 1 - lag]) / lag
            casc[lag] = m

        scale = np.log(casc[22])          # the visible-at-t normaliser
        feats = {f"c{lag}": np.log(casc[lag]) - scale for lag in CASCADES}
        # Dispersion of recent log RV: a volatility-of-volatility signal the
        # HAR cascades cannot express.
        s_lrv = pd.Series(log_rv)
        feats["vol_of_vol"] = s_lrv.rolling(22).std().to_numpy()
        feats["scale"] = scale

        for h in horizons:
            y = np.full(T, np.nan)
            t = np.arange(T)
            ok = t + h < T
            y[ok] = (csum[t[ok] + 1 + h] - csum[t[ok] + 1]) / h

            df = pd.DataFrame({**feats, "y": y})
            df["unique_id"] = uid
            df["h"] = h
            df["origin"] = s.dates
            # The last date inside the target window: the row is trainable only
            # once this date has passed.
            df["ds"] = np.concatenate([s.dates[h:], np.repeat(s.dates[-1], h)])
            df["target_ready"] = df["ds"]
            df["rv_now"] = rv
            parts.append(df.iloc[MIN_HISTORY:])
    frame = pd.concat(parts, ignore_index=True)
    return frame.dropna(subset=["y", "scale"] + [f"c{l}" for l in CASCADES])


FEATURES = [f"c{l}" for l in CASCADES] + ["vol_of_vol", "scale"]


def make_model(name: str, seed: int):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1)
    if name == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=20, max_features=0.6,
            random_state=seed, n_jobs=-1)
    raise SystemExit(f"unknown model {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--model", required=True,
                    choices=["ridge", "lightgbm", "randomforest"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rv-col", default="rv")
    ap.add_argument("--origin-start", default=None)
    ap.add_argument("--refit-every", type=int, default=22)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--origin-step", type=int, default=1)
    args = ap.parse_args()

    panel = Panel.load(f"data/processed/{args.dataset}_rv.parquet", rv_col=args.rv_col)
    uids = panel.uids[: args.limit] if args.limit else panel.uids
    origin_start = args.origin_start or ("2010-01-01" if args.dataset == "eq"
                                         else "2020-01-01")
    max_h = max(args.horizons)

    t0 = time.perf_counter()
    frame = build_frame(panel, uids, args.horizons, origin_start, max_h)
    print(f"feature frame: {len(frame):,} rows")

    out_parts, n_fits = [], 0
    for h in args.horizons:
        fh = frame[frame.h == h].sort_values("origin").reset_index(drop=True)
        # Origins actually scored: inside the evaluation window and with a full
        # target window ahead of them.
        evalmask = (fh.origin >= np.datetime64(origin_start))
        origins = np.sort(fh.loc[evalmask, "origin"].unique())[:: args.origin_step]
        if len(origins) == 0:
            continue

        # log-relative target: log(y) - log(RV^(22)_t)
        y_rel = np.log(fh["y"].to_numpy()) - fh["scale"].to_numpy()
        X_all = fh[FEATURES].to_numpy()
        preds = np.full(len(fh), np.nan)
        sigma2 = np.zeros(len(origins))
        s2_by_origin: dict = {}

        blocks = range(0, len(origins), args.refit_every)
        for b in blocks:
            block = origins[b: b + args.refit_every]
            fit_date = block[0]
            # Trainable rows: target window CLOSED strictly before the origin
            # this block starts from.
            tr = (fh["ds"].to_numpy() < fit_date) & np.isfinite(y_rel)
            if tr.sum() < 500:
                continue
            model = make_model(args.model, args.seed)
            model.fit(X_all[tr], y_rel[tr])
            n_fits += 1
            # Residual variance of THIS fit, for the retransformation bias.
            # In-sample, and from training rows only, so it looks nowhere
            # forward. See features.py: exp(mu) is a median, not a mean.
            resid = y_rel[tr] - model.predict(X_all[tr])
            sigma2[b: b + args.refit_every] = float(np.var(resid))
            te = np.isin(fh["origin"].to_numpy(), block)
            if te.any():
                preds[te] = model.predict(X_all[te])
                s2_by_origin.update({o: float(np.var(resid)) for o in block})

        fh["y_hat_rel"] = preds
        fh["sigma2"] = fh["origin"].map(s2_by_origin).astype(float).fillna(0.0)
        scored = fh[np.isin(fh["origin"].to_numpy(), origins) & np.isfinite(preds)].copy()
        y_hat = np.exp(scored["y_hat_rel"].to_numpy()
                       + scored["scale"].to_numpy()
                       + 0.5 * scored["sigma2"].to_numpy())
        y_hat, _ = floor_forecasts(y_hat, float(np.nanmin(frame["y"])))
        out_parts.append(pd.DataFrame({
            "unique_id": scored["unique_id"].to_numpy(),
            "origin": scored["origin"].to_numpy(),
            "ds": scored["ds"].to_numpy(),
            "h": h,
            "y": scored["y"].to_numpy(),
            "y_hat": y_hat,
            "model": args.model,
            "seed": args.seed,
        }))
        print(f"  h={h:>2}: {len(scored):,} forecasts")

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"{args.model}: {elapsed}s, {n_fits} fits")

    write_run(
        args.run_dir, pd.concat(out_parts, ignore_index=True),
        config={"dataset": args.dataset, "family": "ml", "model": args.model,
                "rv_col": args.rv_col, "origin_start": origin_start,
                "refit_every": args.refit_every, "horizons": args.horizons,
                "features": FEATURES, "cascades": list(CASCADES),
                "target": "log(y) - log(RV22_t)", "global_model": True,
                "seed": args.seed, "n_series": len(uids), "n_fits": n_fits,
                "origin_step": args.origin_step, "limit": args.limit},
        timing={"fit_predict_seconds": elapsed, "device": "cpu", "gpu": None,
                "cpu": platform.processor() or platform.machine(),
                "n_fits": n_fits},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
