"""Trivial benchmarks, the HAR family, and GARCH. CPU only.

    uv run --no-sync python scripts/run_classical.py --dataset eq \
        --models rw mean ewma har har-log har-sqrt --run-dir out/runs/eq-classical

These are the paper's real competition. The sibling project's headline changed
when the statistical baselines landed last and turned out to be the models to
beat; here they arrive first, deliberately, so that no neural result is written
up before the thing it has to beat is known.

REFIT CADENCE. Every fitted model re-estimates on an expanding window every
`--refit-every` origins (default 22, i.e. monthly). One cadence for all of them
so no model is handed more adaptivity than another, and it is a flag rather
than a constant so the sensitivity can be reported. Between refits the
parameters are frozen but the INPUTS still update daily -- that is what a desk
actually does, and it is not the same as freezing the forecast.
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
from finlab.features import (  # noqa: E402
    TRANSFORMS, forward_means, har_features, ols,
)
from finlab.io import write_run  # noqa: E402
from finlab.panel import Panel  # noqa: E402

HORIZONS = (1, 5, 22)
EWMA_LAMBDA = 0.94        # RiskMetrics
MIN_HISTORY = 252         # one year before a series may serve as an origin


# --- individual models -------------------------------------------------------

def forecast_rw(rv, X, idx, h, cfg):
    """Random walk: tomorrow's variance is today's. The benchmark that refuses
    to die, and the one the paper is obliged to include (README section 5)."""
    return rv[idx]


def forecast_mean(rv, X, idx, h, cfg):
    """Historical mean over everything visible. Unconditional, and therefore
    unbeatable-looking in a calm sample and hopeless in a crisis."""
    return np.array([rv[: i + 1].mean() for i in idx])


def forecast_ewma(rv, X, idx, h, cfg):
    """RiskMetrics EWMA. Flat term structure: the h-step forecast equals the
    one-step forecast, which is the model's honest statement about horizons."""
    lam = cfg["ewma_lambda"]
    s = np.empty(len(rv))
    s[0] = rv[0]
    for t in range(1, len(rv)):
        s[t] = lam * s[t - 1] + (1 - lam) * rv[t]
    return s[idx]


def _har_direct(rv, X, idx, h, cfg):
    """HAR estimated directly for each horizon, in the model's own space.

    `cfg["window"]`: None (the default) fits on an EXPANDING window, every
    complete training pair since the series began; an integer fits on the last
    `window` complete pairs only (HAR-log-roll, the rolling-window check).
    `cfg["iv"]`: an implied daily variance aligned with `rv`, entered in logs as
    a fourth regressor (HAR-log-IV).
    """
    tr = TRANSFORMS[cfg["transform"]]
    refit = cfg["refit_every"]
    window = cfg.get("window")

    y_raw = forward_means(rv, h)
    Xw = X.copy()
    yw = y_raw.copy()
    if tr.name != "identity":
        # Transform the regressors too: a log-space HAR regresses log RV on the
        # logs of its cascades, which is what Corsi's log specification means.
        Xw[:, 1:] = tr.forward(np.maximum(X[:, 1:], 1e-300))
        yw = tr.forward(np.maximum(y_raw, 1e-300))
    if cfg.get("iv") is not None:
        # The implied variance quoted at the close of the origin day: visible
        # at t, and forward-looking by construction (a 30-day risk-neutral
        # expectation), which is exactly what the HAR cascades lack.
        Xw = np.column_stack([Xw, np.log(np.maximum(cfg["iv"], 1e-300))])

    out = np.full(len(idx), np.nan)
    beta = None
    sigma2 = 0.0
    last_fit = -10**9

    for k, i in enumerate(idx):
        if i - last_fit >= refit or beta is None:
            # Training pairs must be COMPLETE at origin i -- see features.py.
            end = i - h + 1
            rows = np.arange(MIN_HISTORY, max(MIN_HISTORY, end))
            rows = rows[np.isfinite(Xw[rows]).all(axis=1) & np.isfinite(yw[rows])]
            if window is not None:
                rows = rows[-window:]
            if len(rows) >= 100:
                beta, sigma2 = ols(Xw[rows], yw[rows])
                last_fit = i
        if beta is None or not np.isfinite(Xw[i]).all():
            out[k] = rv[i]          # fall back to the random walk, not to NaN
            continue
        out[k] = tr.inverse(float(Xw[i] @ beta), sigma2)
    return out


def forecast_har(rv, X, idx, h, cfg):
    return _har_direct(rv, X, idx, h, {**cfg, "transform": "identity"})


def forecast_har_log(rv, X, idx, h, cfg):
    return _har_direct(rv, X, idx, h, {**cfg, "transform": "log"})


def forecast_har_sqrt(rv, X, idx, h, cfg):
    return _har_direct(rv, X, idx, h, {**cfg, "transform": "sqrt"})


# The rolling window for HAR-log-roll, in observations: about four years of
# trading days on the equity panel and 2.7 years of calendar days on crypto.
ROLL_WINDOW = 1000


def forecast_har_log_roll(rv, X, idx, h, cfg):
    """HAR-log on a ROLLING window. The benchmark every ratio in the paper is
    taken against is fitted on an expanding window, so in 2025-26 its
    coefficients are dominated by two decades of older data; if a sluggish
    benchmark flattered or penalised the post-release ratios, this variant
    exposes it."""
    return _har_direct(rv, X, idx, h, {**cfg, "transform": "log",
                                       "window": cfg.get("roll_window", ROLL_WINDOW)})


def forecast_har_log_iv(rv, X, idx, h, cfg, iv=None, **_):
    """HAR-log plus the log of the implied daily variance (VIX), equity only."""
    if iv is None:
        raise SystemExit("har-log-iv needs data/processed/vix.parquet (build_extra.py)")
    return _har_direct(rv, X, idx, h, {**cfg, "transform": "log", "iv": iv})


# Blocks whose fit or forecast failed and fell back to the random walk. A
# fallback is not harmless: it makes the model SECRETLY ANOTHER MODEL. The
# first run of this campaign produced an "EGARCH" column that was, at h=5 and
# h=22, bit-for-bit identical to the random walk, because `arch` has no
# analytic multi-step forecast for EGARCH and every one of the 23,052 blocks
# quietly took the except branch. It was invisible in the results table -- just
# a plausible row of numbers. Counted here and enforced in main().
FALLBACKS: dict[str, int] = {}

# One fit per (model, series, block) serves every horizon: the variance path is
# computed once out to the longest horizon and sliced. Simulation forecasts are
# far too expensive to repeat per horizon.
_GARCH_CACHE: dict[tuple, np.ndarray] = {}


def _garch(rv, X, idx, h, cfg, returns, dates, uid, vol_model, max_h):
    """GARCH / EGARCH on daily returns.

    Fitted on returns, not on RV: that is what these models are, and swapping
    in RV would produce a different (and non-standard) animal. `last_obs` fixes
    the parameters at the refit point while arch keeps conditioning on observed
    returns for later origins -- the same frozen-parameters / updating-inputs
    protocol the HAR family uses above.

    EGARCH needs `method="simulation"`. Its multi-step conditional variance has
    no closed form because the recursion is linear in log variance, so arch
    refuses analytically beyond one step. GARCH(1,1) is analytic and exact, and
    is left that way because simulation would only add noise and cost.
    """
    from arch import arch_model

    refit = cfg["refit_every"]
    sims = cfg.get("garch_simulations", 200)
    r = pd.Series(np.asarray(returns) * 100.0, index=pd.DatetimeIndex(dates))
    r = r.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    out = np.full(len(idx), np.nan)
    for b in range(0, len(idx), refit):
        block = idx[b: b + refit]
        i0 = int(block[0])
        ckey = (vol_model, uid, i0)

        if ckey not in _GARCH_CACHE:
            try:
                am = arch_model(r, mean="Constant", vol=vol_model, p=1, q=1,
                                o=1 if vol_model == "EGARCH" else 0,
                                dist="normal")
                res = am.fit(last_obs=r.index[i0], disp="off",
                             show_warning=False)
                kw = ({"method": "simulation", "simulations": sims}
                      if vol_model == "EGARCH" and max_h > 1 else {})
                fc = res.forecast(horizon=max_h, start=r.index[i0],
                                  reindex=False, **kw)
                _GARCH_CACHE[ckey] = fc.variance.to_numpy() / 10000.0
            except Exception as exc:  # noqa: BLE001 - one block, recorded loudly
                FALLBACKS[vol_model] = FALLBACKS.get(vol_model, 0) + 1
                if FALLBACKS[vol_model] <= 3:
                    print(f"    {vol_model} failed at {uid} index {i0}: "
                          f"{type(exc).__name__}: {exc}")
                _GARCH_CACHE[ckey] = None

        var = _GARCH_CACHE[ckey]
        if var is None:
            out[b: b + len(block)] = rv[block]
            continue
        # Average the per-step variances: the target is the MEAN daily variance
        # over the window, not the variance on its final day.
        avg = var[:, :h].mean(axis=1)
        pos = block - i0
        take = pos[pos < len(avg)]
        out[b: b + len(take)] = avg[take]

    bad = ~np.isfinite(out)
    if bad.any():
        FALLBACKS[vol_model] = FALLBACKS.get(vol_model, 0) + int(bad.sum())
        out[bad] = rv[idx[bad]]
    return out


def forecast_garch(rv, X, idx, h, cfg, **kw):
    return _garch(rv, X, idx, h, cfg, vol_model="GARCH", **kw)


def forecast_egarch(rv, X, idx, h, cfg, **kw):
    return _garch(rv, X, idx, h, cfg, vol_model="EGARCH", **kw)


def forecast_garch_oc(rv, X, idx, h, cfg, **kw):
    """GARCH(1,1) on OPEN-TO-CLOSE returns (equity). The Garman-Klass target
    measures trading-hours variance; close-to-close returns add the overnight
    move, about a third of their variance on the median series here."""
    return _garch(rv, X, idx, h, cfg, vol_model="GARCH", **kw)


MODELS = {
    "rw": forecast_rw,
    "mean": forecast_mean,
    "ewma": forecast_ewma,
    "har": forecast_har,
    "har-log": forecast_har_log,
    "har-sqrt": forecast_har_sqrt,
    "garch": forecast_garch,
    "egarch": forecast_egarch,
    "garch-oc": forecast_garch_oc,
    "har-log-roll": forecast_har_log_roll,
    "har-log-iv": forecast_har_log_iv,
}
NEEDS_RETURNS = {"garch", "egarch", "garch-oc"}
NEEDS_IV = {"har-log-iv"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--models", nargs="+", default=["rw", "mean", "ewma",
                                                    "har", "har-log", "har-sqrt"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rv-col", default="rv")
    ap.add_argument("--origin-start", default=None)
    ap.add_argument("--refit-every", type=int, default=22)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N series -- smoke tests")
    ap.add_argument("--origin-step", type=int, default=1)
    args = ap.parse_args()

    panel_path = Path(f"data/processed/{args.dataset}_rv.parquet")
    panel = Panel.load(panel_path, rv_col=args.rv_col)
    uids = panel.uids[: args.limit] if args.limit else panel.uids
    origin_start = args.origin_start or ("2010-01-01" if args.dataset == "eq"
                                         else "2020-01-01")
    cfg_common = {"refit_every": args.refit_every, "ewma_lambda": EWMA_LAMBDA}

    raw = pd.read_parquet(panel_path)
    if "garch-oc" in args.models:
        if args.dataset != "eq":
            raise SystemExit("garch-oc is defined for the equity panel only")
        extra = pd.read_parquet("data/processed/eq_returns.parquet")
        raw = raw.merge(extra[["unique_id", "ds", "ret_oc"]], on=["unique_id", "ds"],
                        how="left")

    vix = None
    if any(m in NEEDS_IV for m in args.models):
        if args.dataset != "eq":
            raise SystemExit("har-log-iv is defined for the equity panel only: no "
                             "implied-volatility index covers the crypto sample")
        vix = (pd.read_parquet("data/processed/vix.parquet")
                 .set_index("ds")["iv_daily"].sort_index())

    def iv_for(dates: np.ndarray) -> np.ndarray:
        """VIX implied daily variance on each date, last quote carried forward."""
        return vix.reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy()

    max_h = max(args.horizons)

    rows, timing = [], {}
    for model in args.models:
        if model not in MODELS:
            raise SystemExit(f"unknown model {model!r}; have {sorted(MODELS)}")
        fn = MODELS[model]
        t0 = time.perf_counter()
        n_floored = 0

        for uid in uids:
            s = panel.series[uid]
            rv = s.rv
            X = har_features(rv)
            origins = panel.origins(uid, start=origin_start,
                                    min_history=MIN_HISTORY, max_h=max_h,
                                    step=args.origin_step)
            if len(origins) == 0:
                continue
            pos = np.array([panel._index[uid][d] for d in origins])

            kw = {}
            if model in NEEDS_RETURNS:
                g = raw[raw.unique_id == uid].sort_values("ds")
                ret_col = "ret_oc" if model == "garch-oc" else "ret"
                kw = {"returns": g[ret_col].to_numpy(), "dates": g["ds"].to_numpy(),
                      "uid": uid, "max_h": max_h}
            if model in NEEDS_IV:
                kw = {"iv": iv_for(s.dates)}

            for h in args.horizons:
                y_hat = np.asarray(fn(rv, X, pos, h, cfg_common, **kw),
                                   dtype=np.float64)
                y_hat, bad = floor_forecasts(y_hat, float(rv[:MIN_HISTORY].min()))
                n_floored += bad
                y_true = np.array([np.mean(rv[i + 1: i + 1 + h]) for i in pos])
                rows.append(pd.DataFrame({
                    "unique_id": uid,
                    "origin": origins,
                    "ds": [s.dates[min(i + h, len(s.dates) - 1)] for i in pos],
                    "h": h,
                    "y": y_true,
                    "y_hat": y_hat,
                    "model": model,
                }))

        timing[model] = round(time.perf_counter() - t0, 2)
        fb = FALLBACKS.get({"garch": "GARCH", "egarch": "EGARCH",
                            "garch-oc": "GARCH"}.get(model, ""), 0)
        print(f"{model:<10} {timing[model]:>8.2f}s"
              + (f"   ({n_floored} forecasts floored)" if n_floored else "")
              + (f"   ({fb} FALLBACKS TO RW)" if fb else ""))
        _GARCH_CACHE.clear()

    # A model that fell back to the random walk on a large share of its blocks
    # is not that model any more. Refuse to write it rather than let a row of
    # plausible numbers into the results table under a name it has not earned.
    for vm, n in FALLBACKS.items():
        share = n / max(len(uids) * len(args.horizons) *
                        max(1, 4000 // args.refit_every), 1)
        if share > 0.02:
            raise SystemExit(
                f"REFUSING TO WRITE: {vm} fell back to the random walk on {n} "
                f"blocks (~{share:.1%}). A fallback makes the model secretly "
                f"another model; fix the forecast method rather than shipping "
                f"this. See the first traceback above.")

    preds = pd.concat(rows, ignore_index=True)
    write_run(
        args.run_dir, preds,
        config={
            "dataset": args.dataset, "family": "classical",
            "models": args.models, "rv_col": args.rv_col,
            "origin_start": origin_start, "origin_step": args.origin_step,
            "refit_every": args.refit_every, "horizons": args.horizons,
            "min_history": MIN_HISTORY, "ewma_lambda": EWMA_LAMBDA,
            "n_series": len(uids), "limit": args.limit,
            # Recorded so a later reader can confirm the model is the model:
            # a non-zero count here means some forecasts are the random walk.
            "fallbacks_to_rw": dict(FALLBACKS),
            "garch_simulations": 200,
            "roll_window": ROLL_WINDOW,
            "iv_source": "FRED VIXCLS, (VIX/100)^2/252" if vix is not None else None,
        },
        timing={"fit_predict_seconds": round(sum(timing.values()), 2),
                "per_model_seconds": timing, "device": "cpu",
                "gpu": None, "cpu": platform.processor() or platform.machine()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
