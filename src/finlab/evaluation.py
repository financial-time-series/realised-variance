"""Losses and significance tests for volatility forecasts.

Three decisions are made here, and each one is a place where a plausible-looking
alternative would produce a wrong answer.

1. THE LOSSES ARE QLIKE AND MSE, AND NOTHING ELSE IS PRIMARY.
   The target is a noisy proxy for latent variance, not variance itself.
   Patton (2011) shows that under a conditionally unbiased proxy only a
   restricted family of losses ranks forecasts consistently -- the ranking a
   loss delivers on the proxy equals the ranking it would deliver on the truth.
   MSE and QLIKE are the two members of that family in common use. MAE, MAPE
   and R-squared on volatility are NOT robust in this sense: they can and do
   reverse rankings when the proxy is noisy, which is exactly our situation on
   the EQ panel. MAE on log RV is reported, clearly labelled, as a
   non-robust descriptive statistic only.

2. SIGNIFICANCE IS TESTED ON THE CROSS-SECTIONALLY AVERAGED LOSS DIFFERENTIAL.
   Assets are strongly correlated -- in a crisis every series is volatile at
   once. Pooling 60 series x 4,000 origins into one Diebold-Mariano test treats
   240,000 correlated observations as independent and will call almost anything
   significant. Averaging the differential across assets first collapses the
   cross-section into one time series whose dependence is purely serial, which
   is what the Newey-West correction is built for.

3. OVERLAPPING WINDOWS ARE CORRECTED, NOT IGNORED.
   With daily origins and h=22 the target windows share 21 of 22 days, so the
   loss differentials are serially correlated by construction. The plain DM
   statistic over-rejects badly here. Harvey-Leybourne-Newbold rescale it for
   the finite-sample bias and refer it to a t distribution; the long-run
   variance uses Newey-West at h-1 lags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# --- losses ------------------------------------------------------------------


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """QLIKE, normalised to zero at a perfect forecast.

        L = y/f - log(y/f) - 1

    Asymmetric on purpose: under-forecasting variance is penalised far harder
    than over-forecasting it, which is the correct asymmetry for risk
    management and the reason QLIKE, not MSE, is the headline loss here.
    """
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    r = y / f
    return r - np.log(r) - 1.0


def mse(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    return (y - f) ** 2


def mae_log(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """NOT proxy-robust. Descriptive only -- see the module docstring."""
    return np.abs(np.log(np.asarray(y, dtype=np.float64))
                  - np.log(np.asarray(f, dtype=np.float64)))


def madl(y: np.ndarray, f: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Mean Absolute Directional Loss, per observation (Michankow, Sakowski and
    Slepaczuk 2024), applied to the DIRECTION OF THE VARIANCE MOVE.

        L = -sign(R * Rhat) * |R|,   R = log(y / base),  Rhat = log(f / base)

    where `base` is the realised variance averaged over the h days up to and
    including the origin: the "no change" reference for an h-day average. The
    loss is -|R| when the forecast calls the direction of the move right and
    +|R| when it calls it wrong, so its mean is bounded by the size of the moves
    and lower is better. It is outside Patton's proxy-robust class and is
    reported as a check from a different family of losses: it charges a
    near-zero variance forecast for calling the direction, never for its size,
    so the y/f blow-up that drives QLIKE cannot occur under it.
    """
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    b = np.asarray(base, dtype=np.float64)
    r = np.log(y / b)
    rhat = np.log(f / b)
    return -np.sign(r * rhat) * np.abs(r)


LOSSES = {"qlike": qlike, "mse": mse, "mae_log": mae_log}
ROBUST_LOSSES = ("qlike", "mse")
# Losses that need the "no change" reference as a third argument.
DIRECTIONAL_LOSSES = {"madl": madl}

# Forecasts must be strictly positive: QLIKE takes their log, and a variance
# forecast of zero is not a forecast. Models that can emit non-positive values
# (OLS on RV levels, most obviously) are floored at this multiple of the
# smallest observed variance in their own training window -- recorded in
# config.json, never applied silently.
FLOOR_FRACTION = 1e-4


def floor_forecasts(f: np.ndarray, train_min: float) -> tuple[np.ndarray, int]:
    """Clip non-positive forecasts. Returns the array and how many were clipped."""
    floor = max(train_min * FLOOR_FRACTION, np.finfo(np.float64).tiny)
    bad = int(np.sum(~(f > 0)))
    return np.where(f > 0, f, floor), bad


# --- Diebold-Mariano with the Harvey-Leybourne-Newbold correction ------------


def newey_west_var(d: np.ndarray, lags: int) -> float:
    """Long-run variance of the mean of d, Bartlett kernel."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    e = d - d.mean()
    gamma0 = float(e @ e) / n
    total = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma_k = float(e[k:] @ e[:-k]) / n
        total += 2.0 * w * gamma_k
    # A Bartlett-weighted sum can still go negative in small samples; falling
    # back on gamma0 is the standard remedy and keeps the statistic defined.
    return total if total > 0 else gamma0


def dm_hln(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """Test H0: equal expected loss. Returns (statistic, two-sided p-value).

    Negative statistic => model A has the lower loss => A is better.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return float("nan"), float("nan")

    var = newey_west_var(d, lags=max(h - 1, 0))
    dm = d.mean() / np.sqrt(var / n)

    # Harvey, Leybourne & Newbold (1997): the DM statistic is biased in finite
    # samples at multi-step horizons; this rescaling plus a t reference removes
    # most of the over-rejection.
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * adj
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


# --- panel helpers -----------------------------------------------------------


AGGREGATORS = ("mean", "median", "trim")
TRIM = 0.10   # cut from EACH tail of the cross-section by the "trim" aggregator


def per_origin_loss(
    preds: pd.DataFrame, loss: str, y_col: str = "y", f_col: str = "y_hat",
    agg: str = "mean", loss_col: str | None = None,
) -> pd.DataFrame:
    """Loss for every row, then aggregated across assets within each origin.

    The returned frame is indexed by origin with one column per model, which is
    the shape both the DM test and the MCS need.

    `agg` chooses the cross-sectional summary. "mean" is the conventional one,
    and the one the paper's first version tested on: a single asset with one
    catastrophic forecast moves that day's mean without bound. "median" scores
    the typical asset on each day, and "trim" the 10% trimmed mean (the most
    extreme tenth of assets dropped from each side). Both make the test speak
    to the same thing as the median columns of the results tables.

    `loss_col` names a precomputed per-row loss (needed for MADL, which takes a
    third argument); otherwise `loss` is looked up in LOSSES.
    """
    df = preds.copy()
    if loss_col is None:
        fn = LOSSES[loss]
        df["loss"] = fn(df[y_col].to_numpy(), df[f_col].to_numpy())
        loss_col = "loss"
    g = df.groupby(["origin", "model"])[loss_col]
    if agg == "mean":
        wide = g.mean()
    elif agg == "median":
        wide = g.median()
    elif agg == "trim":
        # Vectorised trimmed mean: rank within each (origin, model) group and keep
        # the ranks scipy.stats.trim_mean would keep (int(TRIM * n) cut per side).
        d = df[["origin", "model", loss_col]].copy()
        d["_n"] = d.groupby(["origin", "model"])[loss_col].transform("size")
        d["_r"] = d.groupby(["origin", "model"])[loss_col].rank(method="first") - 1
        cut = np.floor(TRIM * d["_n"]).astype(int)
        keep = (d["_r"] >= cut) & (d["_r"] < d["_n"] - cut)
        wide = d[keep].groupby(["origin", "model"])[loss_col].mean()
    else:
        raise ValueError(f"unknown aggregator {agg!r}")
    wide = wide.unstack("model").sort_index()
    # A model missing an origin makes the columns non-comparable; drop those
    # origins for every model rather than letting each model be scored on its
    # own convenient subsample.
    return wide.dropna(axis=0, how="any")


def moving_blocks(n: int, rng: np.random.Generator, block: int) -> np.ndarray:
    """Indices of one moving-block bootstrap sample of length n."""
    if n <= block:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def median_difference_test(
    losses: np.ndarray, origin_rows: list[np.ndarray], bench_col: int,
    reps: int = 1000, block: int = 22, seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Test H0: median loss of each model = median loss of the benchmark.

    `losses` is (n_rows, n_models), one row per (asset, origin) of a balanced
    panel, and `origin_rows[k]` lists the rows of the k-th origin in time order.
    The statistic is the difference in POOLED medians -- exactly what the median
    columns of the results tables report. The null distribution comes from a
    moving-block bootstrap over whole origins (blocks of `block`, carrying the
    cross-section with each origin, as the DM test's averaging does), recentred
    on the observed difference; the p-value is two-sided.

    Returns (difference, p_value), one entry per model column (NaN for the
    benchmark itself).
    """
    rng = np.random.default_rng(seed)
    med = np.median(losses, axis=0)
    obs = med - med[bench_col]
    draws = np.empty((reps, losses.shape[1]))
    n = len(origin_rows)
    for b in range(reps):
        pick = moving_blocks(n, rng, block)
        rows = np.concatenate([origin_rows[k] for k in pick])
        m = np.median(losses[rows], axis=0)
        draws[b] = m - m[bench_col]
    null = draws - draws.mean(axis=0)
    p = (np.abs(null) >= np.abs(obs)[None, :]).mean(axis=0)
    p[bench_col] = np.nan
    obs = obs.astype(float)
    obs[bench_col] = np.nan
    return obs, p


def model_confidence_set(
    losses: pd.DataFrame, alpha: float = 0.10, block_size: int | None = None,
    reps: int = 1000, seed: int = 1,
) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS: the set of models not distinguishable from best.

    Answers the question a table of point estimates cannot: given the sampling
    noise, which models can be excluded from being the best? Reported alongside
    pairwise tests because with twenty models the pairwise comparisons multiply
    and their individual p-values stop meaning what they appear to mean.
    """
    from arch.bootstrap import MCS

    # Two models with bit-identical loss series give their pair a variance of
    # zero, and the MCS divides by it: the procedure dies with an opaque
    # IndexError from deep inside the bootstrap. That is worth catching for a
    # second reason -- identical losses almost always mean a BUG, not a
    # coincidence. In this project an "EGARCH" column turned out to be the
    # random walk at every multi-step horizon, and this was the symptom.
    dup = losses.T.duplicated()
    if dup.any():
        twins = list(losses.columns[dup.to_numpy()])
        print(f"  WARNING: identical loss series, excluded from the MCS: "
              f"{twins}. Two models scoring identically is normally a bug -- "
              f"check that each is the model it claims to be.")
        losses = losses.loc[:, ~dup.to_numpy()]
    if losses.shape[1] < 2:
        raise ValueError("need at least two distinct models for an MCS")

    # A model whose mean loss exceeds the typical model's by orders of magnitude
    # cannot be in the set, and keeping it breaks the procedure: on the equity
    # panel one sampled Moirai-MoE path exploded upward, its mean MSE at h=22
    # reached ~1e15 against ~1e-7 for the rest, every standardised difference
    # against it tied, and arch's elimination step failed on the tie. Such
    # models are recorded as excluded (p-value 0) and the set is computed on the
    # rest.
    means = losses.mean()
    blown = ~np.isfinite(means)
    if float(losses.min().min()) >= 0:   # QLIKE, MSE; MADL is bounded and signed
        blown |= means > 1e6 * float(np.median(means))
    excluded = list(losses.columns[blown.to_numpy()])
    if excluded:
        losses = losses.loc[:, ~blown.to_numpy()]

    if block_size is None:
        # Blocks must be long enough to carry the overlap-induced dependence.
        block_size = max(22, int(len(losses) ** (1 / 3)))
    mcs = MCS(losses, size=alpha, block_size=block_size, reps=reps, seed=seed)
    mcs.compute()
    out = mcs.pvalues.copy()
    out.columns = ["mcs_pvalue"]
    if excluded:
        out = pd.concat([out, pd.DataFrame({"mcs_pvalue": 0.0}, index=excluded)])
    out["in_mcs"] = out["mcs_pvalue"] > alpha
    return out.sort_values("mcs_pvalue", ascending=False)


def spa_test(
    losses: pd.DataFrame, benchmark: str, block_size: int | None = None,
    reps: int = 1000, seed: int = 1,
) -> dict:
    """Hansen's SPA: does ANY model beat the benchmark, after data snooping?

    This is the test that answers "we tried twenty models and one of them won"
    honestly. A low p-value means the best model's advantage over the benchmark
    survives the fact that twenty were tried.
    """
    from arch.bootstrap import SPA

    if block_size is None:
        block_size = max(22, int(len(losses) ** (1 / 3)))
    others = losses.drop(columns=[benchmark])
    spa = SPA(losses[benchmark], others, block_size=block_size, reps=reps, seed=seed)
    spa.compute()
    # Which alternative has the lowest average loss: the model the test is, in
    # effect, about. Reported so a rejection can be attributed.
    best = str((others.mean() - losses[benchmark].mean()).idxmin())
    return {"benchmark": benchmark, "pvalue_consistent": float(spa.pvalues["consistent"]),
            "pvalue_lower": float(spa.pvalues["lower"]),
            "pvalue_upper": float(spa.pvalues["upper"]),
            "n_alternatives": int(others.shape[1]), "best_alternative": best}
