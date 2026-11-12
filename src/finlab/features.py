"""HAR features, rolling-origin training pairs, and the transform round-trip.

Two traps live here, both of the kind that produce a better-looking result
rather than an error.

TRAINING TARGETS MUST BE FULLY OBSERVED AT THE ORIGIN. To fit a model at origin
t for horizon h, the usable training pairs end at t-h, not t: the target of
origin t-h+1 extends to t+1, which has not happened yet. Training on pairs
through t leaks up to h days of the future into every fitted coefficient. The
effect is largest exactly where it matters most -- at h=22, where it would hand
the model three weeks of hindsight.

RETRANSFORMATION IS NOT EXPONENTIATION. A model fitted on log RV predicts
E[log RV]. Exponentiating that gives the conditional MEDIAN, which under
lognormal errors sits below the mean by a factor exp(sigma^2/2). Since QLIKE
punishes under-forecasting hard, the naive round-trip systematically handicaps
every log-space model -- and log-space models are the ones the volatility
literature actually uses, so the bias would fall on the strongest baselines.
"""

from __future__ import annotations

import numpy as np

# Corsi's (2009) three cascades: yesterday, last week, last month.
HAR_LAGS = (1, 5, 22)


def har_features(rv: np.ndarray, lags: tuple[int, ...] = HAR_LAGS) -> np.ndarray:
    """(T, 1 + len(lags)) design matrix. Row t uses only rv[..t] inclusive.

    Rows before the longest lag are filled with NaN and must be excluded by the
    caller; they are not silently back-filled, because a back-filled row is a
    fabricated observation.
    """
    rv = np.asarray(rv, dtype=np.float64)
    T = len(rv)
    X = np.full((T, 1 + len(lags)), np.nan)
    X[:, 0] = 1.0
    csum = np.concatenate([[0.0], np.cumsum(rv)])
    for j, lag in enumerate(lags, start=1):
        # mean of rv[t-lag+1 .. t] -- inclusive of t, the last observed day
        idx = np.arange(T)
        valid = idx >= lag - 1
        X[valid, j] = (csum[idx[valid] + 1] - csum[idx[valid] + 1 - lag]) / lag
    return X


def forward_means(rv: np.ndarray, h: int) -> np.ndarray:
    """y[t] = mean(rv[t+1 .. t+h]); NaN where the window runs off the end."""
    rv = np.asarray(rv, dtype=np.float64)
    T = len(rv)
    csum = np.concatenate([[0.0], np.cumsum(rv)])
    y = np.full(T, np.nan)
    t = np.arange(T)
    ok = t + h < T
    y[ok] = (csum[t[ok] + 1 + h] - csum[t[ok] + 1]) / h
    return y


def training_slice(t_idx: int, h: int, min_history: int) -> slice:
    """Rows usable to FIT at origin index t_idx for horizon h.

    Ends at t_idx - h inclusive. See the module docstring: this single
    subtraction is the difference between an honest model and one that has seen
    the evaluation period.
    """
    return slice(min_history, max(min_history, t_idx - h + 1))


# --- transforms and their honest inverses ------------------------------------

class Transform:
    """A model's working space, and the way back to the RV level.

    `inverse` takes the residual variance of the fit so the retransformation
    bias can be removed. Passing sigma2=0 gives the naive inverse, which is
    what a paper that does not think about this is implicitly using.
    """

    name = "identity"

    def forward(self, y: np.ndarray) -> np.ndarray:
        return y

    def inverse(self, z: np.ndarray, sigma2: float = 0.0) -> np.ndarray:
        return z


class LogTransform(Transform):
    name = "log"

    def forward(self, y):
        return np.log(y)

    def inverse(self, z, sigma2=0.0):
        # E[y] = exp(mu + sigma^2/2) under lognormal errors.
        return np.exp(np.clip(z + 0.5 * sigma2, -700, 700))


class SqrtTransform(Transform):
    name = "sqrt"

    def forward(self, y):
        return np.sqrt(y)

    def inverse(self, z, sigma2=0.0):
        # E[y] = E[z]^2 + Var(z) for y = z^2.
        return np.maximum(z, 0.0) ** 2 + sigma2


TRANSFORMS = {"identity": Transform(), "log": LogTransform(), "sqrt": SqrtTransform()}


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with the residual variance the inverse transform needs."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    return beta, float(resid @ resid / dof)
