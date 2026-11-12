"""One function owns "what is visible at origin t". Every model goes through it.

This module exists because look-ahead bias is not caught by testing. A model
that peeks produces a *better-looking* result, not a crash, and no downstream
check will flag it. The only defence is that there is exactly one place in the
codebase where history is sliced, every runner calls it, and it is short enough
to read in one sitting.

    panel = Panel.load("data/processed/eq_rv.parquet")
    ctx   = panel.context("SPY", origin)        # RV up to AND INCLUDING origin
    y     = panel.target("SPY", origin, h=5)    # mean RV over the NEXT 5 days

THE TARGET. For horizon h the target is the average daily realised variance over
the h days following the origin,

    y(t, h) = (1/h) * sum_{i=1..h} RV_{t+i}

which is Corsi's convention for the HAR family and the reason h=5 and h=22 are
read as "the coming week" and "the coming month" rather than "the 5th day from
now". A single distant day is mostly noise; the average is the quantity a risk
desk actually needs.

TRADING TIME, NOT CALENDAR TIME. Horizons are counted in observations, not days.
EQ has no weekend rows and CX has no gaps, so "22 steps" means 22 trading days
on EQ and 22 calendar days on CX. Mixing the two conventions inside one panel
would silently change what h means mid-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Series:
    """One asset: aligned dates and realised-variance values, ascending."""

    uid: str
    dates: np.ndarray      # datetime64[ns]
    rv: np.ndarray         # float64, strictly positive
    group: str

    def __len__(self) -> int:
        return len(self.rv)


class Panel:
    """The only object allowed to slice history."""

    def __init__(self, series: dict[str, Series], rv_col: str) -> None:
        self.series = series
        self.rv_col = rv_col
        self._index: dict[str, dict[np.datetime64, int]] = {
            uid: {d: i for i, d in enumerate(s.dates)} for uid, s in series.items()
        }

    # --- construction --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, rv_col: str = "rv") -> "Panel":
        df = pd.read_parquet(path)
        if rv_col not in df.columns:
            raise KeyError(f"{rv_col!r} not in {path}; have {list(df.columns)}")
        df = df.sort_values(["unique_id", "ds"])
        if (df[rv_col] <= 0).any():
            raise ValueError(
                f"{path} contains non-positive {rv_col}; QLIKE takes a log of it. "
                "build_panel.py should have removed those rows."
            )
        series = {}
        for uid, g in df.groupby("unique_id", sort=True):
            series[uid] = Series(
                uid=uid,
                dates=g["ds"].to_numpy(dtype="datetime64[ns]"),
                rv=g[rv_col].to_numpy(dtype=np.float64),
                group=str(g["group"].iloc[0]),
            )
        return cls(series, rv_col)

    @property
    def uids(self) -> list[str]:
        return sorted(self.series)

    # --- the two accessors ---------------------------------------------------

    def context(self, uid: str, origin: np.datetime64, max_len: int | None = None) -> np.ndarray:
        """Realised variance up to and INCLUDING the origin.

        Inclusive because the origin is the last day whose close has been
        observed: at the end of day t, RV_t is computable from that day's own
        bars. Excluding it would throw away the single most informative
        observation; including anything after it is look-ahead.
        """
        s = self.series[uid]
        i = self._index[uid].get(origin)
        if i is None:
            raise KeyError(f"{uid} has no observation on {origin}")
        out = s.rv[: i + 1]
        return out[-max_len:] if max_len else out

    def target(self, uid: str, origin: np.datetime64, h: int) -> float:
        """Mean realised variance over the h observations AFTER the origin.

        Returns NaN when the series ends before the window closes, so a
        truncated final window is never scored as if it were complete.
        """
        s = self.series[uid]
        i = self._index[uid].get(origin)
        if i is None:
            raise KeyError(f"{uid} has no observation on {origin}")
        j = i + 1 + h
        if j > len(s.rv):
            return float("nan")
        return float(np.mean(s.rv[i + 1: j]))

    def target_date(self, uid: str, origin: np.datetime64, h: int) -> np.datetime64:
        """The last date inside the target window -- the `ds` of the schema."""
        s = self.series[uid]
        i = self._index[uid][origin]
        j = min(i + h, len(s.dates) - 1)
        return s.dates[j]

    # --- origins -------------------------------------------------------------

    def origins(
        self,
        uid: str,
        start: str | None = None,
        end: str | None = None,
        min_history: int = 252,
        max_h: int = 22,
        step: int = 1,
    ) -> np.ndarray:
        """Every date usable as an origin for this series.

        An origin qualifies only if it has `min_history` observations behind it
        AND `max_h` observations ahead of it. The forward requirement uses the
        LONGEST horizon on purpose: if h=1 were allowed origins that h=22 could
        not use, the horizons would be scored on different samples and their
        losses would not be comparable.
        """
        s = self.series[uid]
        lo, hi = min_history, len(s.rv) - max_h
        if hi <= lo:
            return np.array([], dtype="datetime64[ns]")
        dates = s.dates[lo:hi]
        if start is not None:
            dates = dates[dates >= np.datetime64(start)]
        if end is not None:
            dates = dates[dates <= np.datetime64(end)]
        return dates[::step]

    def frame(self) -> pd.DataFrame:
        """The panel back as a long frame -- for descriptive statistics only."""
        return pd.concat(
            [pd.DataFrame({"unique_id": s.uid, "ds": s.dates,
                           "rv": s.rv, "group": s.group})
             for s in self.series.values()],
            ignore_index=True,
        )
