"""Run directories: one shape for every model family.

The single most load-bearing convention in this repository. Every runner --
a linear baseline on CPU, a gradient-boosted tree, a transformer on an A100 --
writes the same three files into its own directory:

    <run-dir>/config.json        exact arguments, resolved settings, git commit
    <run-dir>/predictions.parquet  the canonical schema below
    <run-dir>/timing.json        wall-clock, device, GPU model

Because the schema is shared, `load_campaign()` concatenates a whole campaign
into one frame and the scoring code contains no model-specific branching.
Adding a model later is a new script, not a new analysis path.

CANONICAL SCHEMA -- adapt the column names once, at the start of a project,
then never again:

    unique_id   str        asset / series identifier
    origin      timestamp  last observation the model was allowed to see
    ds          timestamp  the period being predicted
    h           int        horizon, in periods, from origin to ds
    y_hat       float      point prediction
    model       str        architecture name (NOT including the seed)
    [seed]      int        present only for stochastic models
    [q10, q90]  float      predictive quantiles, where the model emits them

`model` holds the architecture and `seed` is a separate column: seeds are
replicates of one model, not distinct models, and collapsing them later is only
possible if they were never merged into the name.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"unique_id", "origin", "ds", "h", "y_hat", "model"}


def _git_commit() -> str:
    """The commit the code came from, on either machine.

    The remote tree is an rsync copy and NOT a git repository (README, section
    1), so `git rev-parse` there returns nothing and every cluster run would
    record its provenance as "unknown" -- which is exactly the run you later
    need to trace. The sync step drops a `.git_commit` file alongside the code;
    it is read here when git itself is unavailable.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        pass
    for parent in [Path.cwd(), *Path.cwd().parents]:
        stamp = parent / ".git_commit"
        if stamp.exists():
            return stamp.read_text().strip() or "unknown"
    return "unknown"


def write_run(
    run_dir: str | Path,
    predictions: pd.DataFrame,
    config: dict[str, Any],
    timing: dict[str, Any],
) -> Path:
    """Write one run. Fails loudly if the schema is wrong.

    The schema check is here, at write time, deliberately: a malformed run
    discovered during aggregation means the job has to be resubmitted and
    requeued, which on a busy cluster costs hours.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction frame is missing {sorted(missing)}")

    # The CPU model, read from the kernel: platform.processor() returns only
    # "x86_64" on Linux, which cannot place a CPU timing on any hardware.
    if "cpu_model" not in timing:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    timing = {**timing, "cpu_model": line.split(":", 1)[1].strip()}
                    break
        except OSError:
            pass
    timing = {**timing, "host": platform.node()}

    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            {**config, "git_commit": _git_commit(), "host": platform.node()},
            indent=2, default=str,
        )
    )
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2, default=str))

    print(f"\nwrote {run_dir}/predictions.parquet  "
          f"({len(predictions):,} rows, {predictions['model'].nunique()} models, "
          f"{predictions['unique_id'].nunique()} series)")
    return run_dir


def load_campaign(out_root: str | Path, dataset: str | None = None,
                  rv_col: str | None = None, extra_runs=None) -> pd.DataFrame:
    """Concatenate every run under `out_root` into one frame.

    `dataset` is read from each run's config.json, never inferred from the
    identifiers: two datasets in the same project usually share an id scheme
    (a ticker is a ticker), so filtering on ids would silently mix them.
    """
    out_root = Path(out_root)
    parts = []
    for f in sorted(out_root.rglob("predictions.parquet")):
        cfg_path = f.parent / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        if dataset is not None and cfg.get("dataset") != dataset:
            continue
        # The realised-variance measure a run was fed and scored on. The main
        # campaign uses `rv` (Garman-Klass on equities); the robustness runs
        # use Parkinson or Rogers-Satchell. `extra_runs(cfg)` may admit a run
        # fed another measure whose forecasts do not depend on it (GARCH, which
        # reads returns); the caller then re-scores its target.
        run_rv = cfg.get("rv_col", "rv")
        if rv_col is not None and run_rv != rv_col and not (extra_runs and extra_runs(cfg)):
            continue
        df = pd.read_parquet(f)
        df["run"] = f.parent.name
        df["rv_col"] = run_rv
        df["dataset"] = cfg.get("dataset")
        parts.append(df)
    if not parts:
        raise FileNotFoundError(
            f"no predictions.parquet under {out_root}"
            + (f" for dataset={dataset!r}" if dataset else "")
        )
    return pd.concat(parts, ignore_index=True)


def load_costs(out_root: str | Path, dataset: str | None = None) -> pd.DataFrame:
    """Wall-clock and device per run, for the computational-cost table.

    Worth reporting: in the sibling project the pretrained models were both the
    most accurate and ~1300x cheaper than refitting the statistical baselines,
    which reverses the usual assumption and is a result in its own right.
    """
    rows = []
    for cfg_path in sorted(Path(out_root).rglob("config.json")):
        cfg = json.loads(cfg_path.read_text())
        if dataset is not None and cfg.get("dataset") != dataset:
            continue
        t_path = cfg_path.parent / "timing.json"
        t = json.loads(t_path.read_text()) if t_path.exists() else {}
        secs = t.get("fit_predict_seconds") or t.get("predict_seconds")
        common = {
            "run": cfg_path.parent.name,
            "family": cfg.get("family"),
            "variant": cfg.get("input_transform"),
            "rv_col": cfg.get("rv_col", "rv"),
            "device": t.get("device"),
            "gpu": t.get("gpu"),
            "cpu_model": t.get("cpu_model"),
            "seed": cfg.get("seed"),
        }

        # A run may hold several models. run_classical.py fits the whole HAR
        # family and the trivial benchmarks in ONE job and records each one's
        # seconds separately; attributing the run total to a single `model`
        # field would drop five of the six from any per-model cost comparison,
        # which is exactly the comparison the paper makes.
        per_model = t.get("per_model_seconds") or {}
        if per_model:
            for name, s in per_model.items():
                rows.append({**common, "model": name, "seconds": s,
                             "minutes": round(s / 60, 2) if s else None,
                             "n_contexts": None, "ms_per_context": None})
            continue

        rows.append({
            **common,
            "model": cfg.get("model"),
            "seconds": secs,
            "minutes": round(secs / 60, 2) if secs else None,
            # Per-forecast cost, which is the number that actually compares
            # across families: a run's total depends on how many series it
            # happened to cover.
            "n_contexts": t.get("n_contexts"),
            "ms_per_context": t.get("ms_per_context"),
        })
    return pd.DataFrame(rows)
