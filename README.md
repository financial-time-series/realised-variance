# Zero-shot foundation models for realised variance

Code, data and results for the paper *Better on most days, worse on average:
zero-shot foundation models for realised variance under a release-date control*
(manuscript under review).

The study forecasts daily realised variance at horizons of 1, 5 and 22 days on
two panels: 61 US equities and exchange-traded funds, and 12 crypto pairs. It
compares nine zero-shot time-series foundation-model checkpoints (Chronos-T5-B,
Chronos-Bolt-S/B, Chronos-2, Moirai-1.1-B, Moirai-MoE-B, Moirai-2.0-S,
TimesFM-2.5 and TinyTimeMixer-r2) with HAR, GARCH, machine-learning, neural and
hybrid econometric–neural benchmarks. Each checkpoint is scored only on dates
after its own release on the Hugging Face Hub.

## Contents

| path | what |
|---|---|
| `data/processed/eq_rv.parquet` | equity panel: 61 series, 2000-01-04 to 2026-07-31. Daily Garman–Klass variance (`rv`, the main target), Parkinson (`rv_parkinson`) and Rogers–Satchell (`rv_rs`; `rv_rsf` floored at 1% of the series median), squared close-to-close return (`rv_sqret`), close and log return |
| `data/processed/cx_rv.parquet` | crypto panel: 12 Binance USDT pairs, 2018-01-02 to 2026-07-31. Daily realised variance from 288 five-minute returns (`rv`), realised quarticity (`rq`), squared daily return, close, log return and bar count |
| `data/processed/eq_returns.parquet` | open-to-close (`ret_oc`) and overnight (`ret_on`) log returns for every row of the equity panel |
| `data/processed/vix.parquet` | VIX close (FRED series VIXCLS) and the implied daily variance (VIX/100)²/252 |
| `results/tables/` | every aggregated result reported in the paper, as CSV (see below) |
| `src/finlab/` | the shared library: what is visible at each forecast origin (`panel.py`), the two universes (`universe.py`), HAR features (`features.py`), the checkpoint registry with pinned revisions and release dates (`zoo.py`), run directories (`io.py`), losses and tests (`evaluation.py`) |
| `scripts/` | one command-line program per stage |
| `pyproject.toml`, `uv.lock` | core dependencies, locked |
| `requirements-frozen.txt` | exact versions of every package in the environment that produced the results |

## Environment

Python 3.11 and [uv](https://docs.astral.sh/uv/). The model stacks are kept out
of `uv.lock`, because each one brings its own PyTorch requirement. They are
installed in this order, resolved as of the evening the original environment was
built, and the PyTorch build is forced last:

```bash
uv sync --frozen
for pkg in "neuralforecast>=3.0" "chronos-forecasting>=2.0" "timesfm>=2.0" \
           "uni2ts>=1.2" "granite-tsfm>=0.2"; do
  uv pip install --exclude-newer 2026-08-07T21:00:00Z "$pkg"
done
uv pip install "torch==2.8.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

`requirements-frozen.txt` lists the versions this produces.

The nine checkpoints are loaded at the Hub revisions pinned in
`src/finlab/zoo.py`. To download them once:

```bash
HF_HOME=$PWD/.hf_cache python scripts/prefetch.py
```

This also re-checks, against the Hub, each repository's creation date, the date
its weight files last changed, and the revision. Runs can then set
`HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1`.

## Reproducing the results

**Data.** This step is optional, since the processed panels are included. Both
panels were downloaded on 7 August 2026; a fresh download may differ where
providers have revised their history.

```bash
python scripts/fetch_eq.py && python scripts/build_panel.py --dataset eq   # Yahoo Finance, via yfinance
python scripts/fetch_cx.py && python scripts/build_panel.py --dataset cx   # Binance public archive
python scripts/build_extra.py      # VIX from FRED, open-to-close returns, floored Rogers-Satchell
```

**Forecasts.** Run one job per model and panel. Each writes
`predictions.parquet`, `config.json` and `timing.json` into its run directory
under `out/runs/`:

```bash
python scripts/run_classical.py --dataset eq --models rw mean ewma har har-log har-sqrt har-log-roll har-log-iv --run-dir out/runs/eq-classical
python scripts/run_classical.py --dataset eq --models garch --run-dir out/runs/eq-garch          # likewise egarch, garch-oc
python scripts/run_ml.py     --dataset eq --model lightgbm --run-dir out/runs/eq-lightgbm        # likewise ridge, randomforest
python scripts/run_deep.py   --dataset eq --model nhits --seed 1 --run-dir out/runs/eq-nhits-s1   # dlinear, patchtst; seeds 1-3
python scripts/run_hybrid.py --dataset eq --seed 1 --run-dir out/runs/eq-lstm-hybrid-s1          # seeds 1-3
python scripts/run_tsfm.py   --dataset eq --model chronos-bolt-base --input-transform log --run-dir out/runs/eq-chronos-bolt-base-log
```

- **Crypto:** use `--dataset cx`. HAR-log-IV and the open-to-close GARCH are
  equity-only.
- **Zero-shot runs:** run every checkpoint in `src/finlab/zoo.py` with
  `--input-transform log` and `--input-transform raw`.
- **Robustness:** re-run every model on the equity panel with
  `--rv-col rv_parkinson` or `--rv-col rv_rsf`.
- **Long zero-shot runs:** these can be split. Run the same command with
  `--shard k/n` for k = 0, …, n−1 (for example one GPU each), then once more
  without `--shard` to assemble the parts. The reported cost is then the sum of
  the per-series times, as for an unsplit run.

**Aggregation and diagnostics.** These write `results/tables/*.csv`:

```bash
python scripts/aggregate.py          --dataset eq
python scripts/contamination_test.py --dataset eq
python scripts/tail_diagnostics.py   --dataset eq
python scripts/daily_medians.py      --dataset eq
python scripts/direction_stats.py    --dataset eq
python scripts/spa_drop_one.py       # crypto common-window SPA without one erratic candidate
```

- **Crypto:** add `--min-coverage 0.80` to every script that takes it. The deep
  networks take their rolling windows from the end of the shortest series, and
  one pair (EOS) stops in May 2025. Every key this drops lies in 2020–2021,
  outside every leakage-controlled window.
- **Robustness measures:** run `aggregate.py` and `contamination_test.py` with
  `--rv-col rv_parkinson` or `--rv-col rv_rsf`.

Finally, `python scripts/make_tables.py` turns the CSV files into the paper's
LaTeX tables and figures, under `redaction/`.

**Hardware.** Every zero-shot, neural and hybrid run used one NVIDIA
A100-SXM4-40GB GPU with 8 CPU cores. Every other run used AMD EPYC 7763 cores: 4
for the econometric models, 16 for the machine-learning models. The original
runs were launched with batch scripts specific to the computing centre. Those
scripts are not included, although comments in the code still refer to them
(`slurm/run.slurm`, `setup_remote.sh`); the commands above are what they ran.

## Results tables

Files end in `_eq` or `_cx` for the panel. Files ending in `_eq_rv_parkinson`
or `_eq_rv_rsf` hold the equity results under the alternative variance measures.

| files | content |
|---|---|
| `losses_*` | mean, median and 10% trimmed mean of QLIKE, MSE and the directional loss (MADL), per model, window and horizon |
| `dm_*` | Diebold–Mariano (HLN) tests against HAR-log, on per-date mean, median and trimmed-mean losses |
| `medtest_*` | block-bootstrap tests of each model's pooled median loss against HAR-log's |
| `mcs_*` | model confidence set p-values and membership (10%) |
| `spa_*` | Hansen's SPA p-values against HAR-log, for all candidates (`full`) and for those of the paper's first submission (`original`) |
| `contamtest_*` | each checkpoint's median-loss ratio to HAR-log before and after its own release, with placebo release dates |
| `contampool_*` | pooled tests of the before/after shift across checkpoints, against the expanding and the rolling HAR-log |
| `calfactors_*`, `rtcal_*` | calibration constants, estimated before 2025 or in real time after release |
| `windows_*`, `common_*` | evaluation windows (first and last origin, counts), and the common-sample restriction |
| `daily_*`, `direction_*`, `tails_*` | per-date medians, direction calls, near-zero and exploding forecasts |
| `byasset_*`, `bygroup_*` | QLIKE by asset and by asset group |
| `cost_*`, `seeds_*` | compute time per model, with the hardware used, and the seed variation of the trained networks |
| `filters_*`, `data_facts.csv` | sample construction, step by step, and facts about the data quoted in the paper |
| `double_oos.csv` | double out-of-sample selection: chosen on the first post-release block, scored on the second |
| `spa_drop_one_cx.csv` | the crypto common-window SPA test with and without Moirai-MoE-B on raw input |

The individual forecasts (about 1.5 GB) are not in the repository; the
forecast commands above regenerate them.

## Data sources

- **Equity:** daily OHLCV from Yahoo Finance, via `yfinance`.
- **Crypto:** five-minute klines from the Binance public data archive
  (`data.binance.vision`).
- **Implied volatility:** the VIX close, FRED series VIXCLS.

The processed panels are provided to make the results reproducible. The
underlying data remain subject to their providers' terms of use.

## License

MIT; see [LICENSE](LICENSE). The licence covers this repository's code and
derived files. It does not change the terms under which the underlying market
data were obtained (see above).
