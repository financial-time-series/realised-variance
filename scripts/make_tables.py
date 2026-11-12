"""Generate every table and figure in the paper from results/tables/*.csv.

    python scripts/make_tables.py

Writes redaction/tables/*.tex and redaction/figures/*.tex, which the manuscript
`\\input`s. NO NUMBER IN THE PAPER IS TYPED BY HAND: re-running a campaign and
re-running this script updates the manuscript, and `check_limits.py
--verify-numbers` then confirms that what the prose claims and what the tables
hold are the same thing.

REVISION MARKS. Captions of tables that are new or changed in the revision, and
the names of models added in it, are wrapped in \\rev{...}, which the manuscript
defines to print blue in the marked copy and plain in the clean one.

Figures are standalone `tikzpicture` files with inline coordinates, so the
manuscript compiles from the .tex tree alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES_IN = Path("results/tables")
TABLES_OUT = Path("redaction/tables")
FIGS_OUT = Path("redaction/figures")
TABLES_OUT.mkdir(parents=True, exist_ok=True)
FIGS_OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAMILY_ORDER = ["trivial", "har", "garch", "ml", "deep", "hybrid", "tsfm24", "tsfm25"]
FAMILY_LABEL = {
    "trivial": "Trivial benchmarks",
    "har": "HAR family",
    "garch": "GARCH family",
    "ml": "Machine learning",
    "deep": "Deep, trained here",
    "hybrid": "Hybrid econometric--neural, trained here",
    "tsfm24": "Zero-shot foundation (2024)",
    "tsfm25": "Zero-shot foundation (2025)",
}
MODEL_FAMILY = {
    "rw": "trivial", "mean": "trivial", "ewma": "trivial",
    "har": "har", "har-log": "har", "har-sqrt": "har",
    "har-log-roll": "har", "har-log-iv": "har",
    "garch": "garch", "egarch": "garch", "garch-oc": "garch",
    "ridge": "ml", "lightgbm": "ml", "randomforest": "ml",
    "dlinear": "deep", "patchtst": "deep", "nhits": "deep",
    "lstm-hybrid": "hybrid",
}
G2024 = ["chronos-bolt-small", "chronos-bolt-base", "chronos-t5-base",
         "moirai-1.1-base", "moirai-moe-base", "ttm-r2"]
G2025 = ["chronos-2", "timesfm-2.5", "moirai-2.0"]
REVISION_MODELS = {"har-log-roll", "har-log-iv", "garch-oc", "lstm-hybrid"}
PANEL_NAME = {"eq": "equity panel", "cx": "crypto panel"}
PANEL_CAP = {"eq": "Equity panel", "cx": "Crypto panel"}

PRETTY = {
    "rw": "Random walk", "mean": "Historical mean", "ewma": "EWMA (RiskMetrics)",
    "har": "HAR-RV", "har-log": "HAR-log", "har-sqrt": "HAR-sqrt",
    "har-log-roll": "HAR-log, rolling", "har-log-iv": "HAR-log-IV",
    "garch": "GARCH(1,1)", "egarch": "EGARCH(1,1)", "garch-oc": "GARCH(1,1), open--close",
    "ridge": "Ridge", "lightgbm": "LightGBM", "randomforest": "Random forest",
    "dlinear": "DLinear", "patchtst": "PatchTST", "nhits": "NHITS",
    "lstm-hybrid": "LSTM-GARCH hybrid",
    "chronos-bolt-small": "Chronos-Bolt-S", "chronos-bolt-base": "Chronos-Bolt-B",
    "chronos-t5-base": "Chronos-T5-B", "moirai-1.1-base": "Moirai-1.1-B",
    "moirai-moe-base": "Moirai-MoE-B", "ttm-r2": "TTM-r2",
    "chronos-2": "Chronos-2", "timesfm-2.5": "TimesFM-2.5",
    "moirai-2.0": "Moirai-2.0",
}
SIG = r"($^{*}$ 10\%, $^{**}$ 5\%, $^{***}$ 1\%)"


def rev(s: str) -> str:
    return r"\rev{" + s + "}"


def base_name(label: str) -> str:
    return label.split("[")[0].replace("+rtcal", "").replace("+cal", "")


def family_of(label: str) -> str:
    b = base_name(label)
    if b in MODEL_FAMILY:
        return MODEL_FAMILY[b]
    if b in G2024:
        return "tsfm24"
    if b in G2025:
        return "tsfm25"
    return "other"


def pretty(label: str) -> str:
    b = base_name(label)
    name = PRETTY.get(b, b.replace("_", "-"))
    if "[log]" in label:
        name += " (log)"
    elif "[raw]" in label:
        name += " (raw)"
    if label.endswith("+rtcal"):
        name += ", real-time calib."
    elif label.endswith("+cal"):
        name += ", calib."
    return name


def esc(s: str) -> str:
    """LaTeX-escape a label. `<` and `>` silently render as inverted
    punctuation in OT1 text mode; they are mathematised here."""
    out = str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    return out.replace("<", r"$<$").replace(">", r"$>$")


def name_cell(label: str) -> str:
    """A model's name; blue in the marked copy if the model is new."""
    t = esc(pretty(label))
    return rev(t) if base_name(label) in REVISION_MODELS else t


def num(v: float, digits: int) -> str:
    """A table number; scientific notation once it is absurdly large, as the
    mean MSE of a model with one exploded forecast is (~1e23 and beyond)."""
    if np.isfinite(v) and abs(v) >= 1e5:
        e = int(np.floor(np.log10(abs(v))))
        return rf"${v / 10 ** e:.1f}\!\times\!10^{{{e}}}$"
    return f"{v:.{digits}f}"


def stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return r"$^{***}$" if p < 0.01 else r"$^{**}$" if p < 0.05 else \
           r"$^{*}$" if p < 0.10 else ""


def order_key(label: str) -> tuple:
    fam = family_of(label)
    return (FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99, pretty(label))


def read(name: str) -> pd.DataFrame:
    path = TABLES_IN / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fmt_date(s) -> str:
    return str(s)[:10]


def write(name: str, lines: list[str]) -> None:
    (TABLES_OUT / f"{name}.tex").write_text(
        "% GENERATED by scripts/make_tables.py -- do not edit by hand.\n"
        + "\n".join(lines) + "\n")
    print(f"  wrote {TABLES_OUT / (name + '.tex')}")


def window_dates(tag: str, window: str) -> tuple[str, str, int]:
    w = read(f"windows_{tag}.csv")
    if w.empty or window not in set(w.window):
        return "", "", 0
    r = w[w.window == window].iloc[0]
    return fmt_date(r.first_origin), fmt_date(r.last_origin), int(r.n_origins)


# --- results tables ----------------------------------------------------------

def table_main(ds: str, window: str, out_name: str, caption: str,
               families: list[str] | None = None, loss: str = "qlike",
               include_calibrated: bool = False, rv_tag: str | None = None,
               revised: bool = True) -> None:
    """Loss by model x horizon -- MEAN AND MEDIAN, side by side, each tested.

    Mean block: Diebold-Mariano stars against HAR-log on per-origin MEAN
    losses, dagger = 10% MCS on per-origin mean losses. Median block: stars
    from the block-bootstrap test of the difference in pooled medians,
    double dagger = 10% MCS on per-origin MEDIAN losses.
    """
    tag = rv_tag or ds
    losses, dm, mcs, med = (read(f"losses_{tag}.csv"), read(f"dm_{tag}.csv"),
                            read(f"mcs_{tag}.csv"), read(f"medtest_{tag}.csv"))
    if losses.empty:
        return
    sub = losses[(losses.window == window) & (losses.loss == loss)]
    if families:
        sub = sub[sub.model.map(family_of).isin(families)]
    if not include_calibrated:
        sub = sub[~sub.model.str.contains(r"\+")]
    if sub.empty:
        return
    horizons = sorted(sub.h.unique())
    models = sorted(sub.model.unique(), key=order_key)
    sel = lambda d, agg: d[(d.window == window) & (d.loss == loss)
                           & (d.aggregator == agg)].set_index(["model", "h"]) \
        if not d.empty else pd.DataFrame()
    dmi, mcs_mean, mcs_med = sel(dm, "mean"), sel(mcs, "mean"), sel(mcs, "median")
    medi = (med[(med.window == window) & (med.loss == loss)].set_index(["model", "h"])
            if not med.empty else pd.DataFrame())
    mean_p = sub.pivot_table(index="model", columns="h", values="value")
    med_p = sub.pivot_table(index="model", columns="h", values="median")
    lname = {"qlike": "QLIKE", "mse": "MSE"}[loss]
    scale = 1.0
    if loss == "mse":
        # MSE of daily variance is of order 1e-8: report it scaled.
        scale = 10 ** -np.floor(np.log10(float(np.nanmedian(mean_p.to_numpy()))))
    ncol = 1 + 2 * len(horizons)
    cap = caption if not revised else rev(caption)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + cap + r"}", r"\label{tab:" + out_name + r"}",
             r"\begin{tabular}{l" + "r" * (2 * len(horizons)) + r"}", r"\toprule",
             r" & \multicolumn{" + str(len(horizons)) + r"}{c}{Mean " + lname + r"}"
             r" & \multicolumn{" + str(len(horizons)) + r"}{c}{Median " + lname + r"} \\",
             rf"\cmidrule(lr){{2-{1 + len(horizons)}}}"
             rf"\cmidrule(lr){{{2 + len(horizons)}-{1 + 2 * len(horizons)}}}",
             r"Model & " + " & ".join(fr"$h={h}$" for h in horizons) + " & "
             + " & ".join(fr"$h={h}$" for h in horizons) + r" \\", r"\midrule"]
    best_mean = {h: mean_p[h].min() for h in horizons}
    best_med = {h: med_p[h].min() for h in horizons}
    last = None
    digits = 4 if loss == "qlike" else 3
    for m in models:
        fam = family_of(m)
        if fam != last:
            lines += [r"\addlinespace", r"\multicolumn{" + str(ncol) + r"}{l}{\itshape "
                      + FAMILY_LABEL.get(fam, fam) + r"} \\"]
            last = fam
        cells = []
        for h in horizons:
            v = mean_p.loc[m, h] if h in mean_p.columns else np.nan
            if pd.isna(v):
                cells.append("n/a"); continue
            txt = num(v * scale, digits)
            if np.isclose(v, best_mean[h]):
                txt = r"\textbf{" + txt + "}"
            if (m, h) in dmi.index:
                txt += stars(float(dmi.loc[(m, h), "p_value"]))
            if not mcs_mean.empty and (m, h) in mcs_mean.index and bool(mcs_mean.loc[(m, h), "in_mcs"]):
                txt += r"$^{\dagger}$"
            cells.append(txt)
        for h in horizons:
            v = med_p.loc[m, h] if h in med_p.columns else np.nan
            if pd.isna(v):
                cells.append("n/a"); continue
            txt = num(v * scale, digits)
            if np.isclose(v, best_med[h]):
                txt = r"\textbf{" + txt + "}"
            if not medi.empty and (m, h) in medi.index:
                txt += stars(float(medi.loc[(m, h), "p_value"]))
            if not mcs_med.empty and (m, h) in mcs_med.index and bool(mcs_med.loc[(m, h), "in_mcs"]):
                txt += r"$^{\ddagger}$"
            cells.append(txt)
        lines.append(name_cell(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if scale != 1.0:
        lines.append(r"\par\smallskip{\scriptsize MSE values multiplied by $10^{"
                     + f"{int(round(np.log10(scale)))}" + r"}$.}")
    lines.append(r"\end{table*}")
    write(out_name, lines)


def main_caption(ds: str, window: str, loss: str = "QLIKE", extra: str = "") -> str:
    a, b, n = window_dates(ds, window)
    what = {"clean": "leakage-controlled window",
            "frontier": "common leakage-controlled window",
            "stress": "most volatile fifth of leakage-controlled origins"}[window]
    who = {"clean": "after every 2024 checkpoint was published; the 2025 checkpoints are excluded",
           "frontier": "after every checkpoint in the study was published",
           "stress": "classified by each asset's own trailing 22-day variance as visible at the origin"}[window]
    return (f"{PANEL_CAP[ds]}: out-of-sample {loss} on the {what} (origins {a} to {b}, "
            f"{n} dates; {who}). {extra}Bold marks the best per horizon. Mean columns: stars are "
            r"Diebold--Mariano tests against HAR-log on per-origin mean losses " + SIG +
            r", $\dagger$ marks the 10\% model confidence set on per-origin mean losses. "
            r"Median columns: stars are block-bootstrap tests of the difference from HAR-log's "
            r"median, $\ddagger$ marks the 10\% model confidence set on per-origin median losses.")


def table_checkpoints() -> None:
    from finlab.zoo import GENERATION_WINDOWS, ZOO
    paths = {"chronos-t5-base": "20 samples", "moirai-1.1-base": "20 samples",
             "moirai-moe-base": "20 samples", "ttm-r2": "point"}
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{" + rev(
                 r"The nine zero-shot checkpoints. \emph{Created}: the repository's creation date "
                 r"on the Hugging Face Hub; \emph{Files}: the date the weight and configuration "
                 r"files at the pinned revision last changed. A checkpoint is evaluated from the "
                 r"later of the two. \emph{Context}: the input length used, in observations. "
                 r"\emph{Point}: how the point forecast is taken (the median of 20 sample paths, "
                 r"the 0.5 quantile of a quantile head, or a point head). Revisions and dates are "
                 r"re-read from the Hub API on every run.") + r"}",
             r"\label{tab:checkpoints}",
             r"\begin{tabular}{@{}l>{\ttfamily\footnotesize}l>{\ttfamily\footnotesize}lllrl@{}}",
             r"\toprule",
             r"Checkpoint & \normalfont\small Repository & \normalfont\small Revision & Created & "
             r"Files & Context & Point \\", r"\midrule"]
    for gen in ("g2024", "g2025"):
        members = sorted((s for s in ZOO.values() if s.generation == gen),
                         key=lambda s: s.effective_release)
        lines += [r"\addlinespace", r"\multicolumn{7}{@{}l}{\itshape "
                  + ("2024 generation" if gen == "g2024" else "2025 generation")
                  + r", compared jointly from " + GENERATION_WINDOWS[gen] + r"} \\"]
        for s in members:
            ctx = 512 if s.family == "ttm" else s.max_context
            lines.append(f"{esc(pretty(s.name))} & {esc(s.repo)} & {s.revision[:10]} & "
                         f"{s.released} & {s.files_changed} & {ctx:,} & "
                         f"{paths.get(s.name, '0.5 quantile')}".replace(",", "{,}", 1)
                         + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_checkpoints", lines)


def table_ckpt_contrast(ds: str, loss: str = "qlike", bench: str = "har-log") -> None:
    """Per-checkpoint before/after, wide: one row per checkpoint-transform."""
    t = read(f"contamtest_{ds}.csv")
    if t.empty:
        return
    t = t[(t.loss == loss) & (t.benchmark == bench)]
    hs = sorted(t.h.unique())
    rows = sorted(t.model.unique(), key=order_key)
    first = t.pre_start.min()
    last = t.post_end.max()
    cap = (f"{PANEL_CAP[ds]}: median {loss.upper()} relative to HAR-log before and after each "
           r"checkpoint's own release date (Table~\ref{tab:checkpoints}). Pre-release window: "
           f"origins from {first} to the day before the release; post-release window: from the "
           f"release to {last}. Values below 1 favour the checkpoint; $\\Delta=$ post $-$ pre, "
           r"so $\Delta>0$ means it looked better before it was published. Stars: moving-block "
           r"bootstrap of $\Delta$ (blocks of 22 whole origins, 2{,}000 draws) " + SIG + ".")
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:ckpt_" + ds + r"}",
             r"\begin{tabular}{@{}lcrr" + "rrr" * len(hs) + r"@{}}", r"\toprule",
             r" & Evaluated & \multicolumn{2}{c}{Origins} & "
             + " & ".join(r"\multicolumn{3}{c}{$h=" + str(h) + "$}" for h in hs) + r" \\",
             r"\cmidrule(lr){3-4}" + "".join(
                 rf"\cmidrule(lr){{{5 + 3 * i}-{7 + 3 * i}}}" for i in range(len(hs))),
             r"Checkpoint & from & pre & post & " + " & ".join(
                 ["Pre & Post & $\\Delta$"] * len(hs)) + r" \\", r"\midrule"]
    last_fam = None
    for m in rows:
        fam = family_of(m)
        if fam != last_fam:
            lines += [r"\addlinespace", r"\multicolumn{" + str(4 + 3 * len(hs))
                      + r"}{@{}l}{\itshape " + FAMILY_LABEL[fam] + r"} \\"]
            last_fam = fam
        r0 = t[t.model == m].iloc[0]
        cells = [esc(pretty(m)), fmt_date(r0.release), f"{int(r0.n_pre):,}".replace(",", "{,}"),
                 f"{int(r0.n_post):,}".replace(",", "{,}")]
        for h in hs:
            r = t[(t.model == m) & (t.h == h)]
            if r.empty:
                cells += ["n/a"] * 3; continue
            r = r.iloc[0]
            cells += [f"{r.pre_ratio:.3f}", f"{r.post_ratio:.3f}",
                      f"{r.delta:+.3f}{stars(r.p_value)}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(f"tab_ckpt_{ds}", lines)


def table_contam_summary() -> None:
    """Pooled before/after tests for the twelve 2024 variants, both panels."""
    parts = {ds: (read(f"contampool_{ds}.csv"), read(f"contamtest_{ds}.csv")) for ds in ("eq", "cx")}
    if all(p[0].empty for p in parts.values()):
        return
    cap = (r"The before/after shift across the twelve 2024-generation checkpoint--transform "
           r"combinations, each split at its own release date. \emph{Positive}: variants with "
           r"$\Delta>0$; \emph{Median} and \emph{Mean}: of $\Delta$ across the twelve; "
           r"\emph{excl.}: the mean without the variant with the largest $|\Delta|$; \emph{Pooled $p$}: "
           r"bootstrap $p$ for the median $\Delta$, every variant recomputed on the same draw of "
           r"22-origin blocks so that the dependence between checkpoints is preserved; "
           r"\emph{Placebo}: the median $\Delta$ when each release date is moved back by the length "
           r"of its post-release window, so that both windows precede the release, and its pooled "
           r"$p$; \emph{Rolling}: the median $\Delta$ against HAR-log fitted on a rolling window of "
           r"1{,}000 observations instead of an expanding one.")
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:contam}",
             r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{4}{c}{$\Delta$ against HAR-log} & \multicolumn{2}{c}{Placebo} "
             r"& \multicolumn{2}{c}{Rolling} \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Loss & $h$ & Positive & Median & Mean & excl. & Pooled $p$ & Median & $p$ & Median & $p$ \\",
             r"\midrule"]
    for ds in ("eq", "cx"):
        pool, _ = parts[ds]
        if pool.empty:
            continue
        lines += [r"\addlinespace", r"\multicolumn{11}{@{}l}{\itshape " + PANEL_CAP[ds] + r"} \\"]
        for loss in ("qlike", "mse"):
            for i, h in enumerate((1, 5, 22)):
                g = lambda kind, bench: pool[(pool.loss == loss) & (pool.h == h) & (pool.group == "g2024")
                                             & (pool.kind == kind) & (pool.benchmark == bench)]
                r, pl, ro = g("release", "har-log"), g("placebo", "har-log"), g("release", "har-log-roll")
                if r.empty:
                    continue
                r = r.iloc[0]
                tag = (r"\textsc{" + loss + "}") if i == 0 else ""
                pc = (f"{pl.iloc[0].median_delta:+.3f} & {pl.iloc[0].p_median:.3f}" if not pl.empty
                      else "n/a & n/a")
                rc = (f"{ro.iloc[0].median_delta:+.3f} & {ro.iloc[0].p_median:.3f}" if not ro.empty
                      else "n/a & n/a")
                lines.append(f"{tag} & {h} & {int(r.n_positive)}/{int(r.n_variants)} & "
                             f"{r.median_delta:+.3f} & {r.mean_delta:+.3f} & "
                             f"{r.mean_delta_excl_largest:+.3f} & {r.p_median:.3f} & {pc} & {rc}"
                             + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_contam", lines)


def table_calibration(ds: str, h: int = 1, window: str = "clean") -> None:
    """Per-cell effect of the constant: gain, share of the gap closed, and the
    same estimator updated in real time from post-release data only."""
    q = read(f"losses_{ds}.csv")
    if q.empty:
        return
    q = q[(q.window == window) & (q.loss == "qlike") & (q.h == h)].set_index("model")["value"]
    bench = q.get("har-log", np.nan)
    rows = []
    for m in q.index:
        if family_of(m) not in ("tsfm24", "tsfm25") or "+" in m:
            continue
        u, c, rt = q.get(m), q.get(m + "+cal"), q.get(m + "+rtcal")
        if c is None:
            continue
        gap = (u - c) / (u - bench) if u > bench else np.nan
        rows.append((m, u, c, 1 - c / u, gap, rt, (1 - rt / u) if rt is not None else np.nan))
    if not rows:
        return
    rows.sort(key=lambda r: -r[3])
    cal = read(f"calfactors_{ds}.csv")
    a, b, n = window_dates(ds, window)
    cap = (f"{PANEL_CAP[ds]}: one multiplicative constant per series and horizon, the geometric "
           r"mean of $y/\hat y$, applied to each zero-shot checkpoint. Mean QLIKE at $h=" + str(h)
           + f"$ on the leakage-controlled window ({a} to {b}). \\emph{{Calibrated}}: constant "
           r"estimated once on every origin before 2025-01-01 (" +
           ("from 2010" if ds == "eq" else "from 2020") +
           r", at least 252 per series). \emph{Gap closed}: the share of the distance to HAR-log ("
           + f"{bench:.4f}" + r") removed by the constant where the uncalibrated model is behind it (negative: the constant widened it). "
           r"\emph{Real-time}: the same estimator updated at each origin from post-release origins "
           r"whose targets had closed, so that no observation that could lie in a pretraining corpus "
           r"enters it.")
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:calib_" + ds + r"}",
             r"\setlength{\tabcolsep}{2.2pt}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Calibrated} & \multicolumn{2}{c}{Real-time} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"Checkpoint & Uncal. & QLIKE & Gain & Gap & QLIKE & Gain \\", r"\midrule"]
    for m, u, c, gain, gap, rt, rgain in rows:
        gap_t = f"{gap:.1%}" if np.isfinite(gap) else "ahead"
        lines.append(f"{esc(pretty(m))} & {u:.4f} & {c:.4f} & {gain:+.1%} & {gap_t} & "
                     f"{rt:.4f} & {rgain:+.1%}".replace("%", r"\%") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(f"tab_calib_{ds}", lines)


def table_spa_summary() -> None:
    """SPA against HAR-log: p-values for per-origin mean, median and trimmed losses."""
    cap = (r"Hansen's test for superior predictive ability against HAR-log: $p$-values. The "
           r"loss of each model on each date is summarised across assets by the mean (the "
           r"conventional test), the median, or the 10\% trimmed mean. Candidates: every model "
           r"and calibrated variant admissible on the window (the count is in parentheses). "
           r"A small $p$ means that at least one candidate beats HAR-log after accounting for "
           r"the number tried.")
    out = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
           r"\caption{" + rev(cap) + r"}", r"\label{tab:spa}",
           r"\begin{tabular}{@{}lllrrrrrrrrr@{}}", r"\toprule",
           r" & & & \multicolumn{3}{c}{$h=1$} & \multicolumn{3}{c}{$h=5$} & \multicolumn{3}{c}{$h=22$} \\",
           r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}"]
    rows = []
    for ds in ("eq", "cx"):
        s = read(f"spa_{ds}.csv")
        if s.empty:
            continue
        for loss in ("qlike", "mse"):
            for w in ("clean", "frontier", "stress"):
                g = s[(s.loss == loss) & (s.window == w) & (s.set == "full")]
                if g.empty:
                    continue
                cells = []
                for h in (1, 5, 22):
                    for agg in ("mean", "median", "trim"):
                        r = g[(g.h == h) & (g.aggregator == agg)]
                        cells.append(f"{r.iloc[0].pvalue_consistent:.3f}" if not r.empty else "n/a")
                n = int(g.iloc[0].n_alternatives) + 1
                rows.append((ds, loss, w, n, cells))
    if not rows:
        return
    out += [r"Panel & Loss & Window (candidates) & Mean & Median & Trim & Mean & Median & Trim & "
            r"Mean & Median & Trim \\", r"\midrule"]
    last = None
    for ds, loss, w, n, cells in rows:
        lead = PANEL_CAP[ds].split()[0] if (ds, loss) != last else ""
        lt = (r"\textsc{" + loss + "}") if (ds, loss) != last else ""
        if (ds, loss) != last and last is not None:
            out.append(r"\addlinespace")
        out.append(f"{lead} & {lt} & {w} ({n}) & " + " & ".join(cells) + r" \\")
        last = (ds, loss)
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_spa", out)


def table_double_oos(loss: str = "qlike") -> None:
    """Choose each family's specification on the first post-release block,
    score it once on the second (Mroziewicz and Slepaczuk 2026)."""
    rows = []
    for ds in ("eq", "cx"):
        L, M = read(f"losses_{ds}.csv"), read(f"medtest_{ds}.csv")
        D = read(f"dm_{ds}.csv")
        if L.empty:
            continue
        L = L[(L.loss == loss) & ~L.model.str.contains(r"\+")]
        for fam in ("tsfm24", "deep", "ml", "har", "garch", "trivial"):
            for crit in ("median", "value"):
                for h in (1, 5, 22):
                    a = L[(L.window == "blockA") & (L.h == h) & (L.model.map(family_of) == fam)]
                    if fam == "har":
                        a = a[a.model != "har-log"]
                    if a.empty:
                        continue
                    pick = a.loc[a[crit].idxmin(), "model"]
                    b = L[(L.window == "frontier") & (L.h == h)].set_index("model")
                    if pick not in b.index or "har-log" not in b.index:
                        continue
                    mr = b.loc[pick, "median"] / b.loc["har-log", "median"]
                    vr = b.loc[pick, "value"] / b.loc["har-log", "value"]
                    pm = M[(M.window == "frontier") & (M.loss == loss) & (M.h == h) & (M.model == pick)]
                    pd_ = D[(D.window == "frontier") & (D.loss == loss) & (D.h == h)
                            & (D.model == pick) & (D.aggregator == "mean")]
                    rows.append({"ds": ds, "fam": fam, "crit": crit, "h": h, "pick": pick,
                                 "med_ratio": mr, "mean_ratio": vr,
                                 "p_med": float(pm.p_value.iloc[0]) if not pm.empty else np.nan,
                                 "p_mean": float(pd_.p_value.iloc[0]) if not pd_.empty else np.nan})
    if not rows:
        return
    d = pd.DataFrame(rows)
    d.to_csv(TABLES_IN / "double_oos.csv", index=False)
    have = [x for x in ("eq", "cx") if window_dates(x, "blockA")[0]]
    a0, a1, _ = window_dates(have[0], "blockA")
    f0, f1, _ = window_dates(have[0], "frontier")
    cap = (r"Double out-of-sample evaluation. Within each family the specification with the lowest "
           r"median QLIKE on the first post-release block (" + f"{a0} to {a1}" + r") is chosen, then "
           r"scored once on the second (" + f"{f0} to {f1}" + r"), which played no part in the "
           r"choice. Entries are the chosen model's median and mean QLIKE on the second block "
           r"relative to HAR-log's; stars test the median against HAR-log's (block bootstrap) and "
           r"the mean (Diebold--Mariano) " + SIG + r". For the HAR family, HAR-log itself is excluded "
           r"from the choice.")
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:doos}",
             r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}llrrlrrlrr@{}}",
             r"\toprule",
             r" & \multicolumn{3}{c}{$h=1$} & \multicolumn{3}{c}{$h=5$} & \multicolumn{3}{c}{$h=22$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Family & Chosen & Median & Mean & Chosen & Median & Mean & Chosen & Median & Mean \\",
             r"\midrule"]
    for ds in ("eq", "cx"):
        s = d[(d.ds == ds) & (d.crit == "median")]
        if s.empty:
            continue
        lines += [r"\addlinespace", r"\multicolumn{10}{@{}l}{\itshape " + PANEL_CAP[ds] + r"} \\"]
        for fam in ("tsfm24", "deep", "ml", "har", "garch", "trivial"):
            f = s[s.fam == fam]
            if f.empty:
                continue
            cells = [{"tsfm24": "Zero-shot (2024)", "deep": "Deep", "ml": "Machine learning",
                      "har": "HAR", "garch": "GARCH", "trivial": "Trivial"}[fam]]
            for h in (1, 5, 22):
                r = f[f.h == h]
                if r.empty:
                    cells += ["n/a"] * 3; continue
                r = r.iloc[0]
                cells += [esc(pretty(r.pick)),
                          f"{r.med_ratio:.3f}{stars(r.p_med)}", f"{r.mean_ratio:.3f}{stars(r.p_mean)}"]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_doos", lines)


def table_blocks(loss: str = "qlike") -> None:
    """Three consecutive half-year blocks of the post-release period."""
    blocks = [("b2025h1", "2025 H1"), ("b2025h2", "2025 H2"), ("b2026", "2026")]
    out_rows = []
    for ds in ("eq", "cx"):
        L, M = read(f"losses_{ds}.csv"), read(f"medtest_{ds}.csv")
        if L.empty:
            continue
        L = L[(L.loss == loss) & ~L.model.str.contains(r"\+")]
        for b, bl in blocks:
            a0, a1, n = window_dates(ds, b)
            cells = [PANEL_CAP[ds].split()[0], f"{bl} ({n})"]
            for h in (1, 5, 22):
                s = L[(L.window == b) & (L.h == h)].set_index("model")
                if "har-log" not in s.index:
                    cells += ["n/a"] * 3; continue
                z = s[s.index.map(family_of) == "tsfm24"]
                rel = z["median"] / s.loc["har-log", "median"]
                below = int((rel < 1).sum())
                mm = M[(M.window == b) & (M.loss == loss) & (M.h == h) & M.model.isin(z.index)]
                sig = int(((mm["diff"] < 0) & (mm.p_value < 0.05)).sum())
                relm = (z["value"] / s.loc["har-log", "value"])
                cells += [f"{below}/{len(z)} ({sig})", f"{rel.min():.3f}", f"{int((relm < 1).sum())}/{len(z)}"]
            out_rows.append(cells)
    if not out_rows:
        return
    cap = (r"The 2024-generation checkpoints in three consecutive, non-overlapping blocks of the "
           r"post-release period (dates in parentheses). \emph{Median}: variants, of twelve, whose "
           r"median QLIKE is below HAR-log's, with the number significantly below it at 5\% (block "
           r"bootstrap) in parentheses; \emph{Best}: the lowest median QLIKE relative to HAR-log; "
           r"\emph{Mean}: variants whose mean QLIKE is below HAR-log's.")
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:blocks}",
             r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{$h=1$} & \multicolumn{3}{c}{$h=5$} & \multicolumn{3}{c}{$h=22$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"Panel & Block & Median & Best & Mean & Median & Best & Mean & Median & Best & Mean \\",
             r"\midrule"]
    prev = None
    for c in out_rows:
        if prev is not None and c[0] != prev:
            lines.append(r"\addlinespace")
        lines.append(" & ".join([c[0] if c[0] != prev else ""] + c[1:]) + r" \\")
        prev = c[0]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_blocks", lines)


def table_proxies() -> None:
    """Equity results under the three range-based measures of daily variance."""
    meas = [("rv", "Garman--Klass"), ("rv_parkinson", "Parkinson"), ("rv_rsf", "Rogers--Satchell")]
    rows = []
    for col, nm in meas:
        tag = "eq" if col == "rv" else f"eq_{col}"
        L, S, P = read(f"losses_{tag}.csv"), read(f"spa_{tag}.csv"), read(f"contampool_{tag}.csv")
        if L.empty:
            continue
        L = L[(L.loss == "qlike") & (L.window == "clean") & ~L.model.str.contains(r"\+")]
        for h in (1, 5, 22):
            s = L[L.h == h].set_index("model")
            z = s[s.index.map(family_of) == "tsfm24"]
            hb = s.loc["har-log"]
            bm = z["median"].idxmin()
            sp = S[(S.window == "clean") & (S.loss == "qlike") & (S.h == h) & (S.set == "full")] if not S.empty else S
            spm = sp[sp.aggregator == "mean"].pvalue_consistent
            spd = sp[sp.aggregator == "median"].pvalue_consistent
            pc = P[(P.loss == "qlike") & (P.h == h) & (P.group == "g2024") & (P.kind == "release")
                   & (P.benchmark == "har-log")] if not P.empty else P
            rows.append([nm if h == 1 else "", str(h),
                         f"{hb['value']:.4f}", f"{hb['median']:.4f}",
                         f"{z['value'].min():.4f}", f"{z['median'].min():.4f}",
                         f"{int((z['median'] < hb['median']).sum())}/{len(z)}",
                         f"{spm.iloc[0]:.3f}" if len(spm) else "n/a",
                         f"{spd.iloc[0]:.3f}" if len(spd) else "n/a",
                         (f"{int(pc.iloc[0].n_positive)}/{int(pc.iloc[0].n_variants)}" if len(pc) else "n/a"),
                         (f"{pc.iloc[0].median_delta:+.3f}" if len(pc) else "n/a"),
                         (f"{pc.iloc[0].p_median:.3f}" if len(pc) else "n/a")])
    if not rows:
        return
    cap = (r"Equity panel under three range-based measures of daily variance, each used both as "
           r"the models' input and as the target; leakage-controlled window, QLIKE. \emph{HAR-log} "
           r"and \emph{Best zero-shot}: mean and median QLIKE (the best over the twelve "
           r"2024-generation variants, separately for each statistic); \emph{Below}: variants with a "
           r"median below HAR-log's; \emph{SPA}: $p$-values on per-origin mean and median losses; "
           r"\emph{Before/after}: the per-checkpoint contrast of Table~\ref{tab:contam} (positive "
           r"shifts, median $\Delta$, pooled $p$). GARCH forecasts, which read returns, are re-scored "
           r"against each measure.")
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:proxies}",
             r"\begin{tabular}{@{}llrrrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{HAR-log} & \multicolumn{2}{c}{Best zero-shot} & & "
             r"\multicolumn{2}{c}{SPA} & \multicolumn{3}{c}{Before/after} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}\cmidrule(lr){10-12}",
             r"Measure & $h$ & Mean & Median & Mean & Median & Below & Mean & Median & Positive & "
             r"Median $\Delta$ & $p$ \\", r"\midrule"]
    for r in rows:
        if r[0] and lines[-1] != r"\midrule":
            lines.append(r"\addlinespace")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_proxies", lines)


def table_madl() -> None:
    """Mean MADL on the leakage-controlled window, both panels."""
    parts = {ds: (read(f"losses_{ds}.csv"), read(f"dm_{ds}.csv"), read(f"spa_{ds}.csv"))
             for ds in ("eq", "cx")}
    if all(p[0].empty for p in parts.values()):
        return
    models = None
    tab = {}
    for ds, (L, D, S) in parts.items():
        if L.empty:
            continue
        L = L[(L.loss == "madl") & (L.window == "clean") & ~L.model.str.contains(r"\+")]
        tab[ds] = (L.pivot_table(index="model", columns="h", values="value"),
                   D[(D.loss == "madl") & (D.window == "clean") & (D.aggregator == "mean")]
                   .set_index(["model", "h"]),
                   S[(S.loss == "madl") & (S.window == "clean") & (S.set == "full")
                     & (S.aggregator == "mean")].set_index("h"))
        models = sorted(set(L.model) | (set(models) if models else set()), key=order_key)
    spa_txt = "; ".join(
        f"{PANEL_NAME[ds]}: " + ", ".join(f"{s.loc[h, 'pvalue_consistent']:.3f}" for h in (1, 5, 22)
                                          if h in s.index)
        for ds, (_, _, s) in tab.items())
    cap = (r"Mean absolute directional loss (MADL) on the leakage-controlled window: $-\mathrm{sign}"
           r"(R\hat R)\,|R|$ with $R=\log(y/b)$ and $\hat R=\log(\hat y/b)$, where $b$ is the "
           r"realised variance averaged over the $h$ days up to the origin. It scores whether a "
           r"forecast calls the direction of the variance move, weighted by the size of the move, "
           r"and is bounded, so near-zero forecasts cannot dominate it. Lower is better; $-$mean$|R|$ "
           r"is the best attainable. The random walk's forecast equals the reference at $h=1$, so it "
           r"calls no direction and scores zero there. Stars: Diebold--Mariano against HAR-log " + SIG
           + r". SPA $p$ "
           r"against HAR-log at $h=1,5,22$: " + spa_txt + ".")
    hs = (1, 5, 22)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:madl}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Equity panel} & \multicolumn{3}{c}{Crypto panel} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & " + " & ".join(f"$h={h}$" for h in hs) + " & "
             + " & ".join(f"$h={h}$" for h in hs) + r" \\", r"\midrule"]
    last = None
    for m in models:
        fam = family_of(m)
        if fam != last:
            lines += [r"\addlinespace", r"\multicolumn{7}{@{}l}{\itshape " + FAMILY_LABEL.get(fam, fam) + r"} \\"]
            last = fam
        cells = [name_cell(m)]
        for ds in ("eq", "cx"):
            if ds not in tab:
                cells += ["n/a"] * 3; continue
            p, dm, _ = tab[ds]
            for h in hs:
                if m not in p.index or pd.isna(p.loc[m, h]):
                    cells.append("n/a"); continue
                v = f"{p.loc[m, h]:.4f}"
                if np.isclose(p.loc[m, h], p[h].min()):
                    v = r"\textbf{" + v + "}"
                if (m, h) in dm.index:
                    v += stars(float(dm.loc[(m, h), "p_value"]))
                cells.append(v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_madl", lines)


def table_daily() -> None:
    """Pooled versus per-date medians: why the robust SPA tests do not reject."""
    rows = []
    for ds in ("eq", "cx"):
        d = read(f"daily_{ds}.csv")
        if d.empty:
            continue
        b = d[d.model == "har-log"].iloc[0]
        for _, r in d[d.model != "har-log"].sort_values("model", key=lambda c: c.map(order_key)).iterrows():
            rows.append((ds, r, b))
    if not rows:
        return
    cap = (r"Pooled and per-date medians of QLIKE, leakage-controlled window, $h=1$. \emph{Pooled}: "
           r"the median over every asset and date, as in Tables~\ref{tab:tab_main_eq} "
           r"and~\ref{tab:tab_main_cx}. \emph{Per date}: the median across assets on each date, "
           r"averaged over dates, the quantity a test on per-date median losses compares. "
           r"\emph{Below}: the share of dates on which the checkpoint's per-date median is below "
           r"HAR-log's; \emph{Top 5\%}: the share of its total shortfall against HAR-log carried "
           r"by the worst 5\% of dates; \emph{Worst date}: the origin of its largest shortfall.")
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:daily}",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrrrrl@{\hspace{1em}}lrrrrl@{}}", r"\toprule"]
    by = {ds: [(r, b) for d_, r, b in rows if d_ == ds] for ds in ("eq", "cx")}
    lines.append(r" & \multicolumn{5}{c}{Equity panel} & & \multicolumn{5}{c}{Crypto panel} \\")
    lines.append(r"\cmidrule(lr){2-6}\cmidrule(lr){8-12}")
    lines.append(r"Model & Pooled & Per date & Below & Top 5\% & Worst date & Model & Pooled & "
                 r"Per date & Below & Top 5\% & Worst date \\")
    lines.append(r"\midrule")
    def cells(rb):
        if rb is None:
            return [""] * 6
        r, b = rb
        return [esc(pretty(r.model)), f"{r.pooled_median:.4f}", f"{r.mean_daily_median:.4f}",
                f"{r.share_dates_below_bench:.0%}".replace("%", r"\%"),
                f"{r.share_top5pct_dates:.0%}".replace("%", r"\%"), r.worst_date]
    def bench(ds):
        d = read(f"daily_{ds}.csv")
        if d.empty:
            return [""] * 6
        b = d[d.model == "har-log"].iloc[0]
        return ["HAR-log", f"{b.pooled_median:.4f}", f"{b.mean_daily_median:.4f}", "", "", ""]
    lines.append(" & ".join(bench("eq") + bench("cx")) + r" \\")
    lines.append(r"\addlinespace")
    n = max(len(by["eq"]), len(by["cx"]))
    for i in range(n):
        e = by["eq"][i] if i < len(by["eq"]) else None
        c = by["cx"][i] if i < len(by["cx"]) else None
        lines.append(" & ".join(cells(e) + cells(c)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_daily", lines)


def table_direction(h: int = 1) -> None:
    """How often each model calls a rise, against how often one occurs."""
    parts = {ds: read(f"direction_{ds}.csv") for ds in ("eq", "cx")}
    if all(d.empty for d in parts.values()):
        return
    tab = {ds: d[(d.h == h) & ~d.label.str.contains(r"\+")].set_index("label")
           for ds, d in parts.items() if not d.empty}
    models = sorted(set().union(*[set(t.index) for t in tab.values()]), key=order_key)
    occ = {ds: float(t["share_rise"].iloc[0]) for ds, t in tab.items()}
    pct = lambda v: f"{v:.1%}".replace("%", r"\%")
    cap = (r"Direction of the next move in variance, leakage-controlled window, $h=" + str(h) + r"$. "
           r"\emph{Rise}: the share of origins at which the forecast exceeds the realised variance "
           r"averaged over the $h$ days up to the origin, the reference of the directional loss "
           r"(\S\ref{sec:metrics}); a rise follows on " + pct(occ.get("eq", float("nan"))) +
           r" of equity and " + pct(occ.get("cx", float("nan"))) + r" of crypto origins. "
           r"\emph{Right}: the share at which the forecast calls the direction correctly.")
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:direction}",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Equity} & \multicolumn{2}{c}{Crypto} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Model & Rise & Right & Rise & Right \\", r"\midrule"]
    last = None
    for m in models:
        fam = family_of(m)
        if fam != last:
            lines += [r"\addlinespace", r"\multicolumn{5}{@{}l}{\itshape " + FAMILY_LABEL.get(fam, fam) + r"} \\"]
            last = fam
        cells = [name_cell(m)]
        for ds in ("eq", "cx"):
            t = tab.get(ds)
            if t is None or m not in t.index:
                cells += ["n/a", "n/a"]; continue
            cells += [pct(t.loc[m, "share_forecast_rise"]), pct(t.loc[m, "share_direction_right"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_direction", lines)


def table_tickers() -> None:
    """Every series, with its first observation when it starts late."""
    starts = {ds: fmt_date(pd.read_parquet(f"data/processed/{ds}_rv.parquet")["ds"].min())
              for ds in ("eq", "cx")}
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + rev(
                 r"The two universes. Every series runs to the end of its panel (2026-07-31) "
                 r"unless stated, and starts on the panel's first date (equity " + starts["eq"]
                 + ", crypto " + starts["cx"] + r") unless its own first observation is given in "
                 r"parentheses. "
                 r"\emph{Obs.}: daily observations after the filters of "
                 r"Tables~\ref{tab:sample_eq} and~\ref{tab:sample_cx}.") + r"}",
             r"\label{tab:tickers}",
             r"\begin{tabular}{@{}lrp{0.62\linewidth}r@{}}", r"\toprule",
             r"Group & Series & Members & Obs. \\", r"\midrule"]
    names = {"index_etf": "Broad index funds", "sector_etf": "Select Sector SPDRs",
             "macro_etf": "Bond, credit and commodity funds", "stock": "Large-capitalisation stocks",
             "crypto": "Crypto pairs (USDT quote)"}
    for ds in ("eq", "cx"):
        p = pd.read_parquet(f"data/processed/{ds}_rv.parquet")
        p["ds"] = pd.to_datetime(p["ds"])
        first, last = p.ds.min(), p.ds.max()
        f = p.groupby(["group", "unique_id"]).ds.agg(["min", "max", "count"]).reset_index()
        for grp in ([g for g in names if g in set(f.group)]):
            s = f[f.group == grp].sort_values("unique_id")
            items = []
            for _, r in s.iterrows():
                t = r.unique_id.replace("USDT", "")
                notes = []
                if r["min"] > first + pd.Timedelta(days=7):
                    notes.append(fmt_date(r["min"]))
                if r["max"] < last - pd.Timedelta(days=7):
                    notes.append("to " + fmt_date(r["max"]))
                items.append(t + (f" ({', '.join(notes)})" if notes else ""))
            obs = f"{int(s['count'].sum()):,}".replace(",", "{,}")
            lines.append(f"{names[grp]} & {len(s)} & {', '.join(items)} & {obs}" + r" \\")
        if ds == "eq":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_tickers", lines)


def table_cost(ds: str) -> None:
    c = read(f"cost_{ds}.csv")
    if c.empty:
        return
    c = c[c.seconds.notna()].copy()
    c["label"] = np.where(c.variant.fillna("").eq(""), c.model.astype(str),
                          c.model.astype(str) + "[" + c.variant.fillna("") + "]")
    # Seeds: report the mean over the three runs.
    c = (c.groupby(["label", "family"], dropna=False)
          .agg(seconds=("seconds", "mean"), ms=("ms_per_context", "mean"),
               gpu=("gpu", "first"), cpu=("cpu_model", "first")).reset_index())
    c = c.sort_values("seconds")
    gpus = sorted(set(c.gpu.dropna()))
    cpus = sorted(set(c[c.gpu.isna()].cpu.dropna()))
    cap = (f"{PANEL_CAP[ds]}: wall-clock time of one full pass (every origin, every horizon), "
           r"excluding the loading of a checkpoint. GPU runs on one "
           + (", ".join(esc(g) for g in gpus) or "GPU") + r" with 8 cores; CPU runs on "
           + (", ".join(esc(x) for x in cpus) or "CPU") + r" cores, 4 for the econometric "
           r"models and 16 for the machine-learning models. Seeded models: the mean of three "
           r"runs. \emph{ms}: milliseconds per forecast origin, for the zero-shot models.")
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + rev(cap) + r"}", r"\label{tab:cost_" + ds + r"}",
             r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
             r"Model & Device & Seconds & ms \\", r"\midrule"]
    for _, r in c.iterrows():
        dev = "GPU" if isinstance(r.gpu, str) else "CPU"
        ms = f"{r.ms:.3f}" if np.isfinite(r.ms) else ""
        lines.append(f"{name_cell(r.label)} & {dev} & {r.seconds:,.1f}".replace(",", "{,}")
                     + f" & {ms}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(f"tab_cost_{ds}", lines)


def table_sample(ds: str) -> None:
    f = read(f"filters_{ds}.csv")
    if f.empty:
        return
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Sample construction, " + PANEL_NAME[ds] + r". Every filter, in the order "
             r"applied, with the series and observations surviving each.}",
             r"\label{tab:sample_" + ds + r"}",
             r"\begin{tabular}{@{}p{0.56\linewidth}rr@{}}", r"\toprule",
             r"Filter & Series & Observations \\", r"\midrule"]
    for _, r in f.iterrows():
        lines.append(f"{esc(r['step'])} & {int(r['series'])} & "
                     f"{int(r['observations']):,}".replace(",", r"\,") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(f"tab_sample_{ds}", lines)


# --- figures -------------------------------------------------------------------
# Greyscale throughout: identity is carried by position and marker shape.

def figure_windows(ds: str) -> None:
    panel = pd.read_parquet(f"data/processed/{ds}_rv.parquet")[["ds", "rv"]]
    panel["ds"] = pd.to_datetime(panel["ds"])
    m = panel.set_index("ds").groupby(pd.Grouper(freq="MS"))["rv"].median().dropna()
    if m.empty:
        return
    ann = np.sqrt(m * 252.0) * 100.0
    x = m.index.year + (m.index.month - 1) / 12.0
    coords = " ".join(f"({xi:.3f},{yi:.2f})" for xi, yi in zip(x, ann.to_numpy()))
    ymax = float(ann.max()) * 1.14
    xmin, xmax = float(x.min()), float(x.max())
    g24 = 2025.0
    span = xmax - xmin
    step = 1 if span <= 10 else (2 if span <= 16 else 4)
    xtick = ",".join(str(y) for y in range(int(np.ceil(xmin)), int(xmax) + 1, step))
    tex = rf"""% GENERATED by scripts/make_tables.py -- do not edit by hand.
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.95\linewidth, height=0.44\linewidth,
  xlabel={{}}, ylabel={{Annualised volatility (\%)}},
  xmin={xmin:.2f}, xmax={xmax:.2f}, ymin=0, ymax={ymax:.1f},
  xtick={{{xtick}}}, scaled x ticks=false,
  x tick label style={{/pgf/number format/1000 sep={{}}, font=\small}},
  y tick label style={{font=\small}},
  grid=major, grid style={{black!12}},
  axis line style={{black!45}}, tick style={{black!45}}, label style={{font=\small}},
]
\addplot[draw=none, fill=black!7, forget plot] coordinates
  {{({g24:.3f},0) ({xmax:.2f},0) ({xmax:.2f},{ymax:.1f}) ({g24:.3f},{ymax:.1f})}} \closedcycle;
\addplot[black!55, line width=0.6pt, forget plot]
  coordinates {{({g24:.3f},0) ({g24:.3f},{ymax:.1f})}};
\node[anchor=north east, font=\tiny, align=right, text=black!70]
  at (axis cs:{g24 - 0.15:.3f},{ymax * 0.99:.1f}) {{leakage-controlled window $\rightarrow$}};
\addplot[black, line width=0.9pt, mark=none] coordinates {{{coords}}};
\end{{axis}}
\end{{tikzpicture}}
"""
    (FIGS_OUT / f"fig_windows_{ds}.tex").write_text(tex)
    print(f"  wrote {FIGS_OUT}/fig_windows_{ds}.tex")


def figure_contamination(ds: str, loss: str = "qlike") -> None:
    """Dumbbells, one panel per horizon: pre-release (hollow) and post-release
    (solid) median loss relative to HAR-log, each checkpoint split at its own
    release date. Rows keep the same order in every panel."""
    t = read(f"contamtest_{ds}.csv")
    if t.empty:
        return
    t = t[(t.loss == loss) & (t.benchmark == "har-log")]
    order = (t[t.h == 1].sort_values("post_ratio", ascending=False).model.tolist())
    rows = {m: i + 1 for i, m in enumerate(order)}
    lo = min(float(t[["pre_ratio", "post_ratio"]].min().min()), 1.0) * 0.92
    hi = max(float(t[["pre_ratio", "post_ratio"]].max().max()), 1.0) * 1.08
    # Few ticks: three panels share one page width, and a dense log ladder
    # turned into an unreadable smear in the first draft of this figure.
    cand = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    xticks = ",".join(f"{v:g}" for v in cand if lo <= v <= hi) or "1"
    labels = ",".join("{" + esc(pretty(m)) + "}" for m in order)
    n = len(order)
    plots = []
    for k, h in enumerate((1, 5, 22)):
        d = t[t.h == h]
        segs = "\n".join(
            rf"\addplot[black!45, line width=0.6pt, forget plot] coordinates "
            rf"{{({r.pre_ratio:.4f},{rows[r.model]}) ({r.post_ratio:.4f},{rows[r.model]})}};"
            for r in d.itertuples())
        pre = " ".join(f"({r.pre_ratio:.4f},{rows[r.model]})" for r in d.itertuples())
        post = " ".join(f"({r.post_ratio:.4f},{rows[r.model]})" for r in d.itertuples())
        yl = (rf"ytick={{{','.join(str(i) for i in range(1, n + 1))}}}, yticklabels={{{labels}}},"
              if k == 0 else r"yticklabels={}, ytick={" + ",".join(str(i) for i in range(1, n + 1)) + "},")
        leg = ""   # markers are defined in the caption; a legend sat on the data
        plots.append(rf"""\nextgroupplot[title={{$h={h}$}}, {yl}]
\addplot[black!35, line width=0.5pt, forget plot] coordinates {{(1,0.4) (1,{n + 0.6:.1f})}};
{segs}
\addplot[only marks, mark=o, mark size=1.7pt, black] coordinates {{{pre}}};
\addplot[only marks, mark=*, mark size=1.7pt, black] coordinates {{{post}}};
{leg}""")
    tex = rf"""% GENERATED by scripts/make_tables.py -- do not edit by hand.
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=3 by 1, horizontal sep=0.35cm}},
  width=0.36\linewidth, height={max(0.36, 0.024 * n + 0.10):.2f}\linewidth,
  xmode=log, log basis x=10, log ticks with fixed point, xminorticks=false,
  xmin={lo:.3f}, xmax={hi:.3f}, xtick={{{xticks}}},
  ymin=0.4, ymax={n + 0.6:.1f},
  y tick label style={{font=\tiny}}, x tick label style={{font=\tiny, rotate=0}},
  title style={{font=\small, yshift=-0.6ex}},
  xtick align=outside, grid=none, xtick pos=bottom, ytick pos=left,
  axis line style={{black!45}}, tick style={{black!45}},
  legend style={{font=\tiny, at={{(0.97,0.03)}}, anchor=south east, draw=black!30, fill=white}},
  legend cell align=left,
]
{chr(10).join(plots)}
\end{{groupplot}}
\node[font=\small, anchor=north] at ($(group c2r1.south)+(0,-0.55cm)$)
  {{Median {loss.upper()} relative to HAR-log (log scale)}};
\end{{tikzpicture}}
"""
    (FIGS_OUT / f"fig_contamination_{ds}.tex").write_text(tex)
    print(f"  wrote {FIGS_OUT}/fig_contamination_{ds}.tex")


def figure_cost(ds: str, h: int = 1) -> None:
    losses, cost = read(f"losses_{ds}.csv"), read(f"cost_{ds}.csv")
    if losses.empty or cost.empty:
        return
    q = losses[(losses.window == "clean") & (losses.loss == "qlike") & (losses.h == h)
               & ~losses.model.str.contains(r"\+")]
    cost = cost.dropna(subset=["seconds"]).copy()
    cost["key"] = np.where(cost.variant.fillna("").eq(""), cost.model.astype(str),
                           cost.model.astype(str) + "[" + cost.variant.fillna("") + "]")
    merged = q.merge(cost.groupby("key", as_index=False).agg(seconds=("seconds", "median")),
                     left_on="model", right_on="key", how="inner")
    merged = merged[merged.seconds > 0]
    if merged.empty:
        return
    merged["fam"] = merged.model.map(family_of)
    shapes = {"trivial": "x", "har": "square", "garch": "triangle", "ml": "diamond",
              "deep": "o", "hybrid": "+", "tsfm24": "*", "tsfm25": "pentagon*"}
    plots = []
    for fam in [f for f in FAMILY_ORDER if f in set(merged.fam)]:
        sub = merged[merged.fam == fam]
        coords = " ".join(f"({r.seconds:.2f},{r.value:.5f})" for r in sub.itertuples())
        plots.append(rf"\addplot[only marks, mark={shapes.get(fam, 'o')}, mark size=2.0pt, black] "
                     rf"coordinates {{{coords}}};")
        plots.append(r"\addlegendentry{" + esc(FAMILY_LABEL.get(fam, fam)) + r"}")
    notable = pd.concat([merged.nsmallest(1, "seconds"), merged.nsmallest(1, "value")]).drop_duplicates()
    notes = "\n".join(rf"\node[anchor=south west, font=\tiny, text=black!70] at "
                      rf"(axis cs:{r.seconds:.2f},{r.value:.5f}) {{{esc(pretty(r.model))}}};"
                      for r in notable.itertuples())
    ymin, ymax = float(merged.value.min()) * 0.7, float(merged.value.max()) * 1.6
    tex = rf"""% GENERATED by scripts/make_tables.py -- do not edit by hand.
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.95\linewidth, height=0.70\linewidth,
  xlabel={{Wall-clock for one pass over the panel (s, log scale)}},
  ylabel={{Mean QLIKE (log scale, lower is better)}},
  xmode=log, log basis x=10, ymode=log, log basis y=10,
  ymin={ymin:.4f}, ymax={ymax:.1f},
  grid=major, grid style={{black!12}},
  axis line style={{black!45}}, tick style={{black!45}},
  legend style={{font=\tiny, at={{(0.02,0.98)}}, anchor=north west, draw=black!30, fill=white}},
  legend cell align=left,
  tick label style={{font=\small}}, label style={{font=\small}},
]
""" + "\n".join(plots) + "\n" + notes + r"""
\end{axis}
\end{tikzpicture}
"""
    (FIGS_OUT / f"fig_cost_{ds}.tex").write_text(tex)
    print(f"  wrote {FIGS_OUT}/fig_cost_{ds}.tex")


def main() -> int:
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    table_checkpoints()
    table_tickers()
    for ds in ("eq", "cx"):
        print(f"=== {ds} ===")
        table_sample(ds)
        figure_windows(ds)
        if read(f"losses_{ds}.csv").empty:
            print(f"  no losses_{ds}.csv yet -- results tables skipped")
            continue
        main_fams = ["trivial", "har", "garch", "ml", "deep", "hybrid", "tsfm24"]
        table_main(ds, "clean", f"tab_main_{ds}", main_caption(ds, "clean"), families=main_fams)
        table_main(ds, "clean", f"tab_mse_{ds}",
                   main_caption(ds, "clean", "MSE"), families=main_fams, loss="mse")
        table_main(ds, "frontier", f"tab_frontier_{ds}", main_caption(ds, "frontier"),
                   families=["trivial", "har", "ml", "deep", "hybrid", "tsfm24", "tsfm25"])
        table_main(ds, "stress", f"tab_stress_{ds}", main_caption(ds, "stress"), families=main_fams)
        table_ckpt_contrast(ds)
        table_calibration(ds)
        table_cost(ds)
        figure_contamination(ds)
        figure_cost(ds)
    table_contam_summary()
    table_spa_summary()
    table_double_oos()
    table_blocks()
    table_proxies()
    table_madl()
    table_daily()
    table_direction()
    return 0


if __name__ == "__main__":
    sys.exit(main())
