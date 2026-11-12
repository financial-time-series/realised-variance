"""Hybrid econometric-neural benchmark: an LSTM fed GARCH and implied variance.

    .venv/bin/python scripts/run_hybrid.py --dataset eq --seed 1 \
        --run-dir out/runs/eq-lstm-hybrid-s1

Added in revision, at a referee's request, as a stronger test of "keep HAR"
than the trivial, HAR, GARCH, ML and plain neural baselines. It follows the
design of Roszyk and Slepaczuk (2024), whose LSTM takes GARCH volatility
forecasts and the VIX as inputs alongside the series' own history, and of
Perekhodko and Slepaczuk (2025), whose LSTM takes a stochastic-volatility
forecast the same way: the econometric model's forecast is an INPUT to the
network, not a second forecast averaged with it.

INPUTS, per asset and day t, all visible at the close of t:
  z_rv   log RV_t
  z_g    log of the GARCH(1,1) one-step conditional variance sigma^2_{t+1|t},
         from daily close-to-close returns; parameters re-estimated at every
         refit on returns up to the refit date, then held fixed while the
         filter keeps updating on new returns (the protocol of the GARCH rows)
  z_iv   log of the VIX implied daily variance, (VIX/100)^2/252 -- equity only:
         no implied-volatility index covers the crypto sample
each expressed relative to s_t = log RV^(22)_t, the asset's own monthly cascade,
so that one network can serve sixty assets whose variance levels differ by an
order of magnitude (the same normalisation as the ML baselines). The network
reads the last 22 days of these inputs plus s_t, and emits all three horizons.

TARGET AND PROTOCOL. log(y_{t,h}) - s_t for h = 1, 5, 22, MSE loss. Training
rows are those whose longest target window closed before the refit date. Refit
every 252 origins with early stopping on the most recent tenth of the training
rows; the forecast is exp(prediction + s_t + sigma_h^2 / 2), the same
retransformation correction every fitted model in the study receives.
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
SEQ_LEN = 22
HIDDEN = 32


def garch_fit(r_pct: np.ndarray) -> tuple[float, float, float, float]:
    """GARCH(1,1) with constant mean on percent returns: (mu, omega, alpha, beta)."""
    from arch import arch_model
    res = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1,
                     dist="normal").fit(disp="off", show_warning=False)
    p = res.params
    return float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])


def garch_filter(r_pct: np.ndarray, mu: float, omega: float, alpha: float,
                 beta: float) -> np.ndarray:
    """sigma^2_{t+1|t} for every t, in DAILY VARIANCE units (not percent^2).

    out[t] uses returns up to and including t, and nothing later.
    """
    e2 = (r_pct - mu) ** 2
    s2 = np.empty(len(r_pct))
    prev = float(np.var(r_pct[: min(len(r_pct), 252)]))   # initial variance
    for t in range(len(r_pct)):
        prev = omega + alpha * e2[t] + beta * prev
        s2[t] = prev
    return s2 / 1e4


def build_model(n_in: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class Hybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_in, HIDDEN, batch_first=True)
            self.head = nn.Sequential(nn.Linear(HIDDEN + 1, HIDDEN), nn.ReLU(),
                                      nn.Linear(HIDDEN, len(HORIZONS)))

        def forward(self, seq, static):
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], static], dim=1))

    return Hybrid()


def train(model, Xs, Xc, Y, device, seed, max_epochs=40, patience=4, batch=2048):
    """Adam, MSE, early stopping on the most recent tenth (rows are time-sorted)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(Y)
    n_val = max(int(n * 0.1), 1)
    tr_idx, va_idx = np.arange(n - n_val), np.arange(n - n_val, n)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    Xs_t, Xc_t, Y_t = t(Xs), t(Xc), t(Y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        for b in range(0, len(perm), batch):
            i = torch.as_tensor(perm[b: b + batch], device=device)
            opt.zero_grad()
            loss = lossf(model(Xs_t[i], Xc_t[i]), Y_t[i])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = float(lossf(model(Xs_t[va_idx], Xc_t[va_idx]), Y_t[va_idx]))
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fitted = model(Xs_t[tr_idx], Xc_t[tr_idx]).cpu().numpy()
    resid_var = np.var(Y[tr_idx] - fitted, axis=0)
    return model, resid_var


def predict(model, Xs, Xc, device):
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(Xs), 8192):
            out.append(model(torch.as_tensor(Xs[b: b + 8192], dtype=torch.float32, device=device),
                             torch.as_tensor(Xc[b: b + 8192], dtype=torch.float32, device=device))
                       .cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, len(HORIZONS)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["eq", "cx"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rv-col", default="rv")
    ap.add_argument("--origin-start", default=None)
    ap.add_argument("--refit-every", type=int, default=252)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-refits", type=int, default=None, help="smoke tests")
    args = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"device: {device} ({gpu})")

    panel_path = Path(f"data/processed/{args.dataset}_rv.parquet")
    panel = Panel.load(panel_path, rv_col=args.rv_col)
    raw = pd.read_parquet(panel_path)
    uids = panel.uids[: args.limit] if args.limit else panel.uids
    origin_start = np.datetime64(args.origin_start or ("2010-01-01" if args.dataset == "eq"
                                                       else "2020-01-01"))
    use_iv = args.dataset == "eq"
    vix = (pd.read_parquet("data/processed/vix.parquet").set_index("ds")["iv_daily"]
           .sort_index()) if use_iv else None
    max_h = max(HORIZONS)

    # Per-series arrays that do not depend on the refit: RV features and targets.
    S = {}
    for uid in uids:
        s = panel.series[uid]
        rv = s.rv
        T = len(rv)
        csum = np.concatenate([[0.0], np.cumsum(rv)])
        c22 = np.full(T, np.nan)
        idx = np.arange(21, T)
        c22[idx] = (csum[idx + 1] - csum[idx + 1 - 22]) / 22
        scale = np.log(c22)
        ys = []
        for h in HORIZONS:
            y = np.full(T, np.nan)
            ok = np.arange(T) + h < T
            t_ok = np.arange(T)[ok]
            y[ok] = (csum[t_ok + 1 + h] - csum[t_ok + 1]) / h
            ys.append(y)
        g = raw[raw.unique_id == uid].sort_values("ds")
        ret = g["ret"].to_numpy() * 100.0
        ret = np.where(np.isfinite(ret), ret, 0.0)
        iv = (vix.reindex(pd.DatetimeIndex(s.dates), method="ffill").to_numpy()
              if use_iv else None)
        S[uid] = dict(dates=s.dates, rv=rv, scale=scale, y=np.vstack(ys).T,
                      ret=ret, iv=iv,
                      last_target_date=np.concatenate([s.dates[max_h:],
                                                       np.repeat(np.datetime64("NaT"), max_h)]))

    all_origins = np.sort(np.unique(np.concatenate([
        S[u]["dates"][MIN_HISTORY: len(S[u]["dates"]) - max_h] for u in uids])))
    eval_origins = all_origins[all_origins >= origin_start]
    refits = list(range(0, len(eval_origins), args.refit_every))
    if args.max_refits:
        refits = refits[: args.max_refits]
    print(f"{len(uids)} series, {len(eval_origins)} evaluation origins, "
          f"{len(refits)} refits, inputs: rv, garch" + (", vix" if use_iv else ""))

    t0 = time.perf_counter()
    rows, n_fits = [], 0
    for b in refits:
        fit_date = eval_origins[b]
        block_end = (eval_origins[b + args.refit_every]
                     if b + args.refit_every < len(eval_origins)
                     else eval_origins[-1] + np.timedelta64(1, "D"))
        Xs_tr, Xc_tr, Y_tr, D_tr = [], [], [], []
        Xs_te, Xc_te, meta_te = [], [], []
        for uid in uids:
            d = S[uid]
            dates = d["dates"]
            n_fit = int(np.searchsorted(dates, fit_date))     # returns before the refit date
            if n_fit < MIN_HISTORY:
                continue
            mu, om, al, be = garch_fit(d["ret"][:n_fit])
            gvar = garch_filter(d["ret"], mu, om, al, be)
            feats = [np.log(d["rv"]) - d["scale"], np.log(gvar) - d["scale"]]
            if use_iv:
                feats.append(np.log(d["iv"]) - d["scale"])
            F = np.vstack(feats).T                            # (T, k)
            Yrel = np.log(d["y"]) - d["scale"][:, None]       # (T, 3)
            ok_feat = np.isfinite(F).all(axis=1) & np.isfinite(d["scale"])
            T_len = len(dates)
            # sequences need SEQ_LEN finite rows behind t
            good = np.convolve(ok_feat.astype(int), np.ones(SEQ_LEN, int), "full")[:T_len] == SEQ_LEN
            t_all = np.arange(T_len)
            tr_mask = (good & (t_all >= MIN_HISTORY) & np.isfinite(Yrel).all(axis=1)
                       & (d["last_target_date"] < fit_date))
            te_mask = (good & (t_all >= MIN_HISTORY) & (t_all < T_len - max_h)
                       & (dates >= fit_date) & (dates < block_end) & (dates >= origin_start))
            for mask, Xs_l, Xc_l in ((tr_mask, Xs_tr, Xc_tr), (te_mask, Xs_te, Xc_te)):
                ts = t_all[mask]
                if len(ts) == 0:
                    continue
                win = ts[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
                Xs_l.append(F[win].astype(np.float32))
                Xc_l.append(d["scale"][ts][:, None].astype(np.float32))
                if mask is tr_mask:
                    Y_tr.append(Yrel[ts].astype(np.float32))
                    D_tr.append(dates[ts])
                else:
                    meta_te.append(pd.DataFrame({"unique_id": uid, "t": ts}))
        if not Xs_tr or not Xs_te:
            continue
        D = np.concatenate(D_tr)
        order = np.argsort(D, kind="stable")                  # time-sorted for validation
        Xs, Xc, Y = (np.concatenate(Xs_tr)[order], np.concatenate(Xc_tr)[order],
                     np.concatenate(Y_tr)[order])
        mu_s, sd_s = Xs.reshape(-1, Xs.shape[2]).mean(0), Xs.reshape(-1, Xs.shape[2]).std(0) + 1e-8
        mu_c, sd_c = Xc.mean(0), Xc.std(0) + 1e-8
        norm_s = lambda a: (a - mu_s) / sd_s
        norm_c = lambda a: (a - mu_c) / sd_c
        model = build_model(Xs.shape[2], args.seed).to(device)
        model, resid_var = train(model, norm_s(Xs), norm_c(Xc), Y, device, args.seed)
        n_fits += 1
        Xs_e, Xc_e = np.concatenate(Xs_te), np.concatenate(Xc_te)
        pred = predict(model, norm_s(Xs_e), norm_c(Xc_e), device)
        meta = pd.concat(meta_te, ignore_index=True)
        for j, h in enumerate(HORIZONS):
            for uid, g in meta.assign(p=pred[:, j]).groupby("unique_id"):
                d = S[uid]
                ts = g["t"].to_numpy()
                y_hat = np.exp(g["p"].to_numpy() + d["scale"][ts] + 0.5 * resid_var[j])
                y_hat, _ = floor_forecasts(y_hat, float(d["rv"][:MIN_HISTORY].min()))
                rows.append(pd.DataFrame({
                    "unique_id": uid, "origin": d["dates"][ts],
                    "ds": d["dates"][np.minimum(ts + h, len(d["dates"]) - 1)],
                    "h": h, "y": d["y"][ts, j], "y_hat": y_hat,
                    "model": "lstm-hybrid", "seed": args.seed}))
        print(f"  refit {str(fit_date)[:10]}: {len(Y):,} training rows, "
              f"{len(meta):,} test origins, resid var "
              f"{np.array2string(resid_var, precision=3)}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    elapsed = round(time.perf_counter() - t0, 2)
    preds = pd.concat(rows, ignore_index=True)
    print(f"lstm-hybrid seed {args.seed}: {elapsed}s, {n_fits} fits")
    write_run(
        args.run_dir, preds,
        config={"dataset": args.dataset, "family": "hybrid", "model": "lstm-hybrid",
                "seed": args.seed, "rv_col": args.rv_col,
                "origin_start": str(origin_start), "refit_every": args.refit_every,
                "inputs": ["log_rv", "log_garch_var"] + (["log_vix_iv"] if use_iv else []),
                "relative_to": "log RV22_t", "seq_len": SEQ_LEN, "hidden": HIDDEN,
                "loss": "MSE on log(y)-log(RV22_t)", "early_stopping": "last 10%, patience 4",
                "horizons": list(HORIZONS), "n_series": len(uids), "n_fits": n_fits,
                "limit": args.limit},
        timing={"fit_predict_seconds": elapsed, "device": device, "gpu": gpu,
                "cpu": platform.processor() or platform.machine(), "n_fits": n_fits},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
