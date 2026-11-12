"""The zero-shot forecaster registry, and one interface over four model families.

Every foundation model here has its own loading ritual, its own context
convention and its own idea of what a "prediction" is. `ZeroShotForecaster`
hides all of it behind:

    fc = load("chronos-bolt-base", device="cuda")
    y_hat = fc.predict(contexts, h)       # (n_series, h) point forecasts

so `scripts/run_tsfm.py` contains no per-model branching and adding a family is
a new subclass rather than a new script.

RELEASE DATES ARE DATA, NOT TRIVIA. `released` is the checkpoint's creation
date on the Hugging Face Hub, read from the API and verified 2026-08-07 by
`scripts/prefetch.py`, which fails loudly if the Hub and this file disagree. A
model cannot be credited with forecasting a period it may have been trained on.
Meyer et al. report a case in which leakage touching 1.45% of series (0.78% of
observations) moved MAE from 439.14 to 248.24 -- a ~43% apparent improvement
bought entirely by contamination.

REVISIONS ARE PINNED, AND THE FILES' OWN DATES COUNT. A Hub creation date is
when the repository appeared, not when the weights in it were last replaced.
Each Spec therefore also records the commit the study loads (`revision`) and the
date on which the weight or configuration files at that commit last changed
(`files_changed`, from the Hub's per-file lastCommit). The date a checkpoint may
be evaluated from is the LATER of the two (`effective_release`). For TimesFM-2.5
they differ by a month: the repository was created on 2025-09-02 and its weights
were replaced ("Refreshed model with speed optimizations") on 2025-10-01. None of
the nine repositories has changed since the first campaign ran (2026-08-07), so
the revisions below are also the ones that campaign used.

TWO GENERATIONS, TWO WINDOWS. By August 2026 the field has turned over once.
Evaluating only the 2024 checkpoints would test obsolete models; pooling both
generations into one window would silently hand the 2025 checkpoints a period
they may have trained on. So each generation is scored on a window that
post-dates ALL of its members, and the two comparisons are reported separately.
Every baseline is re-scored on each window, so within a table the comparison is
like-for-like.

    generation  newest checkpoint            window opens
    g2024       timesfm-2.0    2024-12-24    2025-01-01
    g2025       chronos-2      2025-10-30    2025-11-01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Window openings, derived from the checkpoint dates below by
# check_release_dates(). Never edit one without re-running that check.
GENERATION_WINDOWS: dict[str, str] = {
    "g2024": "2025-01-01",
    "g2025": "2025-11-01",
}

# Kept as the name the rest of the codebase asks for when it means "the primary
# leakage-controlled window".
LEAKAGE_FREE_START = GENERATION_WINDOWS["g2024"]


@dataclass(frozen=True)
class Spec:
    """What we need to know about a checkpoint before running it."""

    name: str
    repo: str
    released: str          # HF Hub createdAt, verified by scripts/prefetch.py
    family: str
    generation: str
    max_context: int       # longest input the checkpoint accepts, in periods
    batch_size: int = 128  # per-family default; Moirai needs far less
    extra: dict = field(default_factory=dict)
    revision: str = "main"            # pinned commit; verified by scripts/prefetch.py
    files_changed: str | None = None  # last change to the weight/config files loaded

    @property
    def effective_release(self) -> str:
        """The earliest date this checkpoint may honestly be evaluated from."""
        return max(self.released, self.files_changed or self.released)


ZOO: dict[str, Spec] = {
    # ---- 2024 generation ----------------------------------------------------
    # Bolt is the direct multi-step successor to the T5 sampler: no sampling
    # loop, so roughly an order of magnitude cheaper per forecast.
    "chronos-bolt-small": Spec(
        "chronos-bolt-small", "amazon/chronos-bolt-small", "2024-11-25",
        "chronos", "g2024", max_context=2048,
        revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8", files_changed="2024-11-13"),
    "chronos-bolt-base": Spec(
        "chronos-bolt-base", "amazon/chronos-bolt-base", "2024-11-25",
        "chronos", "g2024", max_context=2048,
        revision="5d9f166d69f47aef3401367a7b842e78fe97b121", files_changed="2024-11-13"),
    # The original T5 sampler: the checkpoint the published Chronos results
    # were established on, and released nine months before Bolt.
    #
    # batch 24, not 128: Chronos-T5 is a SAMPLER, so its KV cache grows with
    # batch x samples x context and it OOMs a 44 GB A100 at 128. The Bolt and
    # Chronos-2 checkpoints are direct multi-step and do not pay this cost.
    "chronos-t5-base": Spec(
        "chronos-t5-base", "amazon/chronos-t5-base", "2024-02-21",
        "chronos", "g2024", max_context=512, batch_size=24,
        # model.safetensors was added on 2024-03-27 as a format variant of the
        # original weights; the later date is used, conservatively.
        revision="ad294eaacead15db499b740ea4122266dd2a81a2", files_changed="2024-03-27"),
    "moirai-1.1-base": Spec(
        "moirai-1.1-base", "Salesforce/moirai-1.1-R-base", "2024-06-14",
        "moirai", "g2024", max_context=1024, batch_size=64,
        extra={"patch_size": 32, "num_samples": 20},
        revision="a0358780de4cdd7ac7963b12ae9985d225d55026", files_changed="2024-06-14"),
    "moirai-moe-base": Spec(
        "moirai-moe-base", "Salesforce/moirai-moe-1.0-R-base", "2024-11-01",
        "moirai-moe", "g2024", max_context=1024, batch_size=32,
        extra={"patch_size": 16, "num_samples": 20},
        revision="af123c57cfb02858c599d4e7701dd619a0fc53c0", files_changed="2024-11-15"),
    # Two orders of magnitude smaller than the others. Included precisely
    # because a 1M-parameter model beating a 500M one would be the story.
    "ttm-r2": Spec(
        "ttm-r2", "ibm-granite/granite-timeseries-ttm-r2", "2024-10-08",
        "ttm", "g2024", max_context=512, batch_size=256,
        revision="d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f", files_changed="2024-10-09"),

    # ---- 2025 generation ----------------------------------------------------
    "chronos-2": Spec(
        "chronos-2", "amazon/chronos-2", "2025-10-30",
        "chronos", "g2025", max_context=2048, batch_size=64,
        revision="29ec3766d36d6f73f0696f85560a422f50e8498c", files_changed="2025-10-30"),
    "timesfm-2.5": Spec(
        "timesfm-2.5", "google/timesfm-2.5-200m-pytorch", "2025-09-02",
        "timesfm", "g2025", max_context=2048, batch_size=128,
        # Weights replaced on 2025-10-01 ("Refreshed model with speed
        # optimizations"): that, not the creation date, is when THESE weights
        # became public.
        revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3", files_changed="2025-10-01"),
    "moirai-2.0": Spec(
        "moirai-2.0", "Salesforce/moirai-2.0-R-small", "2025-08-06",
        "moirai2", "g2025", max_context=1024, batch_size=64,
        extra={"num_samples": 20},
        revision="30f43ff08c8494f4943ae1521e9d4e94a0fbb389", files_changed="2025-08-06"),
}


def generation_of(name: str) -> str:
    return ZOO[name].generation


def window_for(name: str) -> str:
    """The generation window: the earliest origin at which this checkpoint is
    compared JOINTLY with every other member of its generation."""
    return GENERATION_WINDOWS[ZOO[name].generation]


def release_for(name: str) -> str:
    """The checkpoint's own release date: the earliest origin at which IT may
    honestly be evaluated. Used by the per-checkpoint before/after contrast."""
    return ZOO[name].effective_release


def cached_revision(repo: str) -> str | None:
    """The commit `main` resolves to in the local HF cache, or None.

    Jobs run offline, so this is the revision actually loaded. Recorded in each
    run's config.json so a reader can match a forecast to exact weights.
    """
    import os
    from pathlib import Path
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    ref = root / "hub" / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def check_release_dates() -> None:
    """Fail if any generation's window does not post-date all its members."""
    for gen, start in GENERATION_WINDOWS.items():
        members = [s for s in ZOO.values() if s.generation == gen]
        if not members:
            continue
        newest = max(s.effective_release for s in members)
        if newest >= start:
            raise ValueError(
                f"generation {gen}: checkpoint released {newest} is not before "
                f"its window start {start}. The evaluation would include the "
                f"pretraining period. Move the window."
            )


# --- the uniform interface ---------------------------------------------------

class ZeroShotForecaster:
    """Point forecasts from a batch of contexts. Subclasses implement _predict."""

    def __init__(self, spec: Spec, device: str, context_len: int | None = None) -> None:
        self.spec = spec
        self.device = device
        self.context_len = min(context_len or spec.max_context, spec.max_context)

    def predict(self, contexts: list[np.ndarray], h: int) -> np.ndarray:
        """(n_series, h) point forecasts from n variable-length contexts.

        Truncation happens HERE, once, so no subclass can quietly accept more
        history than it was built for and no two models get different amounts
        of it by accident.
        """
        trimmed = [np.asarray(c, dtype=np.float32)[-self.context_len:]
                   for c in contexts]
        out = np.asarray(self._predict(trimmed, h), dtype=np.float64)
        if out.shape != (len(contexts), h):
            raise ValueError(
                f"{self.spec.name} returned {out.shape}, expected {(len(contexts), h)}")
        return out

    def _predict(self, contexts: list[np.ndarray], h: int) -> np.ndarray:
        raise NotImplementedError


class Chronos(ZeroShotForecaster):
    """Chronos-T5, Chronos-Bolt and Chronos-2 all dispatch from one base class."""

    def __init__(self, spec, device, context_len=None):
        super().__init__(spec, device, context_len)
        import torch
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            spec.repo, device_map=device,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        )

    def _predict(self, contexts, h):
        import torch
        inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
        # The median path. Chronos is probabilistic; the median is the point
        # summary its own papers report.
        q, _ = self.pipe.predict_quantiles(
            inputs, prediction_length=h, quantile_levels=[0.5])
        # Chronos-2 returns a LIST of (variate, horizon, quantile) tensors --
        # it is multivariate-capable, so it cannot assume a common shape. The
        # T5 and Bolt pipelines return one stacked (batch, horizon, quantile)
        # tensor. Handle both rather than depending on which checkpoint loaded.
        if isinstance(q, (list, tuple)):
            return np.stack([np.asarray(t.float().cpu()).reshape(-1, 1)[:h, 0]
                             if t.ndim == 2 else
                             np.asarray(t.float().cpu())[0, :h, 0] for t in q])
        return q[:, :h, 0].float().cpu().numpy()


class TimesFM(ZeroShotForecaster):
    """TimesFM 2.5. The 2.0 class no longer exists in the shipped package --
    `timesfm` 2.0.2 exposes only TimesFM_2p5_200M_torch, so 2.0 cannot be run
    without pinning an older release. Recorded here because it is the reason
    TimesFM appears only in the 2025 generation."""

    def __init__(self, spec, device, context_len=None):
        super().__init__(spec, device, context_len)
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
        self.model = TimesFM_2p5_200M_torch.from_pretrained(spec.repo)
        self.model.compile(ForecastConfig(
            max_context=self.context_len, max_horizon=64,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True,
            fix_quantile_crossing=True,
        ))

    def _predict(self, contexts, h):
        point, _ = self.model.forecast(horizon=h, inputs=list(contexts))
        return np.asarray(point)[:, :h]


class Moirai(ZeroShotForecaster):
    """Moirai / Moirai-MoE via uni2ts.

    Context length is forced to a multiple of the patch size: uni2ts pads
    internally and a non-multiple produces a tensor-shape error deep in the
    attention bias rather than anything readable.
    """

    def __init__(self, spec, device, context_len=None):
        super().__init__(spec, device, context_len)
        patch = spec.extra.get("patch_size", 32)
        self.patch = patch
        self.context_len = (self.context_len // patch) * patch
        self.num_samples = spec.extra.get("num_samples", 20)
        self._cache: dict[int, object] = {}

    def _model(self, h: int):
        if h in self._cache:
            return self._cache[h]
        if self.spec.family == "moirai-moe":
            from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
            module = MoiraiMoEModule.from_pretrained(self.spec.repo)
            fc = MoiraiMoEForecast(
                module=module, prediction_length=h, context_length=self.context_len,
                patch_size=self.patch, num_samples=self.num_samples, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        else:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            module = MoiraiModule.from_pretrained(self.spec.repo)
            fc = MoiraiForecast(
                module=module, prediction_length=h, context_length=self.context_len,
                patch_size=self.patch, num_samples=self.num_samples, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        self._cache[h] = fc.to(self.device).eval()
        return self._cache[h]

    def _predict(self, contexts, h):
        import torch
        model = self._model(h)
        L = self.context_len
        out = []
        with torch.no_grad():
            for i in range(0, len(contexts), self.spec.batch_size):
                chunk = contexts[i: i + self.spec.batch_size]
                arr = np.stack([
                    np.concatenate([np.full(L - len(c), c[0], dtype=np.float32), c])
                    if len(c) < L else c[-L:]
                    for c in chunk])
                past = torch.tensor(arr, dtype=torch.float32,
                                    device=self.device).unsqueeze(-1)
                obs = torch.ones_like(past, dtype=torch.bool)
                pad = torch.zeros(past.shape[:2], dtype=torch.bool, device=self.device)
                samples = model(past_target=past, past_observed_target=obs,
                                past_is_pad=pad)
                arr = samples.float().cpu().numpy()
                # (batch, samples, h) -- median over the sample dimension
                out.append(np.median(arr, axis=1).reshape(len(chunk), -1)[:, :h])
        return np.concatenate(out, axis=0)


class Moirai2(ZeroShotForecaster):
    """Moirai 2.0 ships a different entry point from the 1.x line."""

    def __init__(self, spec, device, context_len=None):
        super().__init__(spec, device, context_len)
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        self.module = Moirai2Module.from_pretrained(spec.repo)
        self._cache: dict[int, object] = {}

    def _model(self, h: int):
        if h not in self._cache:
            from uni2ts.model.moirai2 import Moirai2Forecast
            # Unlike the 1.x line, Moirai2Forecast takes no num_samples: the
            # 2.0 head is quantile-based rather than sampled.
            fc = Moirai2Forecast(
                module=self.module, prediction_length=h,
                context_length=self.context_len, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            self._cache[h] = fc.to(self.device).eval()
        return self._cache[h]

    def _predict(self, contexts, h):
        import torch
        model = self._model(h)
        L = self.context_len
        out = []
        with torch.no_grad():
            for i in range(0, len(contexts), self.spec.batch_size):
                chunk = contexts[i: i + self.spec.batch_size]
                arr = np.stack([
                    np.concatenate([np.full(L - len(c), c[0], dtype=np.float32), c])
                    if len(c) < L else c[-L:] for c in chunk])
                past = torch.tensor(arr, dtype=torch.float32,
                                    device=self.device).unsqueeze(-1)
                obs = torch.ones_like(past, dtype=torch.bool)
                pad = torch.zeros(past.shape[:2], dtype=torch.bool, device=self.device)
                samples = model(past_target=past, past_observed_target=obs,
                                past_is_pad=pad)
                a = samples.float().cpu().numpy()
                out.append(np.median(a, axis=1).reshape(len(chunk), -1)[:, :h])
        return np.concatenate(out, axis=0)


class TinyTimeMixer(ZeroShotForecaster):
    """TTM is fixed-shape: it accepts exactly `context_length` points and emits
    exactly `prediction_length`. Short contexts are left-padded with their own
    first value and the head is sliced to h -- both recorded because they are
    real differences in what this model sees and returns."""

    def __init__(self, spec, device, context_len=None):
        super().__init__(spec, device, context_len)
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            spec.repo, ignore_mismatched_sizes=True).to(device).eval()
        self.ctx = int(self.model.config.context_length)
        self.head = int(self.model.config.prediction_length)
        self.context_len = self.ctx

    def _predict(self, contexts, h):
        import torch
        if h > self.head:
            raise ValueError(
                f"{self.spec.name} head emits {self.head} steps, need {h}")
        arr = np.stack([
            np.concatenate([np.full(self.ctx - len(c), c[0], dtype=np.float32), c])
            if len(c) < self.ctx else c[-self.ctx:] for c in contexts])
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), self.spec.batch_size):
                batch = torch.tensor(arr[i: i + self.spec.batch_size],
                                     dtype=torch.float32,
                                     device=self.device).unsqueeze(-1)
                pred = self.model(past_values=batch).prediction_outputs
                out.append(pred[:, :h, 0].float().cpu().numpy())
        return np.concatenate(out, axis=0)


_LOADERS: dict[str, Callable[..., ZeroShotForecaster]] = {
    "chronos": Chronos,
    "timesfm": TimesFM,
    "moirai": Moirai,
    "moirai-moe": Moirai,
    "moirai2": Moirai2,
    "ttm": TinyTimeMixer,
}


def load(name: str, device: str = "cuda",
         context_len: int | None = None) -> ZeroShotForecaster:
    if name not in ZOO:
        raise KeyError(f"unknown model {name!r}; have {sorted(ZOO)}")
    spec = ZOO[name]
    return _LOADERS[spec.family](spec, device, context_len)
