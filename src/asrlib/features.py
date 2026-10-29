"""
Featurize input waveforms according to the front end. This pipeline is the same for
every front end. The pipeline:

    level/noise -> extract -> post -> normalize -> SpecAugment

Training, evaluation and transcribe() all use it.

Author: Eric Oliveira
Date: 10/27/2026

* Written with the help of Claude *

"""

from __future__ import annotations

from typing import Literal

import torch

from .conditions import Condition, NoiseBank, set_level
from .frontends import FrontEnd


class Normalizer:
    """Feature normalization after post(). 'utt_cmvn' = per-utterance, per-channel mean/variance.

    It's stateless, so there's nothing to save. A future 'global' mode would store its
    statistics in state_dict(), and load() would restore them.
    """
    MODES = ("utt_cmvn", "none")

    def __init__(self, mode: Literal["utt_cmvn", "none"] = "utt_cmvn"):
        # Fail on a typo, not halfway through training
        if mode not in self.MODES:
            raise ValueError(f"norm must be one of {self.MODES}, not {mode!r}")
        self.mode = mode

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        """(T, F) for ONE utterance → (T, F). Must run before padding, on real frames only."""
        if self.mode == "none":
            return feats

        # Z-score each channel over time; correction=0 keeps one-frame utterances finite
        mean = feats.mean(dim=0, keepdim=True)                     # (1, F)
        std = feats.std(dim=0, keepdim=True, correction=0)         # (1, F)
        return (feats - mean) / (std + 1e-5)                       # (T, F): mean 0, std 1 per channel

    def state_dict(self) -> dict:
        return {"mode": self.mode}


class Featurizer:
    """THE feature pipeline. Training, evaluation and transcribe() all call this one
    object, so inference computes features exactly the way training did.

        wave (16 kHz) ─► condition.apply (level + noise) or level calibration only
                      ─► frontend.extract ─► [cache] ─► frontend.post ─► normalizer ─► [SpecAugment if train]

    Resampling to 16 kHz happens before this, in ASR._load_audio and the corpora.
    """

    def __init__(self, frontend: FrontEnd, normalizer: Normalizer,
                 specaugment: dict | None, train_level_db_spl: float):
        self.frontend = frontend
        self.normalizer = normalizer
        self.specaugment_cfg = specaugment
        # The speech level (dB SPL) every training utterance was scaled to; transcribe() matches it
        self.train_level_db_spl = train_level_db_spl

    def __call__(self, wave, *, condition=None, utt_id="adhoc", bank=None, variant: int = 0,
                 train=False, calibrate_level=True):
        """Waveform f32[N] at 16 kHz → model-ready f32[T, F].

        condition given:  condition.apply(...) sets the level and adds noise.
        condition None:   only level calibration to train_level_db_spl (when calibrate_level),
                          so an arbitrary .wav reaches the front end at the level it was trained on.
        train=True adds SpecAugment; it's never applied at evaluation.
        """
        # 1. Level and noise: a Condition does both; otherwise just match the training level
        if condition is not None:
            wave = condition.apply(wave, utt_id, bank=bank, variant=variant)
        elif calibrate_level:
            wave = set_level(wave, self.train_level_db_spl)

        # 2. The front end's expensive step, then everything after the cache
        raw = self.frontend.extract(wave)                          # (T, F)
        return self.from_raw(raw, train=train)

    def from_raw(self, raw: torch.Tensor, train: bool = False) -> torch.Tensor:
        """The same pipeline starting from cached extract() output: post → normalize → [SpecAugment]."""
        # Cached features are float16 on disk; compute in float32
        feats = self.frontend.post(raw.float())                    # (T, F)
        feats = self.normalizer(feats)

        # SpecAugment only while training
        if train and self.specaugment_cfg:
            feats = self.spec_augment(feats)
        return feats

    def spec_augment(self, feats: torch.Tensor) -> torch.Tensor:
        """Frequency and time masks (Park et al., 2019) on one utterance, (T, F) → (T, F).

        Each mask has a random width in [0, max_width] and a random start. Masked cells are
        filled with the utterance mean (0 after CMVN, as in the notebook), so they carry no information.
        With scale_freq_to_n_feats, the max frequency width is freq_width * n_feats / 80, so
        65 CARFAC channels lose the same fraction of the axis as 80 mel bins.
        """
        cfg = self.specaugment_cfg
        n_feats = feats.shape[1]

        # Max frequency width, optionally scaled to this front end's channel count
        freq_width = cfg["freq_width"]
        if cfg.get("scale_freq_to_n_feats", False):
            freq_width = round(freq_width * n_feats / 80)

        # Copy before masking, so the caller's tensor (e.g. from the cache) is never modified
        out = feats.clone()
        fill = feats.mean()

        # Mask (axis, how many masks, max width): axis 1 = frequency, axis 0 = time
        for axis, n_masks, max_width in ((1, cfg["freq_masks"], freq_width),
                                         (0, cfg["time_masks"], cfg["time_width"])):
            size = out.shape[axis]
            for _ in range(n_masks):
                # Random width, never wider than the axis; then a random start where it fits
                width = int(torch.randint(0, min(max_width, size) + 1, (1,)))
                start = int(torch.randint(0, size - width + 1, (1,)))
                if axis == 1:
                    out[:, start:start + width] = fill
                else:
                    out[start:start + width, :] = fill
        return out