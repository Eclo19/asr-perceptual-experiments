"""
Define the acceptable front-ends for the ASR class. Having modular front-ends is the 
backbone of the ASR class, as its purpose in this work is to compare different 
front-ends for automatic speech recognition in different conditions and with a set of
models. 

Author: Eric Oliveira 
Date: 10/27/2026

* Written with the help of Claude *

"""

from abc import ABC, abstractmethod
from typing import Sequence
import os
import dataclasses

import torch
import torchaudio
import numpy as np 

from .registry import register
from .config import FrontEndConfig, CarfacConfig, LogMelConfig, MFCCConfig
from .constants import HOP_LENGTH, SAMPLE_RATE


# ==============================
# Frontend ABC
# ==============================

class FrontEnd(ABC):
    """Waveform -> (T, F) features at a fixed 100 frames/s.

    Contract that every front end must satisfy (and that tests check):
      * input:  f32[N] mono at 16 kHz, already level-calibrated by the Condition
      * output: f32[T, F] with F == n_feats and T == N // 160 + 1 (±1)
      * channels ordered low → high frequency (CARFAC's channel 0 is the HIGHEST CF: flip it)
      * extract() is deterministic and is the only expensive step; post() is cheap
    """
    cfg: FrontEndConfig
    n_feats: int
    # If True: features are cached under features/
    expensive: bool = False                 

    @abstractmethod
    def extract(self, wave: torch.Tensor) -> torch.Tensor:
        """Raw features f32[T, F]. This is what gets cached."""

    def extract_batch(self, waves: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Many utterances at once; must equal [extract(w) for w in waves]. Override when batching is faster."""
        return [self.extract(w) for w in waves]

    def post(self, raw: torch.Tensor) -> torch.Tensor:
        """Cheap transform applied after the cache (log / cube root / identity)."""
        return raw

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        return self.post(self.extract(wave))

    @staticmethod
    def expected_frames(n_samples: int) -> int:
        return n_samples // HOP_LENGTH + 1

    @property
    def name(self) -> str:
        return self.cfg.name


@register("frontend", "logmel")
class LogMel(FrontEnd):
    """Log-mel filterbank (torchaudio), set entirely by LogMelConfig and the shared constants.

    25 ms Hann window (400 samples, zero-padded to n_fft = 512, the next power of 2), 10 ms hop,
    power spectrum → n_mels triangular filters. center=True pads the signal, so
    T = N // HOP_LENGTH + 1, the frame-count contract.
    """

    def __init__(self, cfg: LogMelConfig):
        self.cfg = cfg
        self.n_feats = cfg.n_mels

        # Window length in samples: 25 ms at 16 kHz → 400
        win_length = int(round(cfg.win_ms / 1000 * SAMPLE_RATE))

        # n_fft: the next power of 2 above the window (512), a deliberate change from the notebook's 400
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft= 1 << (win_length - 1).bit_length(), # Next power of 2
            win_length=win_length,
            hop_length=HOP_LENGTH,
            n_mels=cfg.n_mels,
            f_max=cfg.f_max,                         # None = Nyquist
        )

    def extract(self, wave: torch.Tensor) -> torch.Tensor:
        """f32[N] → mel power spectrogram f32[T, n_mels] (no log yet: that's post())."""
        return self.melspec(wave).transpose(-1, -2)          # (n_mels, T) -> (T, n_mels)

    def post(self, raw: torch.Tensor) -> torch.Tensor:
        """Log compression; log_eps keeps silent bins finite."""
        return torch.log(raw + self.cfg.log_eps)


@register("frontend", "mfcc")
class MFCC(FrontEnd):
    """MFCCs (torchaudio). Already log + DCT, so post() is the identity."""

    def __init__(self, cfg: MFCCConfig):
        self.cfg = cfg
        self.n_feats = cfg.n_mfcc

        # Same framing as LogMel: window from cfg.win_ms, FFT rounded up to a power of 2, 10 ms hop
        win_length = int(round(cfg.win_ms / 1000 * SAMPLE_RATE))
        n_fft = 1 << (win_length - 1).bit_length()

        # melkwargs is required: torchaudio's defaults are hop 200 and 128 mels
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=cfg.n_mfcc,
            melkwargs=dict(n_fft=n_fft, win_length=win_length,
                           hop_length=HOP_LENGTH, n_mels=cfg.n_mels),
        )

    def extract(self, wave: torch.Tensor) -> torch.Tensor:
        return self.mfcc(wave).transpose(-1, -2)            # (n_mfcc, T) → (T, n_mfcc)


@register("frontend", "carfac")
class CARFAC(FrontEnd):
    """CARFAC v2 NAP (JAX port), pooled to 100 frames/s (25 ms Hann windows by default, like log-mel).

    Design happens once, in __init__, and every utterance starts from the same initial
    state. That's safe because JAX is functional: running an utterance returns a NEW
    state instead of changing the old one. (The NumPy port mutates its cfp in place,
    so there you'd need a fresh deepcopy per utterance.)

    extract() returns the firing RATE: NAP + rest_output, always ≥ 0, with channels
    flipped to low → high CF. post() then applies log / cube root / nothing.

    `jax` and `carfac` are imported inside __init__, so the rest of the library works
    on machines without them.
    """
    expensive = True
    CHUNK = SAMPLE_RATE                     # 1 s of samples per lax.scan step

    # hann25 pooling works on a 1 kHz copy of the NAP: means of 16-sample blocks
    BLOCK = 16                              # samples per 1 ms block
    HANN_BLOCKS = 25                        # 25 ms window = 25 blocks

    def __init__(self, cfg: CarfacConfig):
        self.cfg = cfg

        # Stop JAX from reserving most of the GPU at import, which would starve PyTorch
        # (only takes effect if jax hasn't been imported yet in this process)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        # Imported here, not at the top, so machines without JAX/CARFAC can still use mel and MFCC
        import jax
        import jax.numpy as jnp
        import carfac.jax.carfac as cj
        self._jax, self._jnp, self._cj = jax, jnp, cj

        # 1. Design the cochlea once; every utterance starts from the same initial state
        self.hypers, self.weights, self.state0 = cj.design_and_init_carfac(self._design_params())

        # 2. Refuse designs the JAX port computes wrongly (5-tap AGC smoother, see carfac_setup_notes)
        taps = [stage.agc_spatial_n_taps for stage in self.hypers.ears[0].agc]
        if any(t != 3 for t in taps):
            raise ValueError(f"AGC spatial smoothers need {taps} taps, but CARFAC-JAX only applies 3 "
                             f"correctly. Raise agc_params['decimation'][0] (e.g. 6) or apply the patch.")

        # 3. Facts about this design that the rest of the library needs
        self.n_feats = int(self.hypers.ears[0].car.n_ch)                                  # 65 at defaults
        self.cf_hz = np.asarray(self.hypers.ears[0].pole_freqs)[::-1].copy()             # low → high CF
        self.rest_output = float(np.ravel(self.weights.ears[0].ihc.rest_output)[0])      # ≈ 0.81, two_cap

        # 4. hann25 pooling weights: log-mel's 400-sample Hann window (periodic, as torchaudio's),
        #    sampled at the centre of each 1 ms block, then scaled to sum to 1 (a weighted MEAN,
        #    so both poolings give firing rates on the same scale)
        centers = np.arange(self.HANN_BLOCKS) * self.BLOCK + self.BLOCK / 2                # 8, 24, ..., 392
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * centers / (self.HANN_BLOCKS * self.BLOCK))
        self._hann = jnp.asarray(hann / hann.sum(), dtype=jnp.float32)                    # (25,)

        # 5. Compile: vmap batches _one over utterances, jit compiles it (once per batch shape)
        self._batched = jax.jit(jax.vmap(self._one))

    def _design_params(self):
        """CarfacConfig -> CarfacDesignParameters(fs=16000), one parameter group at a time."""
        params = self._cj.CarfacDesignParameters(fs=float(SAMPLE_RATE))
        ear = params.ears[0]                                                               # mono: one ear

        groups = {"car": self.cfg.car_params, "agc": self.cfg.agc_params, "ihc": self.cfg.ihc_params}
        for group, overrides in groups.items():
            target = getattr(ear, group)
            valid = {f.name for f in dataclasses.fields(target)}

            for key, value in overrides.items():
                # Same behaviour as CARFAC's own Params classes: unknown names are an error
                if key not in valid:
                    raise AttributeError(f"{type(target).__name__} has no field {key!r}. "
                                         f"Valid {group}_params: {sorted(valid)}")
                setattr(target, key, value)
        return params

    def _one(self, wave):
        """One utterance, wave f32[L] with L a multiple of CHUNK → pooled NAP f32[L // 160, n_ch].

        hann25: frame t is centred on sample 160·t, like log-mel's frames (center=True).
        box10 keeps the original definition: the mean of samples [160·t, 160·t + 160).
        """
        cj, jnp = self._cj, self._jnp
        box10 = self.cfg.pooling == "box10"

        # Samples averaged per output row of each chunk: 160 (box10: final frames) or 16 (1 kHz blocks)
        group = HOP_LENGTH if box10 else self.BLOCK

        def step(state, chunk):
            # Run 1 s of audio, continuing from the previous chunk's state
            out = cj.run_segment(chunk[:, None], self.hypers, self.weights, state,
                                 open_loop=self.cfg.open_loop)
            nap = out.naps[:, :, 0]                                                       # (16000, n_ch)

            # Mean over each group of samples: 100 frames (box10) or 1000 blocks (hann25) per chunk
            pooled = nap.reshape(self.CHUNK // group, group, -1).mean(axis=1)              # (100 or 1000, n_ch)
            return out.state, pooled

        _, rows = self._jax.lax.scan(step, self.state0, wave.reshape(-1, self.CHUNK))    # (n_chunks, rows, n_ch)
        rows = rows.reshape(-1, self.n_feats)                                             # (L // group, n_ch)
        if box10:
            return rows                                                                   # (L // 160, n_ch)

        # hann25: a 25 ms Hann window every 10 ms over the 1 kHz NAP. Frame t covers blocks 10t-12 .. 10t+12,
        # i.e. samples 160t-192 .. 160t+208 (log-mel: 160t-200 .. 160t+200). Zero padding before the
        # start = the NAP of silence (the resting rate is added later, in extract_batch)
        step_blocks = HOP_LENGTH // self.BLOCK                                            # 10
        n_frames = rows.shape[0] // step_blocks                                           # L // 160
        half = self.HANN_BLOCKS // 2                                                      # 12
        padded = jnp.pad(rows, ((half, half), (0, 0)))                                    # (L // 16 + 24, n_ch)

        # Weighted sum, one window tap at a time: tap k reads every 10th block starting at block k.
        # (25 strided slices instead of a (frames, 25, n_ch) gather: same result, 25x less memory)
        frames = jnp.zeros((n_frames, self.n_feats), rows.dtype)
        for k in range(self.HANN_BLOCKS):
            frames = frames + self._hann[k] * padded[k: k + step_blocks * n_frames: step_blocks]
        return frames                                                                     # (L // 160, n_ch)

    def extract_batch(self, waves: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Several utterances at once -> list of firing RATE tensors f32[N_i // 160 + 1, n_feats]."""
        # Accept torch tensors or arrays; CARFAC wants float32 NumPy/JAX arrays
        arrays = [np.asarray(w.detach().cpu() if isinstance(w, torch.Tensor) else w,
                             dtype=np.float32).reshape(-1) for w in waves]
        lengths = [len(a) for a in arrays]

        # Pad every utterance to the same whole number of seconds, with room for the last frame
        padded_len = -(-(max(lengths) + HOP_LENGTH) // self.CHUNK) * self.CHUNK            # ceil to 16000
        batch = np.zeros((len(arrays), padded_len), np.float32)
        for i, a in enumerate(arrays):
            batch[i, :len(a)] = a

        # Run CARFAC on the whole batch; zero padding can't leak backwards because CARFAC is causal
        pooled = np.asarray(self._batched(self._jnp.asarray(batch)))                      # (B, padded_len // 160, n_ch)

        out = []
        for i, n in enumerate(lengths):
            # Keep the frame-count contract, flip channels to low → high CF, add the resting rate
            rate = pooled[i, : n // HOP_LENGTH + 1, ::-1] + self.rest_output
            out.append(torch.from_numpy(np.ascontiguousarray(rate)))                     # (T, n_feats), ≥ 0
        return out

    def extract(self, wave: torch.Tensor) -> torch.Tensor:
        return self.extract_batch([wave])[0]

    def post(self, raw: torch.Tensor) -> torch.Tensor:
        """Compress the firing rate: 'log', 'cuberoot', or 'none'."""
        rate = raw.clamp_min(0)                                                           # guard tiny negatives
        if self.cfg.post == "log":
            return torch.log(rate + self.cfg.log_eps)
        if self.cfg.post == "cuberoot":
            return rate.pow(1 / 3)
        return raw
        