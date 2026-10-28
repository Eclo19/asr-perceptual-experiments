"""
asr_skeleton.py: skeleton of the ASR class and everything it is built from (v2).

One frozen config (`ASRConfig`) describes a whole system: front end, dataset,
training noise and SNRs, model architecture and training recipe.
`ASR(cfg)` builds an untrained system from it, `fit()` trains it,
`transcribe()` uses it, `evaluate()` / `benchmark()` score it, and
`save()` / `ASR.load()` round-trip it.

To "change an internal state", derive a NEW system:
    asr2 = asr.with_changes(frontend=CarfacConfig(open_loop=True))
A trained model's weights only make sense for the front end and data they were
trained on, so a trained ASR is never mutated in place.

Folders (all under one root directory, see `Workspace`):
    datasets/   speech corpora: local copies first, Hugging Face downloads land here too
    features/   pre-computed (cached) features, reused whenever the same front end + data come up
    noise/      recorded noise (pub, babble, ...); white noise is generated, never stored
    models/     best models: <name>.pt (weights + config) and <name>_meta.json (everything else)
    results/    per-utterance tables from evaluate() / benchmark()

Status: configs, registry, workspace, naming, metadata, seeding and the constructor
are implemented. Every other method raises NotImplementedError("TODO(Mx): ...")
naming the milestone in claude/asrlib_design.md where it gets filled in.

Comment style: comments sit ABOVE the code they describe and start with a capital
letter. End-of-line comments are short observations (shapes, units, gotchas).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Literal, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Anything transcribe() accepts as audio
AudioLike = Union[str, Path, np.ndarray, torch.Tensor]

# The CARFAC commit verified in M4. Part of every CARFAC cache key,
# so features computed with a different CARFAC version are never mixed with these.
CARFAC_COMMIT = "77d8a91f24d2f5873783dc0ce767d32778b938bc"

# Every front end produces 100 frames per second at 16 kHz
SAMPLE_RATE = 16_000
HOP_LENGTH = 160                                # 10 ms


# ===================================================================================
# registry.py: plug-in points, so other researchers can add parts without editing ASR
# ===================================================================================

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "frontend": {},     # "logmel", "mfcc", "carfac", later "gammatone"
    "model": {},        # "transformer_ctc", later "conformer_ctc", "transformer_aed"
    "corpus": {},       # "librispeech", "local_folder", later multilingual corpora
    "tokenizer": {},    # "char", later "bpe"
}


def register(kind: str, name: str):
    """Class decorator that makes a component buildable by name.

    Example:
        @register("frontend", "carfac")
        class CARFAC(FrontEnd): ...

    Registering a name twice replaces the old entry with a warning. That keeps
    notebooks re-runnable: re-running a cell that defines a class just redefines it.
    """
    def deco(cls):
        if name in _REGISTRY[kind]:
            warnings.warn(f"Re-registering {kind} '{name}' (replacing the previous definition)")
        _REGISTRY[kind][name] = cls
        return cls
    return deco


def build(kind: str, name: str, **kwargs) -> Any:
    """Instantiate a registered component; fails loudly and lists the valid names."""
    if name not in _REGISTRY[kind]:
        raise KeyError(f"Unknown {kind} '{name}'. Registered: {sorted(_REGISTRY[kind])}")
    return _REGISTRY[kind][name](**kwargs)


# =============================================================================
# config.py: the single source of truth for what a system IS
# =============================================================================

def _stable_hash(obj: Any, n: int = 6) -> str:
    """Short hash of a JSON-serializable object, identical on every machine and run.

    Python's built-in hash() of a string changes every time Python starts,
    so it can't name folders. This hashes sorted JSON instead.
    """
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:n]


def _freeze(value: Any) -> Any:
    """Turn lists into tuples (recursively) so configs compare equal after a JSON round trip."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------- front ends

@dataclass(frozen=True)
class FrontEndConfig:
    """Base class for front-end configs. Each subclass sets `kind` (its registry name).

    Fields marked with metadata {"cache": False} only change how cached features
    are post-processed when LOADED, so they're left out of the cache key: switching
    them never forces a re-extraction.
    """
    kind: ClassVar[str] = "base"
    _by_kind: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, **kwargs):
        # Every subclass registers itself, so from_dict() can find it by its `kind`
        super().__init_subclass__(**kwargs)
        FrontEndConfig._by_kind[cls.kind] = cls

    @property
    def name(self) -> str:
        """Short readable name used in model and folder names, e.g. 'logmel80'."""
        return self.kind

    def to_dict(self) -> dict:
        return {"kind": self.kind, **dataclasses.asdict(self)}

    @staticmethod
    def from_dict(d: dict) -> "FrontEndConfig":
        d = dict(d)
        cls = FrontEndConfig._by_kind[d.pop("kind")]
        return cls(**d)

    def cache_dict(self) -> dict:
        """Only the fields that change the cached (raw) features."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if f.metadata.get("cache", True)}

    def cache_key(self) -> str:
        """Folder name under features/: readable name + hash of the cache-relevant fields."""
        return f"{self.name}-{_stable_hash({'kind': self.kind, **self.cache_dict()})}"


@dataclass(frozen=True)
class LogMelConfig(FrontEndConfig):
    """Log-mel filterbank: 25 ms windows, 10 ms hop."""
    kind: ClassVar[str] = "logmel"
    n_mels: int = 80
    win_ms: float = 25.0
    log_eps: float = field(default=1e-6, metadata={"cache": False})

    @property
    def name(self) -> str:
        return f"logmel{self.n_mels}"


@dataclass(frozen=True)
class MFCCConfig(FrontEndConfig):
    """MFCCs computed from a log-mel filterbank."""
    kind: ClassVar[str] = "mfcc"
    n_mfcc: int = 40
    n_mels: int = 80
    win_ms: float = 25.0

    @property
    def name(self) -> str:
        return f"mfcc{self.n_mfcc}"


@dataclass(frozen=True)
class CarfacConfig(FrontEndConfig):
    """CARFAC v2 (JAX port): NAP mean-pooled to 100 frames/s.

    The three parameter groups mirror CARFAC's own design call,
    `design_carfac(fs, car_params=..., agc_params=..., ihc_params=...)`, and use
    CARFAC's own field names. They work the same way too: anything you don't set
    keeps CARFAC's default, and a misspelled field raises an error listing the valid
    names. So `CarParams(erb_per_step=0.25)` in CARFAC becomes
    `CarfacConfig(car_params={"erb_per_step": 0.25})` here.

        CarfacConfig()                                         # defaults: 65 channels at 16 kHz
        CarfacConfig(car_params={"erb_per_step": 0.25})        # 131 channels
        CarfacConfig(car_params={"velocity_scale": 0.2})       # stronger fast compression
        CarfacConfig(ihc_params={"ihc_style": "one_cap"})      # v1 inner hair cell
        CarfacConfig(open_loop=True)                           # no AGC ("carfac-noagc")
        CarfacConfig(post="cuberoot")                          # same cache, different compression

    The one default that differs from CARFAC's: agc decimation (6, 2, 2, 2) instead of (8, 2, 2, 2).
    It keeps the AGC spatial smoother at 3 taps at 16 kHz, which is the only case the JAX port
    gets right (see carfac_setup_notes). The front end refuses designs that need 5 taps.

    Input level is NOT set here (notebook 06 had `level_dbfs` here). In asrlib the Condition sets
    the speech level once for every front end. A second calibration inside CARFAC would undo the
    level-sweep experiments by rescaling every input back to -26 dBFS.
    """
    kind: ClassVar[str] = "carfac"

    # Cochlea design, grouped and named as in CARFAC
    car_params: dict = field(default_factory=dict)
    agc_params: dict = field(default_factory=lambda: {"decimation": (6, 2, 2, 2)})
    ihc_params: dict = field(default_factory=dict)

    # Run option, same name as CARFAC's run_segment(open_loop=...)
    open_loop: bool = False

    # Applied when features are LOADED, so they're not part of the cache key
    post: Literal["log", "cuberoot", "none"] = field(default="log", metadata={"cache": False})
    log_eps: float = field(default=1e-3, metadata={"cache": False})

    def __post_init__(self):
        # JSON turns tuples into lists; convert back so a reloaded config equals the original
        for group in ("car_params", "agc_params", "ihc_params"):
            object.__setattr__(self, group, _freeze(getattr(self, group)))

    @property
    def name(self) -> str:
        return f"carfac-{'noagc' if self.open_loop else 'agc'}-{self.post}"

    def cache_key(self) -> str:
        # The CARFAC commit is part of the key: a new CARFAC version means new features
        payload = {"kind": self.kind, **self.cache_dict(), "carfac_commit": CARFAC_COMMIT}
        return f"carfac-{'noagc' if self.open_loop else 'agc'}-{_stable_hash(payload)}"


# ---------------------------------------------------------------------------- the rest

@dataclass(frozen=True)
class ModelConfig:
    """Acoustic model architecture. `arch` is a registry name; the rest are its kwargs."""
    arch: str = "transformer_ctc"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    subsample_channels: int = 128
    subsample_factor: int = 4


@dataclass(frozen=True)
class DataConfig:
    """Which speech corpus and splits to train, validate and test on.

    `source` decides where the audio comes from:
        "auto"   use datasets/<corpus>/ if it's there, otherwise download from Hugging Face
        "local"  only datasets/<corpus>/ (error if missing)
        "hf"     always go through Hugging Face (downloads are cached in datasets/hf_cache/)
    """
    corpus: str = "librispeech"
    train_split: str = "train.100"
    n_train_shards: int | None = None       # None = all 14 shards (100 h); 4 ≈ 28 h
    val_split: str = "validation"
    test_split: str = "test"
    source: Literal["auto", "local", "hf"] = "auto"
    max_train_seconds: float | None = 20.0  # drops long utterances from TRAINING only

    @property
    def tag(self) -> str:
        """Short dataset label for model names, e.g. 'librispeech-train.100' or '...-4sh'."""
        shards = f"-{self.n_train_shards}sh" if self.n_train_shards else ""
        return f"{self.corpus}-{self.train_split}{shards}"


@dataclass(frozen=True)
class Condition:
    """An acoustic condition: how a clean waveform is degraded before the front end.

    Every utterance is first scaled to `speech_level_dbfs` (RMS). This is the only
    place the level is set, for every front end. Then noise is scaled to hit `snr_db`
    relative to the SPEECH. The mixture is never renormalized.

    noise:  None (clean), "white" (generated), or a folder name under noise/ ("pub", "babble", ...)
    split:  which part of each noise recording to use; training and test noise never overlap
    """
    noise: str | None = None
    snr_db: float | None = None
    speech_level_dbfs: float = -26.0
    split: Literal["train", "test"] = "test"

    def __post_init__(self):
        if (self.noise is None) != (self.snr_db is None):
            raise ValueError("noise and snr_db must be both set or both None")

    @property
    def name(self) -> str:
        """'clean', 'pub@5dB', 'clean@-46dBFS', 'babble@0dB@-6dBFS', ..."""
        base = "clean" if self.noise is None else f"{self.noise}@{self.snr_db:g}dB"
        if self.speech_level_dbfs != -26.0:
            base += f"@{self.speech_level_dbfs:g}dBFS"
        return base

    def apply(self, wave: torch.Tensor, utt_id: str, bank: "NoiseBank | None" = None,
              variant: int = 0) -> torch.Tensor:
        """Degrade one waveform, the same way every time.

        Steps:
          1. Scale speech RMS to speech_level_dbfs
          2. Seed a generator from (self.name, utt_id, variant): same noise for every front end
          3. noise = bank.segment(self.noise, self.split, len(wave), rng)
          4. Scale noise so that 10*log10(P_speech / P_noise) == snr_db
          5. Return speech + noise (same length, float32, no clipping)
        """
        raise NotImplementedError("TODO(M2): level calibration + seeded noise mixing")


CLEAN = Condition()


@dataclass(frozen=True)
class TrainConfig:
    """The training recipe. It's part of the model's identity: same config, same model."""
    epochs: int = 40
    peak_lr: float = 1e-3
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_frames_per_batch: int = 40_000      # length-bucketed batches, ≈ 400 s of audio
    grad_clip: float = 1.0
    amp: Literal["auto", "bf16", "fp16", "off"] = "auto"
    specaugment: dict = field(default_factory=lambda: {
        "freq_masks": 2, "freq_width": 15, "time_masks": 2, "time_width": 40,
        # Open item from notebook 06: scale freq_width by n_feats / 80, so 65 CARFAC
        # channels lose the same FRACTION of the axis as 80 mel bins
        "scale_freq_to_n_feats": True})
    train_conditions: tuple[Condition, ...] = (CLEAN,)   # Phase 2: add noisy training conditions
    n_noisy_variants: int = 0               # Phase 2: K cached noisy copies per utterance
    early_stop_patience: int | None = 8
    num_workers: int = 4

    @property
    def noise_tag(self) -> str:
        """'clean' or e.g. 'multi-babble+pub', for model names."""
        kinds = sorted({c.noise for c in self.train_conditions if c.noise})
        return "clean" if not kinds else "multi-" + "+".join(kinds)


@dataclass(frozen=True)
class ASRConfig:
    """Everything that defines a trained ASR system.

    The FRONT END and SEED are the variables being compared. Everything else is
    "controlled" and must match before two systems are compared (see
    `controlled_hash` and `assert_comparable`).
    """
    frontend: FrontEndConfig = field(default_factory=LogMelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    norm: Literal["utt_cmvn", "none"] = "utt_cmvn"
    tokenizer: str = "char"
    seed: int = 0

    # ---- serialization --------------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # asdict() drops the front end's `kind` (a ClassVar), so add it back
        d["frontend"] = self.frontend.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ASRConfig":
        # JSON loses types: rebuild each nested dataclass (and the tuple of Conditions) by hand
        t = dict(d["train"])
        t["train_conditions"] = tuple(Condition(**c) for c in t["train_conditions"])
        return cls(frontend=FrontEndConfig.from_dict(d["frontend"]),
                   model=ModelConfig(**d["model"]),
                   data=DataConfig(**d["data"]),
                   train=TrainConfig(**t),
                   norm=d["norm"], tokenizer=d["tokenizer"], seed=d["seed"])

    # ---- identity ---------------------------------------------------------------
    def hash(self) -> str:
        """Hash of the whole config: same hash means the same experiment."""
        return _stable_hash(self.to_dict())

    def controlled_hash(self) -> str:
        """Hash of everything EXCEPT the front end and seed. Two systems in one
        comparison must share this hash."""
        d = self.to_dict()
        d.pop("frontend")
        d.pop("seed")
        return _stable_hash(d)

    def replace(self, **changes) -> "ASRConfig":
        """New config with some top-level fields swapped (frontend=..., data=..., seed=...)."""
        return dataclasses.replace(self, **changes)


# =============================================================================
# paths.py: one root folder holds everything the library reads or writes
# =============================================================================

class Workspace:
    """The project folder, with these subfolders:

        <root>/datasets/   speech corpora (local copies, plus the Hugging Face cache in hf_cache/)
        <root>/features/   cached features: <frontend cache key>/<corpus>/<split>/<condition>/
        <root>/noise/      noise recordings: noise/pub/*.wav, noise/babble/*.wav, noise/splits.json
        <root>/models/     <name>.pt + <name>_meta.json for every best model
        <root>/results/    <name>/per_utt/<condition>.csv from evaluate()/benchmark()

    The root is chosen in this order: the `root` argument, then the ASRLIB_ROOT
    environment variable, then the current working directory (where your notebook
    or script runs). It's deliberately NOT the folder this .py file lives in: once
    asrlib is pip-installed from GitHub, that folder is inside site-packages and
    gets wiped on every reinstall.

        Laptop: run from your ASR/ folder (or set ASRLIB_ROOT=~/ASR)
        Colab:  Workspace("/content/drive/MyDrive/thesis_asr")
        Quest:  export ASRLIB_ROOT=/projects/<allocation>/thesis_asr
    """
    FOLDERS = ("datasets", "features", "noise", "models", "results")

    def __init__(self, root: str | Path | None = None, create: bool = True):
        # Resolve the root: explicit argument > environment variable > current directory
        chosen = root or os.environ.get("ASRLIB_ROOT") or Path.cwd()
        self.root = Path(chosen).expanduser().resolve()

        # Create any missing subfolders, so later code can assume they exist
        if create:
            for name in self.FOLDERS:
                (self.root / name).mkdir(parents=True, exist_ok=True)

    # One property per folder, so paths are never typed by hand elsewhere
    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def noise(self) -> Path:
        return self.root / "noise"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def model_paths(self, name: str) -> tuple[Path, Path]:
        """(<root>/models/<name>.pt, <root>/models/<name>_meta.json)."""
        return self.models / f"{name}.pt", self.models / f"{name}_meta.json"

    def find_models(self, config_hash: str | None = None) -> list[Path]:
        """Meta files in models/, newest first; optionally only those trained from one config."""
        pattern = f"*_{config_hash}_meta.json" if config_hash else "*_meta.json"
        return sorted(self.models.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    def summary(self) -> str:
        """What's in each folder: a quick check that the root is the one you meant."""
        lines = [f"Workspace: {self.root}"]
        for name in self.FOLDERS:
            folder = self.root / name
            n = len(list(folder.iterdir())) if folder.exists() else 0
            lines.append(f"  {name + '/':10s} {n} item(s)" if folder.exists() else f"  {name + '/':10s} missing")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"


# =============================================================================
# Component interfaces (each becomes its own module later)
# =============================================================================

@dataclass
class Utterance:
    """One item from any corpus."""
    utt_id: str
    wave: torch.Tensor                      # f32[N], mono
    sample_rate: int
    text: str


# ---------------------------------------------------------------------------- frontends.py

class FrontEnd(ABC):
    """Waveform → (T, F) features at a fixed 100 frames/s.

    Contract that every front end must satisfy (and that tests check):
      * input:  f32[N] mono at 16 kHz, already level-calibrated by the Condition
      * output: f32[T, F] with F == n_feats and T == N // 160 + 1 (±1)
      * channels ordered low → high frequency (CARFAC's channel 0 is the HIGHEST CF: flip it)
      * extract() is deterministic and is the only expensive step; post() is cheap
    """
    cfg: FrontEndConfig
    n_feats: int
    expensive: bool = False                 # True → features are cached under features/

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
    """Log-mel filterbank (torchaudio): n_fft=400 (25 ms), hop=160 (10 ms)."""

    def __init__(self, cfg: LogMelConfig):
        self.cfg = cfg
        self.n_feats = cfg.n_mels
        # TODO(M1): self.melspec = torchaudio.transforms.MelSpectrogram(...)

    def extract(self, wave):
        raise NotImplementedError("TODO(M1): melspec(wave).T")    # (n_mels, T) → (T, n_mels)

    def post(self, raw):
        raise NotImplementedError("TODO(M1): torch.log(raw + cfg.log_eps)")


@register("frontend", "mfcc")
class MFCC(FrontEnd):
    """MFCCs (torchaudio). Already log + DCT, so post() is the identity."""

    def __init__(self, cfg: MFCCConfig):
        self.cfg = cfg
        self.n_feats = cfg.n_mfcc
        # TODO(M1): self.mfcc = torchaudio.transforms.MFCC(...)

    def extract(self, wave):
        raise NotImplementedError("TODO(M1): mfcc(wave).T")       # (n_mfcc, T) → (T, n_mfcc)


@register("frontend", "carfac")
class CARFAC(FrontEnd):
    """CARFAC v2 NAP (JAX port), mean-pooled to 100 frames/s. Ported from notebook 06's CarfacFrontEnd.

    Design happens once, in __init__:
        hypers, weights, state0 = cj.design_and_init_carfac(self._design_params())
    and every utterance starts from the same `state0`. That's safe because JAX is
    functional. (The NumPy port mutates its cfp in place, so there you'd need a fresh
    deepcopy per utterance.)

    What extract() returns is the firing RATE: NAP + rest_output, always ≥ 0, with
    channels flipped to low → high CF. post() then applies log / cube root / nothing.

    `jax` and `carfac` are imported inside __init__, so the rest of the library works
    on machines without them.
    """
    expensive = True
    CHUNK = 16_000                          # 1 s chunks inside a lax.scan; pooling happens per chunk

    def __init__(self, cfg: CarfacConfig):
        self.cfg = cfg
        # TODO(M4): import jax, jax.numpy as jnp, carfac.jax.carfac as cj (lazily, here)
        # TODO(M4): self.hypers, self.weights, self.state0 = cj.design_and_init_carfac(self._design_params())
        # TODO(M4): refuse 5-tap AGC smoothers: [a.agc_spatial_n_taps for a in hypers.ears[0].agc] must all be 3
        # TODO(M4): self.n_feats = int(self.hypers.ears[0].car.n_ch)     # 65 at defaults
        # TODO(M4): self.cf_hz = pole_freqs[::-1]                        # low → high, matches the flip
        # TODO(M4): self.rest_output = float(weights.ears[0].ihc.rest_output[0])   # ≈ 0.812 for two_cap
        # TODO(M4): self._batched = jax.jit(jax.vmap(self._one))         # compiles once per (batch, length)
        self.n_feats = 65                   # placeholder until __init__ reads it from the design
        self.cf_hz: np.ndarray | None = None
        self.rest_output: float | None = None

    def _design_params(self):
        """CarfacConfig → cj.CarfacDesignParameters(fs=16000), one group at a time.

        For each (group, overrides) in car / agc / ihc:
            target = params.ears[0].<group>
            For each key, value: if target lacks the field, raise AttributeError listing
            dataclasses.fields(target); otherwise setattr(target, key, value).
        Same names, same behaviour as passing CarParams(...) etc. to CARFAC itself.
        """
        raise NotImplementedError("TODO(M4)")

    def _one(self, wave):
        """wave f32[L] (L a multiple of CHUNK) → pooled NAP f32[L // 160, n_ch].

        lax.scan over 1 s chunks: run_segment(chunk, hypers, weights, state,
        open_loop=cfg.open_loop), then mean over each 160-sample window.
        """
        raise NotImplementedError("TODO(M4)")

    def extract_batch(self, waves):
        """Zero-pad to whole 1 s chunks (+ one hop), run self._batched, then for each utterance:
        keep N // 160 + 1 frames, flip channels, add rest_output.
        CARFAC is causal, so the padding can't change earlier frames (verified ≤ 1e-5)."""
        raise NotImplementedError("TODO(M4)")

    def extract(self, wave):
        return self.extract_batch([wave])[0]

    def post(self, raw):
        """'log' → log(rate.clamp_min(0) + log_eps), 'cuberoot' → rate.clamp_min(0) ** (1/3), 'none' → rate."""
        raise NotImplementedError("TODO(M4)")


# ---------------------------------------------------------------------------- features pipeline

class Normalizer:
    """Feature normalization after post(). 'utt_cmvn' = per-utterance, per-channel mean/variance.

    It's stateless, so there's nothing to save. A future 'global' mode would store its
    statistics in state_dict(), and load() would restore them.
    """

    def __init__(self, mode: Literal["utt_cmvn", "none"] = "utt_cmvn"):
        self.mode = mode

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO(M1): (x - x.mean(0)) / (x.std(0) + 1e-5) when mode == 'utt_cmvn'")

    def state_dict(self) -> dict:
        return {"mode": self.mode}


class Featurizer:
    """THE feature pipeline. Training, evaluation and transcribe() all call this one
    object, so inference computes features exactly the way training did.

        wave ─► resample to 16 kHz ─► condition.apply (level + noise)
             ─► frontend.extract ─► [cache] ─► frontend.post ─► normalizer ─► [SpecAugment if train]
    """

    def __init__(self, frontend: FrontEnd, normalizer: Normalizer,
                 specaugment: dict | None, train_level_dbfs: float = -26.0):
        self.frontend = frontend
        self.normalizer = normalizer
        self.specaugment_cfg = specaugment
        # The level every training utterance was scaled to; transcribe() matches it
        self.train_level_dbfs = train_level_dbfs

    def __call__(self, wave: torch.Tensor, *, condition: Condition | None = None,
                 utt_id: str = "adhoc", bank: "NoiseBank | None" = None,
                 train: bool = False, calibrate_level: bool = True) -> torch.Tensor:
        """Waveform (16 kHz) → model-ready f32[T, F].

        condition given:  condition.apply(...) sets the level and adds noise.
        condition None:   only level calibration to train_level_dbfs (when calibrate_level),
                          so an arbitrary .wav reaches CARFAC at the level it was trained on.
        train=True adds SpecAugment; it's never applied at evaluation.
        """
        raise NotImplementedError("TODO(M1/M2)")

    def from_raw(self, raw: torch.Tensor, train: bool = False) -> torch.Tensor:
        """The same pipeline starting from cached extract() output: post → normalize → [SpecAugment]."""
        raise NotImplementedError("TODO(M3)")

    def spec_augment(self, feats: torch.Tensor) -> torch.Tensor:
        """Frequency and time masks. With scale_freq_to_n_feats, width = freq_width * n_feats / 80."""
        raise NotImplementedError("TODO(M1)")


# ---------------------------------------------------------------------------- tokenizer.py

class Tokenizer(ABC):
    """Text ↔ ids. Id 0 is the CTC blank (and the padding id)."""
    blank_id: int = 0

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def normalize(self, text: str) -> str: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> str: ...

    @abstractmethod
    def state_dict(self) -> dict: ...


@register("tokenizer", "char")
class CharTokenizer(Tokenizer):
    """Lower-case letters, space and apostrophe, plus <blank/pad>, <bos>, <eos>: 31 symbols, as in the notebook."""
    SPECIALS = ("<blank>", "<bos>", "<eos>")
    CHARS = " 'abcdefghijklmnopqrstuvwxyz"

    def __init__(self):
        self.vocab = list(self.SPECIALS) + list(self.CHARS)
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def normalize(self, text):
        raise NotImplementedError("TODO(M0): lower-case, keep known chars, collapse spaces")

    def encode(self, text):
        raise NotImplementedError("TODO(M0)")

    def decode(self, ids):
        raise NotImplementedError("TODO(M0): drop special ids")

    def state_dict(self):
        return {"kind": "char", "vocab": self.vocab}


# ---------------------------------------------------------------------------- models.py

class AcousticModel(nn.Module, ABC):
    """Features → encoder states → (loss | token ids).

    Every architecture (Transformer-CTC now; Conformer or AED later) has the same
    three methods, so Trainer and ASR never change when the architecture does.
    """

    @abstractmethod
    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, F), (B,) → model output (CTC: log-probs (B, T', V)) and output lengths (B,)."""

    @abstractmethod
    def loss(self, feats, feat_lens, tokens, tok_lens) -> torch.Tensor:
        """Scalar training loss for one batch. CTC loss is computed in fp32."""

    @abstractmethod
    def decode(self, feats, feat_lens) -> list[list[int]]:
        """Token ids per utterance (CTC: greedy argmax, merge repeats, drop blanks)."""


@register("model", "transformer_ctc")
class TransformerCTC(AcousticModel):
    """ConvSubsample(×4) → sinusoidal positions → Pre-LN Transformer → Linear → log-softmax.

    The notebook's model, with n_feats and vocab_size passed in instead of read from globals.
    `arch` is accepted and ignored so _build_model can pass every ModelConfig field at once.
    """

    def __init__(self, n_feats: int, vocab_size: int, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, dropout=0.1, subsample_channels=128, subsample_factor=4, arch=None):
        super().__init__()
        self.n_feats = n_feats
        self.vocab_size = vocab_size
        # TODO(M1): build ConvSubsample, PositionalEncoding, TransformerEncoder, output Linear

    def forward(self, feats, feat_lens):
        raise NotImplementedError("TODO(M1)")

    def loss(self, feats, feat_lens, tokens, tok_lens):
        raise NotImplementedError("TODO(M1)")

    def decode(self, feats, feat_lens):
        raise NotImplementedError("TODO(M1)")


# ---------------------------------------------------------------------------- data.py

class SpeechCorpus(ABC):
    """Any source of (utt_id, wave, sr, text)."""
    name: str                               # used in feature cache paths, e.g. "librispeech"
    split: str

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, i: int) -> Utterance: ...

    @abstractmethod
    def durations(self) -> np.ndarray:
        """Samples per utterance, read without decoding audio (for length bucketing and cache offsets)."""

    @abstractmethod
    def utt_ids(self) -> list[str]: ...


@register("corpus", "librispeech")
class LibriSpeech(SpeechCorpus):
    """LibriSpeech, from datasets/ first and Hugging Face otherwise.

    Lookup order (source="auto"):
      1. <root>/datasets/librispeech/<split>/*.parquet   (copies you placed there yourself)
      2. load_dataset("openslr/librispeech_asr", data_files=..., cache_dir=<root>/datasets/hf_cache)
         Only the requested parquet shards are downloaded, then reused from the cache next time.

    Audio stays as raw FLAC bytes (Audio(decode=False)) and is decoded with soundfile,
    as in the notebook, so nothing depends on TorchCodec.
    """
    HF_REPO = "openslr/librispeech_asr"

    def __init__(self, split: str, datasets_dir: Path, n_shards: int | None = None,
                 source: Literal["auto", "local", "hf"] = "auto"):
        self.name = "librispeech"
        self.split = split
        self.datasets_dir = Path(datasets_dir)
        self.n_shards = n_shards
        self.source = source
        # TODO(M1): self.rows = self._load()

    def local_dir(self) -> Path:
        return self.datasets_dir / self.name / self.split

    def _load(self):
        """Pick local parquet files or Hugging Face per `source`; return an HF Dataset with raw audio bytes."""
        raise NotImplementedError("TODO(M1)")

    def __len__(self):
        raise NotImplementedError("TODO(M1)")

    def __getitem__(self, i):
        raise NotImplementedError("TODO(M1): sf.read(io.BytesIO(row['audio']['bytes'])) → Utterance")

    def durations(self):
        raise NotImplementedError("TODO(M1): sf.info(...).frames per row")

    def utt_ids(self):
        raise NotImplementedError("TODO(M1): row['id']")


@register("corpus", "local_folder")
class LocalFolderCorpus(SpeechCorpus):
    """Your own recordings: <root>/datasets/<name>/<split>/ with audio files plus a manifest.csv
    (columns: path, text). Lets you test on anything you record without touching Hugging Face."""

    def __init__(self, split: str, datasets_dir: Path, name: str = "my_recordings", **_):
        self.name = name
        self.split = split
        self.folder = Path(datasets_dir) / name / split

    def __len__(self):
        raise NotImplementedError("TODO(M2)")

    def __getitem__(self, i):
        raise NotImplementedError("TODO(M2)")

    def durations(self):
        raise NotImplementedError("TODO(M2)")

    def utt_ids(self):
        raise NotImplementedError("TODO(M2): file stems")


class FeatureStore:
    """Cached features under <root>/features/, in notebook 06's format.

    One folder per (front end, corpus, split, condition[, variant]):
        features/<frontend cache key>/<corpus>/<split>_<n_utts>/<condition>[_v<k>]/
            rate.npy     float16 memmap, all utterances stacked: (total frames, n_feats)
            index.npz    utt_ids, offsets, n_frames
            meta.json    front-end config, CARFAC commit, n_feats, cf_hz, hours, seconds taken, machine
            DONE         written last; a folder without it is incomplete and gets rebuilt

    Utterances are processed sorted by length, in batches, so padding stays low and
    JAX compiles one program per (batch, seconds) shape.
    """

    def __init__(self, features_dir: Path):
        self.root = Path(features_dir)

    def location(self, frontend: FrontEnd, corpus: SpeechCorpus, condition: Condition,
                 variant: int | None = None) -> Path:
        cond = condition.name + (f"_v{variant}" if variant is not None else "")
        return self.root / frontend.cfg.cache_key() / corpus.name / f"{corpus.split}_{len(corpus)}" / cond

    def has(self, *args, **kwargs) -> bool:
        return (self.location(*args, **kwargs) / "DONE").exists()

    def build(self, frontend: FrontEnd, corpus: SpeechCorpus, condition: Condition,
              bank: "NoiseBank | None", variant: int | None = None, batch_size: int = 32) -> Path:
        """Compute and write one cache folder (see the class docstring). Returns its path."""
        raise NotImplementedError("TODO(M3): port build_split_cache from notebook 06")

    def open(self, location: Path) -> "CachedFeatures":
        raise NotImplementedError("TODO(M3)")


class CachedFeatures:
    """Read-only view of one cache folder: cached[i] → raw f32[T, F] for utterance i."""

    def __init__(self, location: Path):
        self.location = Path(location)
        # TODO(M3): memmap rate.npy, load index.npz

    def __len__(self) -> int:
        raise NotImplementedError("TODO(M3)")

    def __getitem__(self, i: int) -> torch.Tensor:
        raise NotImplementedError("TODO(M3): rate[offsets[i] : offsets[i] + n_frames[i]]")


class FeatureDataset(Dataset):
    """torch Dataset of (feats[T, F], tokens[U], utt_id) for one split.

    Expensive front ends read from the FeatureStore (building the cache first if
    needed); cheap ones compute features on the fly through the Featurizer.
    Several conditions or variants (Phase 2) are concatenated into one dataset.
    """

    def __init__(self, corpus: SpeechCorpus, featurizer: Featurizer, tokenizer: Tokenizer,
                 conditions: Sequence[Condition], store: FeatureStore, bank: "NoiseBank",
                 train: bool, n_variants: int = 0):
        raise NotImplementedError("TODO(M1/M3)")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def collate(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """List of (feats, tokens, utt_id) → feats (B, T_max, F), feat_lens (B,), tokens (B, U_max), tok_lens (B,), utt_ids."""
    raise NotImplementedError("TODO(M1)")


class LengthBucketSampler:
    """Yields batches of similar-length utterances, each under max_frames_per_batch frames."""

    def __init__(self, n_frames: Sequence[int], max_frames: int, shuffle: bool, seed: int):
        raise NotImplementedError("TODO(M1)")


# ---------------------------------------------------------------------------- conditions.py

class NoiseBank:
    """Noise audio for the Conditions.

    "white" is generated from the seeded generator and never stored.
    Every other kind is a folder of recordings you provide (or download once):

        <root>/noise/pub/*.wav        e.g. DEMAND PRESTO + PCAFETER, channel 1
        <root>/noise/babble/*.wav     e.g. 8-talker mixes from LibriSpeech train-other
        <root>/noise/splits.json      written on first use: which seconds of each file are train vs test

    The 60/40 train/test split is decided once and saved, so it never changes between
    sessions or machines. A missing folder raises an error that says where to put the files.
    """
    TRAIN_FRACTION = 0.6

    def __init__(self, noise_dir: Path, sample_rate: int = SAMPLE_RATE):
        self.root = Path(noise_dir)
        self.sample_rate = sample_rate
        # Loaded lazily, one kind at a time: {kind: [f32 arrays]}
        self._audio: dict[str, list[np.ndarray]] = {}

    def available(self) -> list[str]:
        """Kinds you can use right now: 'white' plus every non-empty subfolder of noise/."""
        kinds = ["white"]
        if self.root.exists():
            kinds += sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.iterdir()))
        return kinds

    def segment(self, kind: str, split: Literal["train", "test"], n_samples: int,
                rng: np.random.Generator) -> torch.Tensor:
        """n_samples of noise from the `split` part of `kind`, at a random (seeded) position.
        'white' → rng.standard_normal(n_samples). Shorter recordings are looped."""
        raise NotImplementedError("TODO(M2)")

    def _load_kind(self, kind: str) -> list[np.ndarray]:
        """Read noise/<kind>/* → mono f32 at 16 kHz; create or read splits.json."""
        raise NotImplementedError("TODO(M2)")

    def download_demand(self, environments: Sequence[str] = ("PRESTO", "PCAFETER"), kind: str = "pub") -> None:
        """Optional helper: fetch DEMAND 16 kHz zips from Zenodo into noise/<kind>/ (channel 1 only). Idempotent."""
        raise NotImplementedError("TODO(M2)")


# ---------------------------------------------------------------------------- metrics.py

@dataclass
class EvalResult:
    """Scores for one condition, plus per-utterance rows for significance tests."""
    condition: str
    wer: float
    cer: float
    loss: float
    n_utts: int
    per_utt: Any = None                     # pandas DataFrame[utt_id, ref, hyp, w_err, w_ref, c_err, c_ref]
    wer_ci: tuple[float, float] | None = None

    def row(self) -> dict:
        """Flat dict for a results table (everything except the per-utterance frame)."""
        return {k: v for k, v in dataclasses.asdict(self).items() if k != "per_utt"}


def error_rate(hyps: Sequence[str], refs: Sequence[str], unit: Literal["word", "char"] = "word") -> float:
    """Corpus-level WER/CER: total edit distance / total reference length."""
    raise NotImplementedError("TODO(M0): torchaudio.functional.edit_distance, as in the notebook")


def bootstrap_ci(result: EvalResult, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """WER confidence interval from resampling utterances."""
    raise NotImplementedError("TODO(M2)")


def paired_bootstrap(a: EvalResult, b: EvalResult, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """(WER_a - WER_b, p-value) on the same utterances and condition."""
    raise NotImplementedError("TODO(M2)")


@dataclass
class History:
    """One dict per epoch: train_loss, val_loss, val_wer, val_cer, lr, seconds."""
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_wer: float = float("inf")


# ---------------------------------------------------------------------------- train.py

class Trainer:
    """Optimizer, schedule, mixed precision and the epoch loop: everything only training needs.

    Built by ASR.fit() and thrown away afterwards, so a loaded model never carries an optimizer.
      * AdamW with decay / no-decay parameter groups (no decay on biases and norms)
      * linear warmup → cosine decay, stepped per batch; progress clamped at 1, so the LR can't
        climb back up (the bug found in DL_review's mel run)
      * AMP: bf16 on A100 / L4, fp16 + GradScaler on T4, off on CPU / MPS
      * gradient clipping, early stopping on val WER
    """

    def __init__(self, asr: "ASR", train_loader, val_loader):
        self.asr = asr
        self.train_loader = train_loader
        self.val_loader = val_loader
        # TODO(M1): optimizer, scheduler, scaler, amp dtype from asr.cfg.train

    def train_epoch(self) -> float:
        """One pass over train_loader; returns the mean loss."""
        raise NotImplementedError("TODO(M1)")

    def validate(self) -> dict:
        """Loss, WER, CER on val_loader (greedy decoding, no SpecAugment)."""
        raise NotImplementedError("TODO(M1)")

    def state_dict(self) -> dict:
        """Everything needed to resume: model, optimizer, scheduler, scaler, epoch, RNG states, history."""
        raise NotImplementedError("TODO(M1)")

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError("TODO(M1)")


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA).

    Called once in ASR.__init__, before any weights are built, so the same seed
    gives the same initial weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str | torch.device | None = None) -> torch.device:
    """Explicit choice, else CUDA (Colab/Quest), else MPS (your Mac's GPU), else CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_info() -> dict:
    """Where and with what code a model was trained; goes into every _meta.json."""
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": getattr(torch, "__version__", "?")}

    # Optional libraries: record their versions only if they're installed
    for lib in ("torchaudio", "jax", "carfac", "datasets"):
        try:
            info[lib] = getattr(__import__(lib), "__version__", "installed")
        except ImportError:
            info[lib] = None

    # Which machine: Colab and Slurm (Quest) both leave recognizable traces
    if "google.colab" in sys.modules:
        info["machine"] = "colab"
    elif "SLURM_JOB_ID" in os.environ:
        info["machine"] = f"quest (slurm job {os.environ['SLURM_JOB_ID']})"
    else:
        info["machine"] = platform.node()
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    # The exact asrlib code: the git commit of the folder this file lives in, if it's a repo
    try:
        info["asrlib_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["asrlib_commit"] = None
    info["carfac_commit"] = CARFAC_COMMIT
    return info


def assert_comparable(*systems: "ASR") -> None:
    """Raise if the systems differ in anything other than front end and seed.
    Call this before putting their numbers in the same table."""
    hashes = {s.cfg.controlled_hash() for s in systems}
    if len(hashes) > 1:
        raise ValueError(f"Systems differ in controlled settings: {hashes}. "
                         "Diff their cfg.to_dict() to see what changed.")


# =============================================================================
# asr.py: the facade
# =============================================================================

class ASR:
    """A complete, self-contained speech recognizer: front end + normalizer +
    tokenizer + acoustic model, built from one ASRConfig.

    Lifecycle
    ---------
        asr = ASR(cfg, root="~/ASR")            # untrained; weights fixed by cfg.seed
        asr.precompute_features()               # optional: fill features/ first (e.g. a Quest CPU job)
        asr.sanity_check()                      # shapes + overfit one batch
        asr.fit()                               # best model → models/<name>.pt + <name>_meta.json
        asr.transcribe("clip.wav")              # → "the cat sat ..."
        asr.evaluate(condition=Condition("pub", 5))
        asr.benchmark([CLEAN, Condition("white", 10), ...])
        ASR.load("<name>")                      # from models/, or a path to a .pt

    Owns:              cfg, workspace, frontend, normalizer, featurizer, tokenizer, model, device, history
    Builds as needed:  corpora, feature store, noise bank, datasets, loaders, Trainer
    Never:             reads globals, mutates cfg, or changes its front end after construction
    """

    # ---- construction ------------------------------------------------------------
    def __init__(self, cfg: ASRConfig | None = None, root: str | Path | None = None,
                 device: str | torch.device | None = None):
        """Build an untrained system from `cfg`.

        Order matters: seed → front end (sets n_feats) → normalizer → tokenizer
        (sets vocab_size) → model(n_feats, vocab_size) → featurizer → move to device.
        """
        self.cfg: ASRConfig = cfg if cfg is not None else ASRConfig()
        self.workspace = Workspace(root)
        self.device: torch.device = pick_device(device)
        self.history = History()
        self.is_trained = False
        # Set by fit() or load(): the file stem in models/
        self.name: str | None = None

        # Build the components in dependency order
        seed_everything(self.cfg.seed)
        self.frontend: FrontEnd = self._build_frontend()
        self.normalizer: Normalizer = self._build_normalizer()
        self.tokenizer: Tokenizer = self._build_tokenizer()
        self.model: AcousticModel = self._build_model().to(self.device)
        self.featurizer: Featurizer = self._build_featurizer()

    def _build_frontend(self) -> FrontEnd:
        """build('frontend', cfg.frontend.kind, cfg=cfg.frontend)."""
        raise NotImplementedError("TODO(M1): ASR._build_frontend, via build('frontend', ...)")

    def _build_normalizer(self) -> Normalizer:
        raise NotImplementedError("TODO(M1): Normalizer(self.cfg.norm)")

    def _build_tokenizer(self) -> Tokenizer:
        raise NotImplementedError("TODO(M1): build('tokenizer', self.cfg.tokenizer)")

    def _build_model(self) -> AcousticModel:
        """build('model', cfg.model.arch, n_feats=frontend.n_feats, vocab_size=tokenizer.vocab_size,
        **asdict(cfg.model)). n_feats comes from the front end, never from a constant."""
        raise NotImplementedError("TODO(M1)")

    def _build_featurizer(self) -> Featurizer:
        """Featurizer(frontend, normalizer, cfg.train.specaugment, train_level_dbfs=<level of train conditions>)."""
        raise NotImplementedError("TODO(M1)")

    def with_changes(self, **changes) -> "ASR":
        """A NEW untrained ASR whose config differs only in `changes`, in the same workspace.

            asr.with_changes(frontend=CarfacConfig(open_loop=True))
            asr.with_changes(seed=2)
            asr.with_changes(data=dataclasses.replace(asr.cfg.data, n_train_shards=4))
            asr.with_changes(train=dataclasses.replace(asr.cfg.train,
                             train_conditions=(CLEAN, Condition("pub", 5, split="train"))))
        """
        return ASR(self.cfg.replace(**changes), root=self.workspace.root, device=self.device)

    # ---- naming and metadata --------------------------------------------------------
    def default_name(self, date: dt.date | None = None) -> str:
        """<date>_<frontend>_<dataset>_<noise>_s<seed>_<config hash>,
        e.g. 2026-10-27_carfac-agc-log_librispeech-train.100_clean_s1_a1b2c3.

        The date says when training started. The hash makes the name unique to the
        config: fit() uses it to find and resume an unfinished run, even from another day.
        """
        date = date or dt.date.today()
        c = self.cfg
        return f"{date.isoformat()}_{c.frontend.name}_{c.data.tag}_{c.train.noise_tag}_s{c.seed}_{c.hash()}"

    def meta(self) -> dict:
        """Everything about this model except the weights: written to models/<name>_meta.json."""
        c = self.cfg
        return {
            "name": self.name,
            "status": "trained" if self.is_trained else "untrained",
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            "config_hash": c.hash(),
            "controlled_hash": c.controlled_hash(),
            "seed": c.seed,
            "frontend": {**c.frontend.to_dict(), "name": c.frontend.name,
                         "cache_key": c.frontend.cache_key(), "n_feats": self.frontend.n_feats},
            "dataset": {**dataclasses.asdict(c.data), "tag": c.data.tag},
            "noise": {"used_in_training": c.train.noise_tag != "clean",
                      "train_conditions": [cond.name for cond in c.train.train_conditions],
                      "n_noisy_variants": c.train.n_noisy_variants},
            "model": {**dataclasses.asdict(c.model), "n_params": self.n_params},
            "training": {**dataclasses.asdict(c.train),
                         "train_conditions": [cond.name for cond in c.train.train_conditions],
                         "best_epoch": self.history.best_epoch,
                         "best_val_wer": self.history.best_val_wer,
                         "epochs_run": len(self.history.epochs),
                         "history": self.history.epochs},
            "normalization": c.norm,
            "tokenizer": c.tokenizer,
            "environment": environment_info(),
            # Filled in by evaluate()/benchmark(): {condition name: EvalResult.row()}
            "evaluations": {},
            # The full config, so ASRConfig.from_dict(meta["config"]) rebuilds the system
            "config": c.to_dict(),
        }

    # ---- inspection -------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def summary(self) -> str:
        """Multi-line overview: name, config hash, front end and n_feats, model size, training status,
        best val WER, workspace root."""
        raise NotImplementedError("TODO(M1)")

    def featurize(self, audio: AudioLike, sr: int | None = None,
                  condition: Condition | None = None, utt_id: str = "adhoc") -> torch.Tensor:
        """Features f32[T, F] exactly as the model sees them at evaluation time (no SpecAugment).
        For plotting a mel spectrogram and a CARFAC NAP side by side."""
        raise NotImplementedError("TODO(M1)")

    def sanity_check(self, n_utts: int = 16, steps: int = 300) -> dict:
        """The notebook's pre-flight checks, on a throwaway copy of the model:
          * one batch goes through forward with the expected shapes
          * output frames per token > 1 after subsampling (needed for CTC)
          * the initial loss is finite
          * it can overfit n_utts clean utterances (loss → ~0, greedy text ≈ reference)
        Returns the measured values. Doesn't touch self.model or the global RNG state."""
        raise NotImplementedError("TODO(M1)")

    # ---- data ---------------------------------------------------------------------------
    def corpus(self, split: str) -> SpeechCorpus:
        """build('corpus', cfg.data.corpus, split=split, datasets_dir=workspace.datasets,
        n_shards=(cfg.data.n_train_shards if split == cfg.data.train_split else None),
        source=cfg.data.source)."""
        raise NotImplementedError("TODO(M1)")

    def noise_bank(self) -> NoiseBank:
        return NoiseBank(self.workspace.noise)

    def feature_store(self) -> FeatureStore:
        return FeatureStore(self.workspace.features)

    def precompute_features(self, splits: Sequence[str] | None = None,
                            conditions: Sequence[Condition] | None = None) -> list[Path]:
        """Fill features/ ahead of time (only for expensive front ends; mel/MFCC return []).

        Defaults: train split under cfg.train.train_conditions (+ variants), and val + test, clean.
        Skips folders that already have DONE, so it's safe to re-run or to split across Quest jobs.
        """
        raise NotImplementedError("TODO(M3)")

    def _make_dataset(self, split: str, conditions: Sequence[Condition], train: bool) -> FeatureDataset:
        raise NotImplementedError("TODO(M1/M3)")

    def _make_loader(self, dataset: FeatureDataset, train: bool):
        """DataLoader with LengthBucketSampler(max_frames_per_batch) and collate()."""
        raise NotImplementedError("TODO(M1)")

    # ---- training -------------------------------------------------------------------------
    def fit(self, name: str | None = None, resume: bool = True) -> History:
        """Train according to cfg.data and cfg.train; keep only the best model.

        1. Name: `name`, else an unfinished model in models/ with this config hash
           (when resume=True), else default_name().
        2. Datasets: train split under cfg.train.train_conditions; val split, clean.
        3. Trainer loop. After each epoch:
             - models/<name>_meta.json is rewritten (history so far, status "training")
             - if val WER improved: models/<name>.pt is overwritten with the new best
             - models/<name>_last.pt holds the Trainer state for resuming
        4. At the end: load the best weights into self.model, mark the model trained,
           write the final meta, delete <name>_last.pt.
        """
        raise NotImplementedError("TODO(M1)")

    # ---- inference -------------------------------------------------------------------------
    @torch.inference_mode()
    def transcribe(self, audio: AudioLike | Sequence[AudioLike], sr: int | None = None,
                   batch_size: int = 16, calibrate_level: bool = True) -> str | list[str]:
        """Audio → text, using the same Featurizer as training.

        Accepts a path (.wav/.flac), a 1-D array or tensor (then `sr` is required),
        or a list of these. Returns a str for one input and a list for a list.

        1. Load → mono → float32 → resample to 16 kHz
        2. featurizer(wave, condition=None, train=False, calibrate_level=...)
        3. Pad into batches sorted by length (original order restored at the end)
        4. model.eval(); model.decode(feats, lens); tokenizer.decode
        5. Restore the model's previous train/eval mode
        """
        raise NotImplementedError("TODO(M1)")

    def _load_audio(self, audio: AudioLike, sr: int | None) -> torch.Tensor:
        """Any AudioLike → f32[N] mono at 16 kHz (soundfile + torchaudio resample)."""
        raise NotImplementedError("TODO(M1)")

    # ---- evaluation --------------------------------------------------------------------------
    @torch.inference_mode()
    def evaluate(self, split: str | None = None, condition: Condition = CLEAN,
                 max_utts: int | None = None, ci: bool = True) -> EvalResult:
        """WER/CER/loss on a split (default cfg.data.test_split) under one condition.

        Test conditions are NOT part of the config: you choose them at evaluation time.
        Writes results/<name>/per_utt/<condition>.csv and adds the scores to
        meta["evaluations"] in models/<name>_meta.json.
        """
        raise NotImplementedError("TODO(M2)")

    def benchmark(self, conditions: Sequence[Condition], split: str | None = None,
                  max_utts: int | None = None):
        """evaluate() for every condition → pandas DataFrame, one row per condition, with
        name, front end, seed and config hashes as columns, ready to concatenate across models."""
        raise NotImplementedError("TODO(M2)")

    # ---- persistence ---------------------------------------------------------------------------
    def save(self, name: str | None = None) -> tuple[Path, Path]:
        """Write models/<name>.pt and models/<name>_meta.json; returns both paths.

        <name>.pt holds {"config": cfg.to_dict(), "model_state": model.state_dict(),
        "tokenizer": ..., "normalizer": ...}: the weights plus what's needed to rebuild
        the module. It doesn't pickle the nn.Module itself, because pickled modules
        stop loading as soon as a class is renamed or moved.

        <name>_meta.json holds self.meta(): readable without torch.
        """
        raise NotImplementedError("TODO(M1)")

    @classmethod
    def load(cls, name_or_path: str | Path, root: str | Path | None = None,
             device: str | torch.device | None = None) -> "ASR":
        """Load by name from <root>/models/, or from a path to a .pt file.

        Rebuilds with ASR(ASRConfig.from_dict(ckpt["config"]), root, device), then
        load_state_dict(strict=True). Warns if the saved CARFAC commit differs from CARFAC_COMMIT.
        """
        raise NotImplementedError("TODO(M1)")

    def __repr__(self) -> str:
        state = "trained" if self.is_trained else "untrained"
        return (f"ASR(name={self.name}, frontend={self.frontend.name}, n_feats={self.frontend.n_feats}, "
                f"model={self.cfg.model.arch}, params={self.n_params:,}, seed={self.cfg.seed}, "
                f"cfg={self.cfg.hash()}, {state})")


# =============================================================================
# Playground: `python asr_skeleton.py`
# =============================================================================
if __name__ == "__main__":
    import tempfile

    # Two systems that differ only in the front end
    mel_cfg = ASRConfig(seed=1)
    carfac_cfg = mel_cfg.replace(frontend=CarfacConfig(car_params={"erb_per_step": 0.5}))
    noagc_cfg = mel_cfg.replace(frontend=CarfacConfig(open_loop=True, post="cuberoot"))

    for label, c in (("mel", mel_cfg), ("carfac", carfac_cfg), ("carfac noagc", noagc_cfg)):
        print(f"{label:13s} name={c.frontend.name:22s} cache={c.frontend.cache_key():26s} "
              f"hash={c.hash()} controlled={c.controlled_hash()}")

    # Changing `post` must NOT change the CARFAC cache key (same cached features)
    assert CarfacConfig(post="log").cache_key() == CarfacConfig(post="cuberoot").cache_key()

    # Configs survive a JSON round trip unchanged (tuples inside CARFAC params included)
    for c in (mel_cfg, carfac_cfg, noagc_cfg):
        assert ASRConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c, "config round trip failed"

    print("conditions:", [x.name for x in (CLEAN, Condition("pub", 5), Condition(speech_level_dbfs=-46))])

    # A throwaway workspace, to see the folder layout without touching your real one
    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace(tmp)
        print(ws.summary())
        print("noise kinds available:", NoiseBank(ws.noise).available())

        # The constructor stops at the first TODO: that's where to start
        try:
            ASR(carfac_cfg, root=tmp, device="cpu")
        except NotImplementedError as e:
            print("next TODO:", e)