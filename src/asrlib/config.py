
"""
Important configurations for the ASR class. This is the source of truth for what a system IS. 

Author: Eric Oliveira 
Date: 10/27/2026

* Written with the help of Claude *

"""

import json
from typing import Any
import hashlib
import dataclasses
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from .constants import CARFAC_COMMIT
from .conditions import Condition, CLEAN

# ==============================
# Helpers
# ==============================

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


# ==============================
# Config Classes
# ==============================

@dataclass(frozen=True) # ensure fields cannot be overwritten in runtime
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

    @property # Traditional class lookup methods. This gets the name of the front end
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
    log_eps: float = field(default=1e-8, metadata={"cache": False})

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
    gets right. The front end refuses designs that need 5 taps.

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
        # Literal types aren't enforced by Python, so check the one we change often
        if self.post not in ("log", "cuberoot", "none"):
            raise ValueError(f"post must be 'log', 'cuberoot' or 'none', not {self.post!r}")

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