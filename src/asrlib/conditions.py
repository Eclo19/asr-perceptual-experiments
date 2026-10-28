"""
Determines the audio conditions. Includes noises. This function also handles loudness 
with the convention written in constants that 0 dbFS == 104 dbSPL. This is crucial for
CARFAC's level compression. 

Author: Eric Oliveira 
Date: 10/27/2026

* Written with the help of Claude *

"""

import json
import os
import tempfile
from math import gcd
import hashlib
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .constants import SAMPLE_RATE, FULL_SCALE_DB_SPL, DEFAULT_SPEECH_DB_SPL

# ================================
# Levels: dB SPL <--> digital RMS
# ================================

# Every level in asrlib is in dB SPL. dBFS (RMS relative to 1.0) only appears inside these
# functions, through one constant:  dB SPL = dBFS + FULL_SCALE_DB_SPL

def rms(wave: torch.Tensor) -> float:
    """Root-mean-square amplitude of a waveform, in full-scale units (float64 for accuracy)."""
    return float(wave.double().pow(2).mean().sqrt())


def db_spl_to_rms(db_spl: float) -> float:
    """dB SPL → digital RMS. 65 dB SPL → 0.0112, 104 dB SPL → 1.0."""
    return 10 ** ((db_spl - FULL_SCALE_DB_SPL) / 20)


def rms_to_db_spl(value: float) -> float:
    """Digital RMS → dB SPL. Silence (RMS 0) → -inf."""
    return 20 * np.log10(value) + FULL_SCALE_DB_SPL if value > 0 else float("-inf")


def set_level(wave: torch.Tensor, db_spl: float) -> torch.Tensor:
    """Scale a waveform so its RMS is `db_spl` (dB SPL). Silent input is returned unchanged."""
    current = rms(wave)

    # Nothing to scale: an all-zero (or numerically silent) 
    if current < 1e-10:
        return wave
    
    return wave * (db_spl_to_rms(db_spl) / current)

# ================================
# Unit Variance White Noise
# ================================

def white_noise(n_samples: int, rng: np.random.Generator) -> torch.Tensor:
    """Gaussian white noise, f32[n_samples]. Its level is set afterwards by set_level."""
    return torch.from_numpy(rng.standard_normal(n_samples).astype(np.float32))


# ================================
# Condition Class
# ================================

@dataclass(frozen=True)
class Condition:
    """An acoustic condition: how a clean waveform is degraded before the front end.

    Every utterance is first scaled to `speech_db_spl` (RMS). This is the only place
    the level is set, for every front end. Then noise is scaled to `speech_db_spl - snr_db`,
    so the SNR is relative to the SPEECH. The mixture is never renormalized.

    noise:  None (clean), "white" (generated), or a folder name under noise/ ("pub", "babble", ...)
    split:  which part of each noise recording to use; training and test noise never overlap

        Condition()                                  # clean speech at 65 dB SPL
        Condition("pub", 5)                          # pub noise at 5 dB SNR (test part of the recordings)
        Condition("babble", 0, split="train")        # for multi-condition training
        Condition(speech_db_spl=45)                  # quiet clean speech (level sweep)
    """
    noise: str | None = None
    snr_db: float | None = None
    speech_db_spl: float = DEFAULT_SPEECH_DB_SPL
    split: Literal["train", "test"] = "test"

    def __post_init__(self):
        # A noise needs an SNR and an SNR needs a noise
        if (self.noise is None) != (self.snr_db is None):
            raise ValueError("noise and snr_db must be both set or both None")

        # Literal isn't enforced at runtime, so check the split here
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', not {self.split!r}")

    @property
    def name(self) -> str:
        """'clean', 'pub@5dB', 'clean@45dBSPL', 'babble@0dB@85dBSPL', 'pub@5dB-trainnoise', ...

        Used in feature-cache folder names, so every field that changes the audio appears in it.
        """
        base = "clean" if self.noise is None else f"{self.noise}@{self.snr_db:g}dB"

        # The level only appears when it isn't the default, to keep common names short
        if self.speech_db_spl != DEFAULT_SPEECH_DB_SPL:
            base += f"@{self.speech_db_spl:g}dBSPL"

        # Training-part noise must never share a cache folder with test-part noise
        if self.noise is not None and self.split == "train":
            base += "-trainnoise"
        return base

    def rng(self, utt_id: str, variant: int = 0) -> np.random.Generator:
        """A random generator that depends only on (noise kind, split, utterance, variant).

        The SNR and level are deliberately left out: every SNR of a sweep uses the SAME noise
        segment, only scaled differently, so WER-vs-SNR curves aren't confounded by which
        piece of noise each point happened to get. hashlib (not hash()) makes it identical
        on every machine and run.
        """
        key = f"{self.noise}|{self.split}|{utt_id}|{variant}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
        return np.random.default_rng(seed)

    def apply(self, wave: torch.Tensor, utt_id: str, bank: NoiseBank | None = None,
              variant: int = 0) -> torch.Tensor:
        """Degrade one waveform f32[N], the same way every time → f32[N].

        variant > 0 gives a different noise segment for the same utterance (Phase 2's K copies).
        """
        # 1. Speech at the calibrated level
        speech = set_level(wave.float(), self.speech_db_spl)

        # Clean: the level is the whole condition
        if self.noise is None:
            return speech

        # 2. The noise segment for this utterance: same one for every front end and every SNR
        rng = self.rng(utt_id, variant)
        if self.noise == "white":
            noise = white_noise(len(speech), rng)
        elif bank is None:
            raise ValueError(f"Condition {self.name!r} needs a NoiseBank to read {self.noise!r} noise")
        else:
            noise = bank.segment(self.noise, self.split, len(speech), rng)

        # 3. Noise level = speech level - SNR, then mix (float32, no clipping, no renormalizing)
        noise = set_level(noise, self.speech_db_spl - self.snr_db)
        return speech + noise


# Clean speech at the default level, defined once for readability
CLEAN = Condition()

class NoiseBank:
    """Noise audio for the Conditions.

    "white" is generated from the seeded generator and never stored.
    Every other kind is a folder of recordings you provide (or download once):

        <root>/noise/pub/*.wav        e.g. DEMAND PRESTO + PCAFETER, channel 1
        <root>/noise/babble/*.wav     e.g. 8-talker mixes from LibriSpeech train-other
        <root>/noise/splits.json      written on first use: which samples of each file are train vs test

    The 60/40 train/test split is decided once and saved, so it never changes between
    sessions or machines. A missing folder raises an error that says where to put the files.
    """
    # train/test split of the noise recordings themselves, not of the speech
    TRAIN_FRACTION = 0.6
    EXTENSIONS = (".wav", ".flac")

    def __init__(self, noise_dir: Path, sample_rate: int = SAMPLE_RATE):
        self.root = Path(noise_dir)
        self.sample_rate = sample_rate
        self.splits_path = self.root / "splits.json"

        # Loaded lazily, one kind at a time: {kind: {"train": [f32 arrays], "test": [f32 arrays]}}
        self._audio: dict[str, dict[str, list[np.ndarray]]] = {}

    def available(self) -> list[str]:
        """Kinds you can use right now: 'white' plus every non-empty subfolder of noise/."""
        kinds = ["white"]
        if self.root.exists():
            kinds += sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.iterdir()))
        return kinds

    def preload(self, kinds: Sequence[str]) -> None:
        """Load (and fix the splits of) these kinds now, in the main process, before DataLoader workers start."""
        for kind in kinds:
            if kind != "white":
                self._load_kind(kind)

    def segment(self, kind: str, split: Literal["train", "test"], n_samples: int,
                rng: np.random.Generator) -> torch.Tensor:
        """n_samples of noise from the `split` part of `kind`, at a random (seeded) position → f32[n_samples].

        A file is chosen with probability proportional to its length (every second of noise equally
        likely), then a start point within it. Parts shorter than n_samples are looped.
        """
        # White noise needs no files
        if kind == "white":
            return white_noise(n_samples, rng)

        parts = self._load_kind(kind)[split]

        # Pick a file, weighted by length, then a start point anywhere in it
        lengths = np.array([len(p) for p in parts], dtype=np.float64)
        part = parts[rng.choice(len(parts), p=lengths / lengths.sum())]
        start = int(rng.integers(len(part)))

        # Read n_samples from the start point, wrapping around the end if needed (loops short parts)
        idx = (start + np.arange(n_samples)) % len(part)
        return torch.from_numpy(part[idx])                                             # f32[n_samples], a copy

    def _load_kind(self, kind: str) -> dict[str, list[np.ndarray]]:
        """Read noise/<kind>/* → mono f32 at 16 kHz, cut into train/test parts per splits.json (cached)."""
        if kind in self._audio:
            return self._audio[kind]

        # Find the recordings, or explain where they should go
        folder = self.root / kind
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in self.EXTENSIONS) if folder.is_dir() else []
        if not files:
            raise FileNotFoundError(
                f"No {kind!r} noise found. Put .wav/.flac recordings in {folder}/ "
                f"(available now: {self.available()}).")

        splits = self._read_splits()
        parts: dict[str, list[np.ndarray]] = {"train": [], "test": []}
        changed = False

        for path in files:
            audio = self._read_mono(path)                                              # f32[n]
            key = f"{kind}/{path.name}"

            # First time this file is seen: fix its split point and remember it
            if key not in splits:
                cut = int(len(audio) * self.TRAIN_FRACTION)
                splits[key] = {"n_samples": len(audio), "train": [0, cut], "test": [cut, len(audio)]}
                changed = True

            # A file that changed since its split was fixed would silently move the train/test boundary
            entry = splits[key]
            if entry["n_samples"] != len(audio):
                raise ValueError(f"{path} has {len(audio)} samples but splits.json recorded "
                                 f"{entry['n_samples']}. Restore the file, or delete its entry to re-split.")

            for split in ("train", "test"):
                lo, hi = entry[split]
                parts[split].append(audio[lo:hi])

        if changed:
            self._write_splits(splits)

        self._audio[kind] = parts
        return parts

    def _read_mono(self, path: Path) -> np.ndarray:
        """One recording → f32[n] mono at self.sample_rate (first channel of multi-channel files)."""
        audio, sr = sf.read(path, dtype="float32", always_2d=True)                    # (n, channels)
        audio = audio[:, 0]

        # Resample with a polyphase filter if the file isn't at our rate (e.g. 48 kHz DEMAND files)
        if sr != self.sample_rate:
            g = gcd(sr, self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // g, sr // g).astype(np.float32)
        return np.ascontiguousarray(audio)

    def _read_splits(self) -> dict:
        if self.splits_path.exists():
            return json.loads(self.splits_path.read_text())
        return {}

    def _write_splits(self, splits: dict) -> None:
        # Write to a temporary file, then rename: a crash or a second process never sees half a file
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(splits, f, indent=2, sort_keys=True)
        os.replace(tmp, self.splits_path)