"""
Feature cache: compute expensive features (CARFAC) once, store them on disk, and read them
back instantly during training and evaluation.

    FeatureStore     where each cache folder lives; builds it if missing
    CachedFeatures   read-only view of one folder: cached[i] → raw f16[T, F] for utterance i

Author: Eric Oliveira
Date: 10/29/2026

* Written with the help of Claude *

"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .conditions import Condition, NoiseBank
from .constants import HOP_LENGTH, SAMPLE_RATE
from .data import SpeechCorpus
from .frontends import FrontEnd
from .utils import environment_info


class FeatureStore:
    """Cached features under <root>/features/

    One folder per (front end, corpus, split, condition[, variant]):
        features/<frontend cache key>/<corpus>/<split>_<n_utts>/<condition>[_v<k>]/
            rate.npy     float16 memmap, all utterances stacked in corpus order: (total frames, n_feats)
            index.npz    utt_ids, offsets, n_frames
            meta.json    front-end config, n_feats, cf_hz, hours, seconds taken, environment
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

    def get(self, frontend: FrontEnd, corpus: SpeechCorpus, condition: Condition,
            bank: NoiseBank | None = None, variant: int | None = None,
            batch_size: int = 32) -> CachedFeatures:
        """The cache for this combination, building it first if it isn't finished."""
        if not self.has(frontend, corpus, condition, variant):
            self.build(frontend, corpus, condition, bank, variant, batch_size)
        return CachedFeatures(self.location(frontend, corpus, condition, variant))

    def build(self, frontend: FrontEnd, corpus: SpeechCorpus, condition: Condition,
              bank: NoiseBank | None = None, variant: int | None = None, batch_size: int = 32) -> Path:
        """Compute and write one cache folder (see the class docstring). Returns its path.

        Run it in the main process (or a Quest job), never inside DataLoader workers: JAX can't fork.
        """
        loc = self.location(frontend, corpus, condition, variant)
        if (loc / "DONE").exists():
            return loc
        loc.mkdir(parents=True, exist_ok=True)

        # Where each utterance's frames go in the big stacked array (corpus order, not sorted order)
        n_samples = corpus.durations()
        n_frames = n_samples // HOP_LENGTH + 1
        offsets = np.concatenate([[0], np.cumsum(n_frames)[:-1]])

        # One float16 array on disk for the whole split; "w+" overwrites a half-written one
        rate = np.lib.format.open_memmap(loc / "rate.npy", mode="w+", dtype=np.float16,
                                         shape=(int(n_frames.sum()), frontend.n_feats))

        # Similar lengths together: less padding inside extract_batch, fewer JAX compilations
        order = np.argsort(n_samples, kind="stable")
        t0, audio_s = time.perf_counter(), 0.0
        n_batches = -(-len(order) // batch_size)

        for b, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]

            # Same degradation the live pipeline applies: level, then noise (seeded by utterance)
            utts = [corpus[int(i)] for i in idx]
            waves = [condition.apply(u.wave, u.utt_id, bank=bank, variant=variant or 0) for u in utts]

            # Write each utterance's frames into its own slot, checking the frame-count contract
            for i, feats in zip(idx, frontend.extract_batch(waves)):
                expected = (int(n_frames[i]), frontend.n_feats)
                if tuple(feats.shape) != expected:
                    raise ValueError(f"Utterance {i}: got {tuple(feats.shape)}, expected {expected}")
                rate[offsets[i]: offsets[i] + n_frames[i]] = np.asarray(feats, dtype=np.float16)

            # Progress every 20 batches: real-time factor = compute seconds per audio second
            audio_s += sum(len(w) for w in waves) / SAMPLE_RATE
            if b % 20 == 0 or b == n_batches - 1:
                elapsed = time.perf_counter() - t0
                print(f"  {min(start + batch_size, len(order)):6d}/{len(order)} utts | "
                      f"{audio_s / 3600:6.2f} h audio | {elapsed / 60:6.1f} min | RTF {elapsed / audio_s:.4f}")

        rate.flush()
        del rate

        # The index says where each utterance is; the meta says how it was made
        np.savez(loc / "index.npz", utt_ids=np.array(corpus.utt_ids()), offsets=offsets, n_frames=n_frames)
        meta = {
            "frontend": frontend.cfg.to_dict(), "cache_key": frontend.cfg.cache_key(),
            "n_feats": frontend.n_feats,
            "cf_hz": [] if getattr(frontend, "cf_hz", None) is None else [float(f) for f in frontend.cf_hz],
            "corpus": corpus.name, "split": corpus.split, "n_utts": len(corpus),
            "condition": condition.name, "variant": variant,
            "hours": float(n_samples.sum() / SAMPLE_RATE / 3600),
            "seconds": time.perf_counter() - t0,
            "environment": environment_info(),
        }
        (loc / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

        # Written last: only a complete folder ever has it
        (loc / "DONE").touch()
        return loc


class CachedFeatures:
    """Read-only view of one finished cache folder: cached[i] → raw f16[T, F] for utterance i.

    The big array is opened lazily (memory-mapped) in whichever process reads it, so sending
    this object to DataLoader workers copies only the small index, never the features.
    """

    def __init__(self, location: Path):
        self.location = Path(location)
        if not (self.location / "DONE").exists():
            raise FileNotFoundError(f"{self.location} is incomplete (no DONE file); build it first")

        # The small index is loaded now; rate.npy is opened on first access
        index = np.load(self.location / "index.npz")
        self.utt_ids = list(index["utt_ids"])
        self.offsets = index["offsets"]
        self.n_frames = index["n_frames"]
        self._rate: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int) -> torch.Tensor:
        # Memory-map on first use: the OS reads only the pages this utterance needs
        if self._rate is None:
            self._rate = np.load(self.location / "rate.npy", mmap_mode="r")
        lo = self.offsets[i]

        # np.array copies the slice out of the read-only map, so torch gets ordinary memory
        return torch.from_numpy(np.array(self._rate[lo: lo + self.n_frames[i]]))       # f16[T, F]

    def __getstate__(self) -> dict:
        # When pickled for a worker, leave the memory map behind; the worker opens its own
        state = self.__dict__.copy()
        state["_rate"] = None
        return state