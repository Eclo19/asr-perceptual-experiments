"""
Speech data: corpora (where audio and text come from), and the pieces that turn a corpus
into training batches.

    SpeechCorpus        any source of (utt_id, wave, text): LibriSpeech, your own recordings
    FeatureDataset      corpus + conditions + featurizer -> (feats, tokens, utt_id) per item
    collate             a list of items -> padded batch tensors + true lengths
    LengthBucketSampler batches of similar-length utterances, so padding stays tiny

Author: Eric Oliveira
Date: 10/29/2026

* Written with the help of Claude *

"""

from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .conditions import CLEAN, Condition, NoiseBank
from .constants import HOP_LENGTH, SAMPLE_RATE
from .features import Featurizer
from .registry import register
from .tokenizer import Tokenizer


@dataclass
class Utterance:
    """One item from any corpus."""
    utt_id: str
    wave: torch.Tensor                      # f32[N], mono, 16 kHz
    sample_rate: int
    text: str


def _to_mono_16k(audio: np.ndarray, sr: int) -> torch.Tensor:
    """Decoded audio (n,) or (n, channels) → f32[N] mono at 16 kHz."""
    if audio.ndim == 2:
        audio = audio[:, 0]                                                    # first channel
    wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return AF.resample(wave, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else wave


# ==============================
# Corpora
# ==============================

class SpeechCorpus(ABC):
    """Any source of (utt_id, wave, text). Lengths, ids and texts are available without decoding audio."""
    name: str                               # used in feature cache paths, e.g. "librispeech"
    split: str

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, i: int) -> Utterance: ...

    @abstractmethod
    def durations(self) -> np.ndarray:
        """Samples per utterance at 16 kHz, read without decoding audio."""

    @abstractmethod
    def utt_ids(self) -> list[str]: ...

    @abstractmethod
    def texts(self) -> list[str]: ...


@register("corpus", "librispeech")
class LibriSpeech(SpeechCorpus):
    """LibriSpeech ("clean" subset), from datasets/ first and Hugging Face otherwise.

    split: "train.100", "train.360", "validation" or "test" (the folder names in the HF mirror).

    Lookup order (source="auto"):
      1. <root>/datasets/librispeech/<split>/*.parquet  
      2. Hugging Face, downloading only the requested parquet shards into <root>/datasets/hf_cache/

    Audio stays as raw FLAC bytes (Audio(decode=False)) and is decoded with soundfile,
    as in the notebook, so nothing depends on TorchCodec.
    """
    HF_REPO = "openslr/librispeech_asr"

    def __init__(self, split: str, datasets_dir: Path, n_shards: int | None = None,
                 source: Literal["auto", "local", "hf"] = "auto", **_):
        self.name = "librispeech"
        self.split = split
        self.datasets_dir = Path(datasets_dir)
        self.n_shards = n_shards
        self.source = source
        self.rows = self._load()
        self._durations: np.ndarray | None = None

    def local_dir(self) -> Path:
        return self.datasets_dir / self.name / self.split

    def _load(self):
        """Local parquet files or Hugging Face, per `source` -> an HF Dataset with raw audio bytes."""

        # Imported here so the rest of asrlib works without the `datasets` library
        from datasets import Audio, load_dataset

        local_files = sorted(self.local_dir().glob("*.parquet"))
        use_local = self.source == "local" or (self.source == "auto" and local_files)

        if use_local:
            if not local_files:
                raise FileNotFoundError(f"No parquet files in {self.local_dir()} (source='local')")
            rows = load_dataset("parquet", data_files=[str(f) for f in local_files], split="train")
        else:
            # Ask for specific shards, so only those are downloaded (each train shard ≈ 7 h, 460 MB)
            if self.n_shards is not None:
                files = [f"clean/{self.split}/{i:04d}.parquet" for i in range(self.n_shards)]
            else:
                files = f"clean/{self.split}/*.parquet"
            rows = load_dataset(self.HF_REPO, data_files={"data": files}, split="data",
                                cache_dir=str(self.datasets_dir / "hf_cache"),
                                verification_mode="no_checks")

        # Keep the FLAC bytes; we decode them ourselves
        return rows.cast_column("audio", Audio(decode=False))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Utterance:
        row = self.rows[int(i)]
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        return Utterance(row["id"], _to_mono_16k(audio, sr), SAMPLE_RATE, row["text"])

    def durations(self) -> np.ndarray:
        """Samples per utterance, from the FLAC headers only (computed once, then remembered)."""
        if self._durations is None:
            audio = self.rows.select_columns(["audio"])
            out = np.empty(len(audio), dtype=np.int64)
            for i, row in enumerate(audio):
                info = sf.info(io.BytesIO(row["audio"]["bytes"]))
                out[i] = round(info.frames * SAMPLE_RATE / info.samplerate)
            self._durations = out
        return self._durations

    def utt_ids(self) -> list[str]:
        return list(self.rows["id"])

    def texts(self) -> list[str]:
        return list(self.rows["text"])


@register("corpus", "local_folder")
class LocalFolderCorpus(SpeechCorpus):
    """Your own recordings: <root>/datasets/<name>/<split>/ holding audio files and a manifest.csv
    with columns `path,text` (paths relative to that folder). The utterance id is the file name
    without its extension."""

    def __init__(self, split: str, datasets_dir: Path, name: str = "my_recordings", **_):
        self.name = name
        self.split = split
        self.folder = Path(datasets_dir) / name / split

        # Read the manifest once: one (path, text) pair per utterance
        manifest = self.folder / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Expected {manifest} with columns path,text")
        with manifest.open(newline="") as f:
            self.entries = [(self.folder / row["path"], row["text"]) for row in csv.DictReader(f)]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Utterance:
        path, text = self.entries[int(i)]
        audio, sr = sf.read(path, dtype="float32")
        return Utterance(path.stem, _to_mono_16k(audio, sr), SAMPLE_RATE, text)

    def durations(self) -> np.ndarray:
        infos = [sf.info(path) for path, _ in self.entries]
        return np.array([round(x.frames * SAMPLE_RATE / x.samplerate) for x in infos], dtype=np.int64)

    def utt_ids(self) -> list[str]:
        return [path.stem for path, _ in self.entries]

    def texts(self) -> list[str]:
        return [text for _, text in self.entries]


# ==============================
# Dataset: corpus × conditions → model inputs
# ==============================

class FeatureDataset(Dataset):
    """(feats f32[T, F], tokens int64[U], utt_id) for every (condition, variant, utterance).

    Several conditions (Phase 2) are simply concatenated: item i is condition i // n_utts,
    utterance i % n_utts. Noisy conditions get `n_variants` different noise draws each.

    Features come from `cache` when given (one CachedFeatures per (condition, variant), built by
    cache.py; required for CARFAC), otherwise they're computed live by the Featurizer (mel, MFCC).
    """

    def __init__(self, corpus: SpeechCorpus, featurizer: Featurizer, tokenizer: Tokenizer,
                 conditions: Sequence[Condition] = (CLEAN,), train: bool = False,
                 bank: NoiseBank | None = None, n_variants: int = 0,
                 cache: Sequence | None = None, max_seconds: float | None = None):
        self.corpus = corpus
        self.featurizer = featurizer
        self.train = train
        self.bank = bank
        self.cache = cache

        # One entry per (condition, variant): clean conditions need only variant 0
        self.items = [(c, v) for c in conditions for v in range(max(1, n_variants) if c.noise else 1)]
        if cache is not None and len(cache) != len(self.items):
            raise ValueError(f"Got {len(cache)} caches for {len(self.items)} (condition, variant) pairs")

        # Optionally drop very long utterances (training only): keep a list of corpus indices
        durations = corpus.durations()
        keep = np.arange(len(corpus))
        if max_seconds is not None:
            keep = keep[durations <= max_seconds * SAMPLE_RATE]
        self.keep = keep
        self._frames = durations[keep] // HOP_LENGTH + 1  # frames per kept utterance

        # Text → token ids once, up front (no audio needed)
        texts, ids = corpus.texts(), corpus.utt_ids()
        self.tokens = [torch.tensor(tokenizer.encode(texts[j]), dtype=torch.long) for j in keep]
        self.ids = [ids[j] for j in keep]

    def __len__(self) -> int:
        return len(self.items) * len(self.keep)

    def n_frames(self) -> np.ndarray:
        """Frames per item, in dataset order (for LengthBucketSampler)."""
        return np.tile(self._frames, len(self.items))

    def __getitem__(self, i: int):
        # Which (condition, variant) and which utterance this item is
        k, j = divmod(int(i), len(self.keep))
        condition, variant = self.items[k]

        if self.cache is not None:
            # Cached raw features → post → normalize → [SpecAugment]
            raw = self.cache[k][int(self.keep[j])]
            feats = self.featurizer.from_raw(raw, train=self.train)
        else:
            # Live: decode audio → condition (level + noise) → full pipeline
            utt = self.corpus[int(self.keep[j])]
            feats = self.featurizer(utt.wave, condition=condition, utt_id=utt.utt_id,
                                    bank=self.bank, variant=variant, train=self.train)

        return feats, self.tokens[j], self.ids[j]                                # (T, F), (U,), str


# ==============================
# Batching
# ==============================

def collate(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """List of (feats (T_i, F), tokens (U_i,), utt_id) -> padded batch tensors + TRUE lengths."""
    feats, tokens, utt_ids = zip(*batch)
    feat_lens = torch.tensor([f.shape[0] for f in feats])                        # (B,)
    tok_lens = torch.tensor([t.shape[0] for t in tokens])                        # (B,)
    feats = pad_sequence(feats, batch_first=True)                                # (B, T_max, F), zero-padded
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)             # (B, U_max), 0 = blank/pad
    return feats, feat_lens, tokens, tok_lens, list(utt_ids)


class LengthBucketSampler:
    """Yields batches (lists of dataset indices) of similar-length utterances.

    Utterances are sorted by length (with random tie-breaking each epoch), then cut into batches
    whose padded size, batch size x longest utterance, stays under `max_frames`. So batches of
    short utterances hold many items and batches of long ones hold few, and padding is tiny.
    The order of batches is shuffled every epoch; call set_epoch() each epoch for a new order.

    Use it as DataLoader(dataset, batch_sampler=LengthBucketSampler(...), collate_fn=collate).
    """

    def __init__(self, n_frames: Sequence[int], max_frames: int, shuffle: bool = True, seed: int = 0):
        self.n_frames = np.asarray(n_frames)
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:

        # Same seed + epoch = same batches, so runs are reproducible and resumable
        rng = np.random.default_rng(self.seed + self.epoch)

        # Sort by length; random jitter only breaks ties, so equal lengths mix differently each epoch
        jitter = rng.uniform(0, 1, len(self.n_frames)) if self.shuffle else np.zeros(len(self.n_frames))
        order = np.lexsort((jitter, self.n_frames))

        # Greedily fill each batch until one more utterance would exceed the frame budget
        batches, current, longest = [], [], 0
        for i in order:
            longest_if_added = max(longest, self.n_frames[i])
            if current and longest_if_added * (len(current) + 1) > self.max_frames:
                batches.append(current)
                current, longest_if_added = [], self.n_frames[i]
            current.append(int(i))
            longest = longest_if_added
        if current:
            batches.append(current)

        # Shuffle the order of the batches (not their contents)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())