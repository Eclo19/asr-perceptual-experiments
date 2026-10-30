"""
The ASR class: ties everything together. An ASR owns its config, workspace, history, front end,
normalizer, tokenizer, model and featurizer, and builds everything else when it needs it.

Training a model with a given front end, data, noise and recipe, then using it:

    cfg = ASRConfig(frontend=CarfacConfig(), data=DataConfig(n_train_shards=4), seed=1)
    asr = ASR(cfg, root="~/ASR")
    asr.fit()                                            # -> models/<name>.pt + <name>_meta.json
    asr.transcribe("clip.wav")                           # -> "he hoped there would be stew"
    asr.benchmark([CLEAN, Condition("pub", 5), Condition("white", 0)])

Author: Eric Oliveira
Date: 10/29/2026

* Written with the help of Claude *

"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
import random
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from .cache import CachedFeatures, FeatureStore
from .conditions import Condition, NoiseBank
from .config import ASRConfig
from .constants import CARFAC_COMMIT, AudioLike
from .data import FeatureDataset, LengthBucketSampler, SpeechCorpus, _to_mono_16k, collate
from .features import Featurizer, Normalizer
from .frontends import FrontEnd
from .metrics import EvalResult, History
from .models import AcousticModel
from .registry import build
from .tokenizer import Tokenizer
from .train import Trainer, resolve_amp
from .utils import environment_info, pick_device, seed_everything
from .workspace import Workspace


def assert_comparable(*systems: ASR) -> None:
    """Raise if the systems differ in anything other than front end and seed.
    Call this before putting their numbers in the same table."""
    hashes = {s.cfg.controlled_hash() for s in systems}
    if len(hashes) > 1:
        raise ValueError(f"Systems differ in controlled settings: {hashes}. "
                         "Diff their cfg.to_dict() to see what changed.")


class ASR:
    """A complete, self-contained speech recognizer: front end + normalizer +
    tokenizer + acoustic model, built from one ASRConfig.

    Lifecycle
    ---------
        asr = ASR(cfg, root="~/ASR")            # untrained; weights fixed by cfg.seed
        asr.precompute_features()               # optional: fill features/ first (e.g. a Quest CPU job)
        asr.sanity_check()                      # shapes + overfit one batch
        asr.fit()                               # best model → models/<name>.pt + <name>_meta.json
        asr.transcribe("clip.wav")              # -> "the cat sat ..."
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

        # "untrained" -> "training" (fit running or interrupted) -> "trained"
        self.status = "untrained"

        # Set by fit(), save() or load(): the file stem in models/
        self.name: str | None = None

        # Corpora are loaded once per split and reused
        self._corpora: dict[str, SpeechCorpus] = {}

        # Build the components in dependency order
        seed_everything(self.cfg.seed)
        self.frontend: FrontEnd = self._build_frontend()
        self.normalizer: Normalizer = self._build_normalizer()
        self.tokenizer: Tokenizer = self._build_tokenizer()
        self.model: AcousticModel = self._build_model().to(self.device)
        self.featurizer: Featurizer = self._build_featurizer()

    def _build_frontend(self) -> FrontEnd:
        return build("frontend", self.cfg.frontend.kind, cfg=self.cfg.frontend)

    def _build_normalizer(self) -> Normalizer:
        return Normalizer(self.cfg.norm)

    def _build_tokenizer(self) -> Tokenizer:
        return build("tokenizer", self.cfg.tokenizer)

    def _build_model(self) -> AcousticModel:
        # n_feats comes from the front end and vocab_size from the tokenizer, never from constants
        return build("model", self.cfg.model.arch, n_feats=self.frontend.n_feats,
                     vocab_size=self.tokenizer.vocab_size, blank_id=self.tokenizer.blank_id,
                     **dataclasses.asdict(self.cfg.model))

    def _build_featurizer(self) -> Featurizer:
        # All training conditions must share one speech level: it's the level transcribe() matches
        levels = {c.speech_db_spl for c in self.cfg.train.train_conditions}
        if len(levels) != 1:
            raise ValueError(f"Training conditions use several speech levels {sorted(levels)}; "
                             "use one, so transcribe() knows which level to calibrate to")
        return Featurizer(self.frontend, self.normalizer, self.cfg.train.specaugment,
                          train_level_db_spl=levels.pop())

    def with_changes(self, **changes) -> ASR:
        """A NEW untrained ASR whose config differs only in `changes`, in the same workspace.

            asr.with_changes(frontend=CarfacConfig(open_loop=True))
            asr.with_changes(seed=2)
            asr.with_changes(data=dataclasses.replace(asr.cfg.data, n_train_shards=4))
            asr.with_changes(train=dataclasses.replace(asr.cfg.train,
                             train_conditions=(CLEAN, Condition("pub", 5, split="train"))))
        """
        return ASR(self.cfg.replace(**changes), root=self.workspace.root, device=self.device)

    # ---- naming and metadata --------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self.status == "trained"

    @property
    def clean(self) -> Condition:
        """Clean speech at the training level: what validation and transcribe() use."""
        return Condition(speech_db_spl=self.featurizer.train_level_db_spl)

    def default_name(self, date: dt.date | None = None) -> str:
        """<date>_<frontend>_<dataset>_<noise>_s<seed>_<config hash>,
        e.g. 2026-10-29_carfac-agc-log_librispeech-train.100_clean_s1_a1b2c3.

        The date says when training started. The hash makes the name unique to the
        config: fit() uses it to find and resume an unfinished run, even from another day.
        """
        date = date or dt.date.today()
        c = self.cfg
        return f"{date.isoformat()}_{c.frontend.name}_{c.data.tag}_{c.train.noise_tag}_s{c.seed}_{c.hash()}"

    def meta(self) -> dict:
        """Everything about this model except the weights: written to models/<name>_meta.json."""
        c = self.cfg
        best = self.history.best_val_wer
        return {
            "name": self.name,
            "status": self.status,
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
                         "best_val_wer": None if best == float("inf") else best,
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

    def _write_meta(self) -> None:
        """Rewrite models/<name>_meta.json, keeping any evaluations already recorded there."""
        _, meta_path = self.workspace.model_paths(self.name)
        meta = self.meta()
        if meta_path.exists():
            meta["evaluations"] = json.loads(meta_path.read_text()).get("evaluations", {})
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    # ---- inspection -------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def summary(self) -> str:
        """Multi-line overview of what this system is and where it stands."""
        best = self.history.best_val_wer
        return "\n".join([
            f"ASR {self.name or '(unsaved)'}  [{self.status}]",
            f"  config      {self.cfg.hash()} (controlled {self.cfg.controlled_hash()}), seed {self.cfg.seed}",
            f"  front end   {self.frontend.name}: {self.frontend.n_feats} features, 100 frames/s",
            f"  model       {self.cfg.model.arch}: {self.n_params:,} parameters on {self.device}",
            f"  data        {self.cfg.data.tag}, training noise: {self.cfg.train.noise_tag}",
            f"  best val    {'—' if best == float('inf') else f'{best:.2%} WER (epoch {self.history.best_epoch})'}",
            f"  workspace   {self.workspace.root}",
        ])

    def featurize(self, audio: AudioLike, sr: int | None = None,
                  condition: Condition | None = None, utt_id: str = "adhoc") -> torch.Tensor:
        """Features f32[T, F] exactly as the model sees them at evaluation time (no SpecAugment).
        For plotting a mel spectrogram and a CARFAC NAP side by side."""
        wave = self._load_audio(audio, sr)
        bank = self.noise_bank() if condition is not None and condition.noise else None
        return self.featurizer(wave, condition=condition, utt_id=utt_id, bank=bank, train=False)

    def sanity_check(self, n_utts: int = 16, steps: int = 300) -> dict:
        """The notebook's pre-flight checks, on a throwaway copy of the model:
          * one batch goes through forward with the expected shapes
          * output frames per token > 1 after subsampling (needed for CTC)
          * the initial loss is finite
          * it can overfit n_utts clean utterances (loss → ~0, greedy text ≈ reference)
        Returns the measured values. Doesn't touch self.model or the global RNG state."""
        # Save every random generator, so this check doesn't change what training does afterwards
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())

        # A fixed batch of the first n_utts validation utterances (clean, no SpecAugment)
        dataset = self._make_dataset(self.cfg.data.val_split, [self.clean], train=False)
        batch = collate([dataset[i] for i in range(min(n_utts, len(dataset)))])
        feats, feat_lens, tokens, tok_lens = [t.to(self.device) for t in batch[:4]]

        # Shapes and the CTC length requirement, on a copy of the model
        probe = copy.deepcopy(self.model).train()
        log_probs, out_lens = probe(feats, feat_lens)
        assert log_probs.shape == (feats.size(0), int(out_lens.max()), self.tokenizer.vocab_size)
        min_ratio = float((out_lens.float() / tok_lens.float()).min())
        initial = probe.loss(feats, feat_lens, tokens, tok_lens).item()
        print(f"log_probs {tuple(log_probs.shape)} | min frames per token {min_ratio:.2f} (must be > 1) | "
              f"initial loss {initial:.2f}")

        # Overfit the batch: if this fails, lengths, masks or the loss wiring are broken
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss = probe.loss(feats, feat_lens, tokens, tok_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            if step % 100 == 0 or step == steps:
                probe.eval()
                hyp = self.tokenizer.decode(probe.decode(feats[:1], feat_lens[:1])[0])
                probe.train()
                print(f"  step {step:4d} | loss {loss.item():.3f} | {hyp[:60]!r}")

        ref = self.tokenizer.decode(tokens[0, : tok_lens[0]].tolist())
        print(f"  target          {ref[:60]!r}")

        random.setstate(rng_state[0]); np.random.set_state(rng_state[1]); torch.set_rng_state(rng_state[2])
        return {"initial_loss": initial, "final_loss": loss.item(), "min_frames_per_token": min_ratio,
                "example_hyp": hyp, "example_ref": ref}

    # ---- data ---------------------------------------------------------------------------
    def corpus(self, split: str) -> SpeechCorpus:
        """The corpus for one split (loaded once, then reused)."""
        if split not in self._corpora:
            d = self.cfg.data
            self._corpora[split] = build(
                "corpus", d.corpus, split=split, datasets_dir=self.workspace.datasets,
                n_shards=d.n_train_shards if split == d.train_split else None, source=d.source)
        return self._corpora[split]

    def noise_bank(self) -> NoiseBank:
        return NoiseBank(self.workspace.noise)

    def feature_store(self) -> FeatureStore:
        return FeatureStore(self.workspace.features)

    def _caches(self, corpus: SpeechCorpus, conditions: Sequence[Condition], bank: NoiseBank,
                n_variants: int = 0) -> list[CachedFeatures]:
        """One CachedFeatures per (condition, variant), built here in the main process if missing.

        Same order as FeatureDataset.items: for each condition, variants 0..K-1 if noisy, else just 0.
        """
        store = self.feature_store()
        caches = []
        for c in conditions:
            for v in range(max(1, n_variants) if c.noise else 1):
                variant = v if (c.noise and n_variants > 0) else None
                caches.append(store.get(self.frontend, corpus, c, bank=bank, variant=variant))
        return caches

    def _make_dataset(self, split: str, conditions: Sequence[Condition], train: bool,
                      n_variants: int = 0) -> FeatureDataset:
        corpus = self.corpus(split)

        # Load recorded noise in the main process, so splits.json is fixed before workers start
        bank = self.noise_bank()
        bank.preload(sorted({c.noise for c in conditions if c.noise}))

        # Expensive front ends (CARFAC) are read from the cache; cheap ones run live in the workers
        cache = self._caches(corpus, conditions, bank, n_variants) if self.frontend.expensive else None
        return FeatureDataset(corpus, self.featurizer, self.tokenizer, conditions, train=train,
                              bank=bank, n_variants=n_variants, cache=cache,
                              max_seconds=self.cfg.data.max_train_seconds if train else None)

    def _make_loader(self, dataset, train: bool, n_frames: np.ndarray | None = None) -> DataLoader:
        """DataLoader with LengthBucketSampler(max_frames_per_batch) and collate()."""
        n_frames = dataset.n_frames() if n_frames is None else n_frames
        sampler = LengthBucketSampler(n_frames, self.cfg.train.max_frames_per_batch,
                                      shuffle=train, seed=self.cfg.seed)
        workers = self.cfg.train.num_workers
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                          pin_memory=self.device.type == "cuda", persistent_workers=workers > 0)

    def precompute_features(self, splits: Sequence[str] | None = None,
                            conditions: Sequence[Condition] | None = None) -> list[Path]:
        """Fill features/ ahead of time (only for expensive front ends; mel/MFCC return []).

        Defaults: train split under cfg.train.train_conditions (+ variants), and val + test, clean.
        Skips folders that already have DONE, so it's safe to re-run or to split across Quest jobs.
        """
        if not self.frontend.expensive:
            print(f"{self.frontend.name} is computed on the fly; nothing to precompute.")
            return []

        # (split, conditions, variants) jobs
        d, t = self.cfg.data, self.cfg.train
        if splits is None and conditions is None:
            jobs = [(d.train_split, t.train_conditions, t.n_noisy_variants),
                    (d.val_split, [self.clean], 0), (d.test_split, [self.clean], 0)]
        else:
            jobs = [(s, conditions or [self.clean], 0) for s in (splits or [d.test_split])]

        bank = self.noise_bank()
        locations = []
        for split, conds, n_variants in jobs:
            bank.preload(sorted({c.noise for c in conds if c.noise}))
            caches = self._caches(self.corpus(split), conds, bank, n_variants)
            locations += [c.location for c in caches]
        return locations

    # ---- training -------------------------------------------------------------------------
    def fit(self, name: str | None = None, resume: bool = True) -> History:
        """Train according to cfg.data and cfg.train; keep only the best model.

        1. Name: `name`, else an unfinished model in models/ with this config hash
           (when resume=True), else default_name(). A finished one is loaded instead of retrained.
        2. Datasets: train split under cfg.train.train_conditions; val split, clean.
        3. Trainer loop. After each epoch:
             - if val WER improved: models/<name>.pt is overwritten with the new best
             - models/<name>_last.pt holds the Trainer state for resuming
             - models/<name>_meta.json is rewritten (history so far, status "training")
        4. At the end: load the best weights into self.model, mark the model trained,
           write the final meta, delete <name>_last.pt.
        """
        # 1. Pick the name: an earlier run of this exact config, or a new one
        if name is None and resume:
            for meta_path in self.workspace.find_models(self.cfg.hash()):
                earlier = json.loads(meta_path.read_text())
                if earlier["status"] == "trained":
                    print(f"{earlier['name']} is already trained with this config; loading it.")
                    self._load_weights(self.workspace.model_paths(earlier["name"])[0])
                    self.name, self.status = earlier["name"], "trained"
                    return self.history
                name = earlier["name"]
                break
        self.name = name or self.default_name()
        best_pt, _ = self.workspace.model_paths(self.name)
        last_pt = self.workspace.models / f"{self.name}_last.pt"

        # 2. Data
        t = self.cfg.train
        train_ds = self._make_dataset(self.cfg.data.train_split, t.train_conditions, train=True,
                                      n_variants=t.n_noisy_variants)
        val_ds = self._make_dataset(self.cfg.data.val_split, [self.clean], train=False)
        trainer = Trainer(self.model, self.tokenizer, t, self.device,
                          self._make_loader(train_ds, train=True), self._make_loader(val_ds, train=False),
                          self.history)

        # Continue an interrupted run exactly where it stopped (weights_only=False: RNG states inside)
        if resume and last_pt.exists():
            trainer.load_state_dict(torch.load(last_pt, map_location=self.device, weights_only=False))
            print(f"Resuming {self.name} after epoch {trainer.epoch}")

        self.status = "training"
        self._write_meta()

        # 3. Saving happens here; the Trainer itself never touches files
        def on_epoch_end(tr: Trainer, record: dict, is_best: bool) -> None:
            self.history = tr.history
            if is_best:
                self.save(self.name)
            torch.save(tr.state_dict(), last_pt)
            self._write_meta()

        self.history = trainer.fit(on_epoch_end=on_epoch_end)

        # 4. Finish: best weights in memory, final files, resume file removed
        self._load_weights(best_pt)
        self.status = "trained"
        self.save(self.name)
        last_pt.unlink(missing_ok=True)
        return self.history

    # ---- inference -------------------------------------------------------------------------
    @torch.inference_mode()
    def transcribe(self, audio: AudioLike | Sequence[AudioLike], sr: int | None = None,
                   batch_size: int = 16, calibrate_level: bool = True) -> str | list[str]:
        """Audio → text, using the same Featurizer as training.

        Accepts a path (.wav/.flac), a 1-D array or tensor (then `sr` is required),
        or a list of these. Returns a str for one input and a list for a list.
        """
        single = isinstance(audio, (str, Path, np.ndarray, torch.Tensor))
        items = [audio] if single else list(audio)

        # 1–2. Load each input and compute its features exactly as at evaluation time
        feats = [self.featurizer(self._load_audio(a, sr), train=False, calibrate_level=calibrate_level)
                 for a in items]

        # 3. Batch by length; remember where each result belongs
        order = sorted(range(len(feats)), key=lambda i: feats[i].shape[0])
        texts = [""] * len(feats)

        was_training = self.model.training
        self.model.eval()
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = pad_sequence([feats[i] for i in idx], batch_first=True).to(self.device)
            lens = torch.tensor([feats[i].shape[0] for i in idx], device=self.device)

            # 4. Decode, then put each text back in its input's position
            for i, ids in zip(idx, self.model.decode(batch, lens)):
                texts[i] = self.tokenizer.decode(ids)

        # 5. Leave the model in the mode we found it
        self.model.train(was_training)
        return texts[0] if single else texts

    def _load_audio(self, audio: AudioLike, sr: int | None) -> torch.Tensor:
        """Any AudioLike → f32[N] mono at 16 kHz."""
        if isinstance(audio, (str, Path)):
            data, file_sr = sf.read(str(Path(audio).expanduser()), dtype="float32")
            return _to_mono_16k(data, file_sr)
        if sr is None:
            raise ValueError("Pass sr= when giving raw samples (array or tensor)")
        data = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
        return _to_mono_16k(data.astype(np.float32), sr)

    # ---- evaluation --------------------------------------------------------------------------
    @torch.inference_mode()
    def evaluate(self, split: str | None = None, condition: Condition | None = None,
                 max_utts: int | None = None, ci: bool = True) -> EvalResult:
        """WER/CER/loss on a split (default cfg.data.test_split) under one condition
        (default: clean at the training level).

        Test conditions are NOT part of the config: you choose them at evaluation time.
        Writes results/<name>/per_utt/<condition>.csv and adds the scores to
        meta["evaluations"] in models/<name>_meta.json (when the model has a name).
        """
        split = split or self.cfg.data.test_split
        condition = condition or self.clean
        dataset = self._make_dataset(split, [condition], train=False)

        # Optionally only the first max_utts utterances (quick checks)
        n_frames = dataset.n_frames()
        if max_utts is not None and max_utts < len(dataset):
            dataset, n_frames = Subset(dataset, range(max_utts)), n_frames[:max_utts]
        loader = self._make_loader(dataset, train=False, n_frames=n_frames)

        was_training = self.model.training
        self.model.eval()
        amp = resolve_amp(self.cfg.train.amp, self.device)
        total, n, utt_ids, hyps, refs = 0.0, 0, [], [], []

        for feats, feat_lens, tokens, tok_lens, ids in loader:
            feats, feat_lens = feats.to(self.device), feat_lens.to(self.device)
            tokens, tok_lens = tokens.to(self.device), tok_lens.to(self.device)
            with torch.autocast(self.device.type, dtype=amp or torch.float32, enabled=amp is not None):
                loss = self.model.loss(feats, feat_lens, tokens, tok_lens)
                decoded = self.model.decode(feats, feat_lens)

            total += loss.item() * feats.size(0)
            n += feats.size(0)
            utt_ids += ids
            hyps += [self.tokenizer.decode(h) for h in decoded]
            refs += [self.tokenizer.decode(t[:l].tolist()) for t, l in zip(tokens.cpu(), tok_lens.cpu())]

        self.model.train(was_training)
        result = EvalResult.from_utterances(condition.name, utt_ids, refs, hyps, loss=total / max(n, 1), ci=ci)

        # Record it next to the model: per-utterance CSV + a row in _meta.json
        if self.name is not None:
            out_dir = self.workspace.results / self.name / "per_utt"
            out_dir.mkdir(parents=True, exist_ok=True)
            result.per_utt.to_csv(out_dir / f"{split}__{condition.name}.csv", index=False)

            _, meta_path = self.workspace.model_paths(self.name)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["evaluations"][f"{split}/{condition.name}"] = {**result.row(), "split": split}
                meta_path.write_text(json.dumps(meta, indent=2, default=str))
        return result

    def benchmark(self, conditions: Sequence[Condition], split: str | None = None,
                  max_utts: int | None = None) -> pd.DataFrame:
        """evaluate() for every condition → one row per condition, with the model's identity as
        columns, ready to concatenate across models: pd.concat([a.benchmark(C), b.benchmark(C)])."""
        rows = []
        for c in conditions:
            r = self.evaluate(split, c, max_utts=max_utts)
            print(f"{c.name:24s} WER {r.wer:6.2%}  CER {r.cer:6.2%}")
            rows.append({"name": self.name, "frontend": self.cfg.frontend.name, "seed": self.cfg.seed,
                         "config_hash": self.cfg.hash(), "controlled_hash": self.cfg.controlled_hash(),
                         "split": split or self.cfg.data.test_split, "noise": c.noise, "snr_db": c.snr_db,
                         "speech_db_spl": c.speech_db_spl, **r.row()})
        return pd.DataFrame(rows)

    # ---- persistence ---------------------------------------------------------------------------
    def save(self, name: str | None = None) -> tuple[Path, Path]:
        """Write models/<name>.pt and models/<name>_meta.json; returns both paths.

        <name>.pt holds the config and the weights (plus tokenizer/normalizer state): enough to rebuild
        the module. It doesn't pickle the nn.Module itself, because pickled modules stop loading as
        soon as a class is renamed or moved.
        """
        self.name = name or self.name or self.default_name()
        pt_path, meta_path = self.workspace.model_paths(self.name)
        torch.save({"config": self.cfg.to_dict(),
                    "model_state": self.model.state_dict(),
                    "tokenizer": self.tokenizer.state_dict(),
                    "normalizer": self.normalizer.state_dict(),
                    "carfac_commit": CARFAC_COMMIT}, pt_path)
        self._write_meta()
        return pt_path, meta_path

    def _load_weights(self, pt_path: Path) -> dict:
        ckpt = torch.load(pt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])                          # strict: shapes must match
        return ckpt

    @classmethod
    def load(cls, name_or_path: str | Path, root: str | Path | None = None,
             device: str | torch.device | None = None) -> ASR:
        """Load by name from <root>/models/, or from a path to a .pt file."""
        path = Path(name_or_path).expanduser()
        if path.suffix != ".pt":
            path = Workspace(root).model_paths(str(name_or_path))[0]

        # Rebuild the system from its own config, then load the weights into it
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        asr = cls(ASRConfig.from_dict(ckpt["config"]), root=root, device=device)
        asr._load_weights(path)
        asr.name, asr.status = path.stem, "trained"

        # A different CARFAC version would compute different features than the model was trained on
        if ckpt.get("carfac_commit") != CARFAC_COMMIT and asr.cfg.frontend.kind == "carfac":
            warnings.warn(f"Model trained with CARFAC {ckpt.get('carfac_commit')}, "
                          f"installed asrlib expects {CARFAC_COMMIT}")

        # Bring the training history back from the meta file, if it's there
        meta_path = path.with_name(f"{path.stem}_meta.json")
        if meta_path.exists():
            training = json.loads(meta_path.read_text())["training"]
            asr.history = History(epochs=training["history"], best_epoch=training["best_epoch"],
                                  best_val_wer=training["best_val_wer"] or float("inf"))
        return asr

    def __repr__(self) -> str:
        return (f"ASR(name={self.name}, frontend={self.frontend.name}, n_feats={self.frontend.n_feats}, "
                f"model={self.cfg.model.arch}, params={self.n_params:,}, seed={self.cfg.seed}, "
                f"cfg={self.cfg.hash()}, {self.status})")