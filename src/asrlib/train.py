"""
Train an acoustic model with a given recipe (TrainConfig): optimizer, learning-rate schedule,
mixed precision, gradient clipping, validation and early stopping.

The Trainer never reads or writes files. ASR.fit() builds it, saves checkpoints from the
on_epoch_end callback, and throws it away afterwards, so a loaded model never carries an 
optimizer.

Author: Eric Oliveira
Date: 10/29/2026

* Written with the help of Claude *

"""

from __future__ import annotations

import dataclasses
import math
import random
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig
from .metrics import History, error_rate
from .models import AcousticModel
from .tokenizer import Tokenizer


def resolve_amp(setting: str, device: torch.device) -> torch.dtype | None:
    """Mixed-precision dtype for this machine: bf16 on A100/L4 (and newer), fp16 on T4, None = off."""
    if setting == "off" or device.type != "cuda":
        return None
    if setting == "bf16":
        return torch.bfloat16
    if setting == "fp16":
        return torch.float16

    # "auto": bf16 needs compute capability 8+ (Ampere or newer)
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


class Trainer:
    """Optimizer, schedule, mixed precision and the epoch loop: everything only training needs.

      * AdamW with decay / no-decay parameter groups (no decay on biases and norms)
      * linear warmup → cosine decay, stepped per batch; progress clamped at 1, so the LR can't
        climb back up (the bug found in DL_review's mel run)
      * AMP: bf16 on A100 / L4, fp16 + GradScaler on T4, off on CPU / MPS
      * gradient clipping, early stopping on val WER
    """

    def __init__(self, model: AcousticModel, tokenizer: Tokenizer, tcfg: TrainConfig,
                 device: torch.device, train_loader, val_loader, history: History | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.tcfg = tcfg
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history = history if history is not None else History()
        self.epoch = 0  # epochs completed

        # Weight decay only on matrices: shrinking biases and LayerNorm gains only hurts
        decay = [p for p in model.parameters() if p.ndim > 1]
        no_decay = [p for p in model.parameters() if p.ndim <= 1]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": tcfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=tcfg.peak_lr)

        # Warmup then cosine, as a multiplier on peak_lr, stepped once per batch
        self.total_steps = tcfg.epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_factor)

        # Mixed precision; the gradient scaler is only needed for fp16 (bf16 has fp32's range)
        self.amp_dtype = resolve_amp(tcfg.amp, device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)

    def _lr_factor(self, step: int) -> float:
        """0 -> 1 linearly over warmup_steps, then 1 -> 0 along a cosine. Clamped: never rises again."""
        warmup = self.tcfg.warmup_steps
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, self.total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    def _to_device(self, batch):
        """(feats, feat_lens, tokens, tok_lens, utt_ids) → the four tensors on the training device."""
        feats, feat_lens, tokens, tok_lens, _ = batch
        return [t.to(self.device, non_blocking=True) for t in (feats, feat_lens, tokens, tok_lens)]

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype or torch.float32,
                              enabled=self.amp_dtype is not None)

    # ---- one epoch ---------------------------------------------------------------------

    def train_epoch(self) -> float:
        """One pass over train_loader; returns the mean loss (weighted by batch size)."""
        self.model.train()

        # A new batch order each epoch (LengthBucketSampler is seeded by epoch, so resuming is exact)
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)

        total, n = 0.0, 0
        for batch in self.train_loader:
            feats, feat_lens, tokens, tok_lens = self._to_device(batch)

            # Forward + loss (mixed precision inside; the CTC loss itself is fp32)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = self.model.loss(feats, feat_lens, tokens, tok_lens)

            # Backward; unscale before clipping so the threshold applies to the true gradients
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)

            # Update (the scaler skips the step if fp16 produced inf/NaN gradients), then the LR
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total += loss.item() * feats.size(0)
            n += feats.size(0)
        return total / max(n, 1)

    @torch.inference_mode()
    def validate(self) -> dict:
        """Loss, WER, CER on val_loader (greedy decoding, no SpecAugment), plus one example."""
        self.model.eval()
        total, n, hyps, refs = 0.0, 0, [], []

        for batch in self.val_loader:
            feats, feat_lens, tokens, tok_lens = self._to_device(batch)
            with self._autocast():
                loss = self.model.loss(feats, feat_lens, tokens, tok_lens)
                ids = self.model.decode(feats, feat_lens)

            total += loss.item() * feats.size(0)
            n += feats.size(0)

            # References come back from the token ids, so they're normalized exactly like the hyps
            hyps += [self.tokenizer.decode(h) for h in ids]
            refs += [self.tokenizer.decode(t[:l].tolist()) for t, l in zip(tokens.cpu(), tok_lens.cpu())]

        return {"val_loss": total / max(n, 1), "val_wer": error_rate(hyps, refs, "word"),
                "val_cer": error_rate(hyps, refs, "char"), "example": (hyps[0], refs[0])}

    # ---- the whole run ------------------------------------------------------------------

    def fit(self, on_epoch_end: Callable[[Trainer, dict, bool], None] | None = None) -> History:
        """Train from self.epoch (0, or wherever a resumed run left off) to tcfg.epochs.

        After every epoch: record it in History, call on_epoch_end(trainer, record, is_best)
        (ASR.fit saves checkpoints there), and stop early if val WER hasn't improved for
        early_stop_patience epochs.
        """
        patience = self.tcfg.early_stop_patience

        while self.epoch < self.tcfg.epochs:
            t0 = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]

            train_loss = self.train_epoch()
            val = self.validate()
            self.epoch += 1

            # Record the epoch, and whether it's the best so far
            record = {"epoch": self.epoch, "train_loss": train_loss, "val_loss": val["val_loss"],
                      "val_wer": val["val_wer"], "val_cer": val["val_cer"], "lr": lr,
                      "seconds": time.perf_counter() - t0}
            self.history.epochs.append(record)
            is_best = val["val_wer"] < self.history.best_val_wer
            if is_best:
                self.history.best_val_wer = val["val_wer"]
                self.history.best_epoch = self.epoch

            hyp, ref = val["example"]
            print(f"epoch {self.epoch:3d}/{self.tcfg.epochs} | lr {lr:.1e} | train {train_loss:.3f} | "
                  f"val {val['val_loss']:.3f} | WER {val['val_wer']:.1%} CER {val['val_cer']:.1%} | "
                  f"{record['seconds']:.0f}s{'  *best' if is_best else ''}\n    hyp: {hyp[:90]}\n    ref: {ref[:90]}")

            if on_epoch_end is not None:
                on_epoch_end(self, record, is_best)

            # Early stopping: no improvement for `patience` epochs
            if patience is not None and self.epoch - self.history.best_epoch >= patience:
                print(f"No val WER improvement for {patience} epochs; stopping.")
                break

        return self.history

    # ---- resuming -----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Everything needed to resume exactly: weights, optimizer moments, LR schedule position,
        scaler, epoch, history and every random generator."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "history": dataclasses.asdict(self.history),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch = state["epoch"]
        self.history = History(**state["history"])

        # Restore the random generators, so dropout and SpecAugment continue the same sequence
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])