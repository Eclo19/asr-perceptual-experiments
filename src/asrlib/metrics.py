"""
Metrics for the model + front end's performance. This is what quantifies the experiments.
Quantify errors in both words and chars units.

    WER / CER          how many words / characters the recognizer got wrong (corpus level)
    bootstrap_ci       how much that number could move with a different test set of the same kind
    paired_bootstrap   whether front end A really beats front end B on the same utterances

Hypotheses and references must already be normalized by the same Tokenizer.normalize(),
so only real recognition errors count (not case or punctuation).

Author: Eric Oliveira
Date: 10/29/2026

* Written with the help of Claude *

"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


# ==============================
# Edit distance
# ==============================

def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance: the fewest substitutions + deletions + insertions turning hyp into ref.

    Works on any sequences: lists of words for WER, strings (lists of characters) for CER.
    Dynamic programming, keeping only two rows: O(len(ref) · len(hyp)) time, O(len(hyp)) memory.
    """
    # prev[j] = distance between the first i-1 ref units and the first j hyp units
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1,                  # deletion: a ref unit the hyp missed
                         cur[j - 1] + 1,               # insertion: an extra hyp unit
                         prev[j - 1] + (r != h))       # substitution (free if they match)
        prev = cur
    return prev[-1]


def _units(text: str, unit: Literal["word", "char"]) -> list[str]:
    """'he hoped' -> ['he', 'hoped'] for words, ['h', 'e', ' ', 'h', ...] for characters."""
    return text.split() if unit == "word" else list(text)


# ==============================
# Scores
# ==============================

def utterance_errors(utt_ids: Sequence[str], refs: Sequence[str], hyps: Sequence[str]) -> pd.DataFrame:
    """One row per utterance: its error counts and reference lengths, in words and characters.

    Keeping counts (not per-utterance rates) is what makes corpus WER and the bootstrap exact.
    """
    rows = []
    for utt_id, ref, hyp in zip(utt_ids, refs, hyps, strict=True):
        rows.append({
            "utt_id": utt_id, "ref": ref, "hyp": hyp,
            "w_err": edit_distance(_units(ref, "word"), _units(hyp, "word")),
            "w_ref": len(_units(ref, "word")),
            "c_err": edit_distance(ref, hyp),
            "c_ref": len(ref),
        })
    return pd.DataFrame(rows)


def error_rate(hyps: Sequence[str], refs: Sequence[str], unit: Literal["word", "char"] = "word") -> float:
    """Corpus-level WER/CER: total edit distance / total reference length.

    Not the mean of per-utterance rates: long utterances weigh more, as they should.
    """
    errors = sum(edit_distance(_units(r, unit), _units(h, unit)) for h, r in zip(hyps, refs, strict=True))
    total = sum(len(_units(r, unit)) for r in refs)
    return errors / total if total else float("nan")


@dataclass
class EvalResult:
    """Scores for one condition, plus per-utterance rows for significance tests."""
    condition: str
    wer: float
    cer: float
    loss: float
    n_utts: int
    per_utt: Any = None    # pandas DataFrame[utt_id, ref, hyp, w_err, w_ref, c_err, c_ref]
    wer_ci: tuple[float, float] | None = None

    @classmethod
    def from_utterances(cls, condition: str, utt_ids: Sequence[str], refs: Sequence[str],
                        hyps: Sequence[str], loss: float = float("nan"), ci: bool = True) -> EvalResult:
        """Score a whole test set: per-utterance table -> corpus WER/CER -> optional 95% CI."""
        per_utt = utterance_errors(utt_ids, refs, hyps)
        result = cls(condition=condition,
                     wer=float(per_utt.w_err.sum() / per_utt.w_ref.sum()),
                     cer=float(per_utt.c_err.sum() / per_utt.c_ref.sum()),
                     loss=loss, n_utts=len(per_utt), per_utt=per_utt)
        if ci:
            result.wer_ci = bootstrap_ci(result)
        return result

    def row(self) -> dict:
        """Flat dict for a results table (everything except the per-utterance frame)."""
        out = {k: v for k, v in dataclasses.asdict(self).items() if k not in ("per_utt", "wer_ci")}
        out["wer_lo"], out["wer_hi"] = self.wer_ci if self.wer_ci else (None, None)
        return out


# ==============================
# Statistics
# ==============================

def _resampled_wer(w_err: np.ndarray, w_ref: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Corpus WER of each bootstrap sample. idx: (n_samples, n_utts) utterance indices → (n_samples,)."""
    return w_err[idx].sum(axis=1) / w_ref[idx].sum(axis=1)


def bootstrap_ci(result: EvalResult, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """WER confidence interval from resampling utterances (percentile bootstrap).

    Draw n fake test sets of the same size by picking utterances with replacement, compute
    each one's corpus WER, and take the middle (1 - alpha) of those values.
    """
    w_err = result.per_utt.w_err.to_numpy()
    w_ref = result.per_utt.w_ref.to_numpy()

    # Each row of idx is one fake test set
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w_err), size=(n, len(w_err)))                       # (n, n_utts)
    wers = _resampled_wer(w_err, w_ref, idx)                                      # (n,)

    lo, hi = np.quantile(wers, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(a: EvalResult, b: EvalResult, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """(WER_a - WER_b, two-sided p-value) on the same utterances and condition.

    The SAME resampled utterances are scored for both systems in every sample, so the
    test compares A and B on identical material (paired), which is far more sensitive
    than comparing two separate confidence intervals.
    """
    if a.condition != b.condition:
        raise ValueError(f"Different conditions: {a.condition!r} vs {b.condition!r}")

    # Line the two systems up utterance by utterance
    merged = a.per_utt.merge(b.per_utt, on="utt_id", suffixes=("_a", "_b"))
    if len(merged) != len(a.per_utt) or len(merged) != len(b.per_utt):
        raise ValueError("The two results don't cover the same utterances")

    w_ref = merged.w_ref_a.to_numpy()
    err_a, err_b = merged.w_err_a.to_numpy(), merged.w_err_b.to_numpy()
    observed = err_a.sum() / w_ref.sum() - err_b.sum() / w_ref.sum()

    # Same fake test sets for both systems
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w_ref), size=(n, len(w_ref)))
    deltas = _resampled_wer(err_a, w_ref, idx) - _resampled_wer(err_b, w_ref, idx)   # (n,)

    # Two-sided: how often the resampled difference lands on either side of zero
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(observed), float(min(p, 1.0))


# ==============================
# Training log
# ==============================

@dataclass
class History:
    """One dict per epoch: train_loss, val_loss, val_wer, val_cer, lr, seconds."""
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_wer: float = float("inf")