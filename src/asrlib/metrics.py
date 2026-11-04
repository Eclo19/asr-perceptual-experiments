"""
Metrics for the model + front end's performance. This is what quantifies the experiments.
Quantify errors in both words and chars units.

    WER / CER          how many words / characters the recognizer got wrong (corpus level)
    S / D / I          the word errors split into substitutions, deletions and insertions
    PER                phoneme error rate: did the model hear the right SOUNDS? A misspelling such as
                       "medisene" for "medicine" is a whole wrong word for WER but only 2 of 7 phonemes
                       for PER. Text → phonemes with g2p_en (CMUdict, plus a neural guess for words not in it)
    WPFER              weighted phone feature error rate: PER with partial credit for near-misses.
                       A substitution costs how far apart the two phones are in articulatory features
                       (panphon's 22 weighted features): /b/ → /p/ (voicing only) is cheap, /b/ → /s/ is
                       expensive. Deletions and insertions cost 1, as in PER, so WPFER ≤ PER always.
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
import functools
import re
import warnings
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


def edit_ops(ref: Sequence, hyp: Sequence) -> tuple[int, int, int]:
    """The same minimum edits as edit_distance, split into (substitutions, deletions, insertions).

    Each DP cell keeps (total, S, D, I) along its best path. On a tie, a substitution is preferred
    over a deletion, and a deletion over an insertion (the usual convention, as in sclite), so
    S + D + I always equals edit_distance(ref, hyp).
    """
    # prev[j] = best (total, S, D, I) for the first i-1 ref units and the first j hyp units
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]                     # j insertions
    for i, r in enumerate(ref, start=1):
        cur = [(i, 0, i, 0)]                                                # i deletions
        for j, h in enumerate(hyp, start=1):
            t, s_, d, n = prev[j - 1]
            miss = int(r != h)
            sub = (t + miss, s_ + miss, d, n)                               # substitution (free on a match)
            t, s_, d, n = prev[j]
            dele = (t + 1, s_, d + 1, n)                                    # ref unit the hyp missed
            t, s_, d, n = cur[j - 1]
            ins = (t + 1, s_, d, n + 1)                                     # extra hyp unit
            # min() returns the FIRST minimum: substitution, then deletion, then insertion
            cur.append(min(sub, dele, ins, key=lambda c: c[0]))
        prev = cur
    _, s_, d, n = prev[-1]
    return s_, d, n


# ==============================
# Phonemes (for PER)
# ==============================

_G2P = None
_G2P_FAILED = False


def _g2p():
    """The g2p_en converter, created once. None (with one warning) if g2p_en can't be used here.

    Install: pip install --no-deps g2p_en && pip install nltk inflect
    (--no-deps for g2p_en only: it lists the old `distance` package, which fails to build and isn't used)
    """
    global _G2P, _G2P_FAILED
    if _G2P is None and not _G2P_FAILED:
        try:
            import nltk
            # CMUdict is all we use from NLTK: download it once if it's missing
            try:
                nltk.data.find("corpora/cmudict")
            except LookupError:
                nltk.download("cmudict", quiet=True)
            from g2p_en import G2p
            _G2P = G2p()
        except Exception as e:                                                # noqa: BLE001
            _G2P_FAILED = True
            warnings.warn(f"PER disabled: g2p_en isn't usable here ({type(e).__name__}: {e}). "
                          "Install with: pip install --no-deps g2p_en && pip install nltk inflect")
    return _G2P


@functools.lru_cache(maxsize=None)
def word_phonemes(word: str) -> tuple[str, ...]:
    """One word → its phonemes without stress marks: "medicine" → (M, EH, D, AH, S, AH, N).

    Known words come from CMUdict (first pronunciation); anything else, misspellings included, from
    g2p_en's neural model (a guess). Stress digits are dropped, so AH0 and AH1 are the same sound.
    Cached: each distinct word is converted once per process.
    """
    g = _g2p()
    prons = g.cmu.get(word)
    seq = prons[0] if prons else g.predict(word)
    return tuple(re.sub(r"\d", "", p) for p in seq)


def phonemes(text: str) -> list[str] | None:
    """A normalized transcript → one phoneme sequence (word boundaries dropped: "as part" and
    "aspart" sound alike). None when g2p_en isn't available."""
    if _g2p() is None:
        return None
    return [p for w in text.split() for p in word_phonemes(w)]


# ==============================
# Phone features (for WPFER)
# ==============================

# g2p_en's 39 ARPAbet phonemes (stress removed) → IPA, as panphon reads it. AH is mapped to schwa,
# its most common value (unstressed); ER to the r-coloured vowel ɜ˞. Affricates and diphthongs are
# two IPA segments: their feature vectors are averaged, so each phone stays one unit, aligned with PER.
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ",
    "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜ˞", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h",
    "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ",
    "UH": "ʊ", "UW": "u", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

_PHONE_TABLE = None
_PHONE_TABLE_FAILED = False


def _phone_table():
    """(phone → row index, substitution-cost matrix) built once from panphon; None if panphon isn't usable.

    Substitution cost between phones a and b = Σ_f w_f·|a_f − b_f| / (2·Σ_f w_f), over panphon's 22
    weighted features (values −1/0/+1, so the largest possible difference is 2·Σw). It lies in [0, 1]:
    0 for the same sound, 1 for phones opposite in every feature.

    Install: pip install --no-deps panphon && pip install editdistance munkres pyyaml unicodecsv
    (--no-deps for panphon only: it lists `setuptools`, whose pin can clash with other packages)
    """
    global _PHONE_TABLE, _PHONE_TABLE_FAILED
    if _PHONE_TABLE is None and not _PHONE_TABLE_FAILED:
        try:
            import panphon
            ft = panphon.FeatureTable()
            weights = np.asarray(ft.weights, dtype=np.float64)              # (22,): pairs with the first 22 features
            phones = sorted(ARPABET_TO_IPA)
            vecs = []
            for ph in phones:
                segs = ft.word_to_vector_list(ARPABET_TO_IPA[ph], numeric=True)
                if not segs:
                    raise ValueError(f"panphon doesn't know {ARPABET_TO_IPA[ph]!r} (ARPAbet {ph})")
                vecs.append(np.mean(np.asarray(segs, dtype=np.float64), axis=0)[: len(weights)])
            vecs = np.stack(vecs)                                            # (39, 22)
            diff = np.abs(vecs[:, None, :] - vecs[None, :, :]) @ weights      # (39, 39)
            _PHONE_TABLE = ({ph: i for i, ph in enumerate(phones)}, diff / (2 * weights.sum()))
        except Exception as e:                                                # noqa: BLE001
            _PHONE_TABLE_FAILED = True
            warnings.warn(f"WPFER disabled: panphon isn't usable here ({type(e).__name__}: {e}). Install with: "
                          "pip install --no-deps panphon && pip install editdistance munkres pyyaml unicodecsv")
    return _PHONE_TABLE


def weighted_phone_distance(ref: Sequence[str], hyp: Sequence[str]) -> float | None:
    """Edit distance between two phoneme sequences where a substitution costs the phones' feature
    distance (0–1, see _phone_table) and a deletion or insertion costs 1. None without panphon.

    Same dynamic programming as edit_distance, one row at a time with NumPy. Within a row the
    insertion term chains left to right (cur[j] depends on cur[j-1]); that chain is a running
    minimum: cur[j] = min_k≤j (base[k] − k) + j, computed with np.minimum.accumulate.
    """
    table = _phone_table()
    if table is None:
        return None
    index, cost = table
    r = np.array([index[p] for p in ref], dtype=np.int64)
    h = np.array([index[p] for p in hyp], dtype=np.int64)
    if len(r) == 0 or len(h) == 0:
        return float(len(r) + len(h))                                      # Only deletions or insertions

    cols = np.arange(len(h) + 1, dtype=np.float64)
    prev = cols.copy()                                                      # Row 0: j insertions
    for i, ri in enumerate(r, start=1):
        base = np.empty(len(h) + 1)
        base[0] = i                                                         # Column 0: i deletions
        base[1:] = np.minimum(prev[1:] + 1.0,                               # Deletion of ref[i-1]
                              prev[:-1] + cost[ri, h])                      # Substitution (0 on a match)
        prev = np.minimum.accumulate(base - cols) + cols                    # Insertions, chained along the row
    return float(prev[-1])


# ==============================
# Scores
# ==============================

def _error_counts(ref: str, hyp: str, with_phonemes: bool = True) -> dict:
    """All per-utterance counts for one (ref, hyp) pair."""
    ref_w, hyp_w = _units(ref, "word"), _units(hyp, "word")
    s_, d, n = edit_ops(ref_w, hyp_w)
    out = {"w_err": s_ + d + n, "w_ref": len(ref_w), "w_sub": s_, "w_del": d, "w_ins": n,
           "c_err": edit_distance(ref, hyp), "c_ref": len(ref)}

    # PER and WPFER: NaN when g2p_en / panphon aren't available, so scoring never fails because of them
    ref_p = phonemes(ref) if with_phonemes else None
    if ref_p is None:
        out["p_err"], out["p_ref"], out["pf_err"] = np.nan, np.nan, np.nan
    else:
        hyp_p = phonemes(hyp)
        out["p_err"], out["p_ref"] = edit_distance(ref_p, hyp_p), len(ref_p)
        wd = weighted_phone_distance(ref_p, hyp_p)
        out["pf_err"] = np.nan if wd is None else wd
    return out


def utterance_errors(utt_ids: Sequence[str], refs: Sequence[str], hyps: Sequence[str],
                     with_phonemes: bool = True) -> pd.DataFrame:
    """One row per utterance: error counts and reference lengths, in words, characters and phonemes.

    Columns: utt_id, ref, hyp, w_err, w_ref, w_sub, w_del, w_ins, c_err, c_ref, p_err, p_ref, pf_err
    (pf_err: the feature-weighted phone distance, a float; WPFER = Σ pf_err / Σ p_ref).
    Keeping counts (not per-utterance rates) is what makes corpus rates and the bootstrap exact.
    """
    rows = [{"utt_id": utt_id, "ref": ref, "hyp": hyp, **_error_counts(ref, hyp, with_phonemes)}
            for utt_id, ref, hyp in zip(utt_ids, refs, hyps, strict=True)]
    return pd.DataFrame(rows)


def add_error_columns(per_utt: pd.DataFrame) -> pd.DataFrame:
    """Fill in S/D/I and phoneme columns missing from an older per-utterance table (e.g. a CSV
    written before PER existed), recomputing them from its ref and hyp text. No model needed."""
    needed = {"w_sub", "w_del", "w_ins", "p_err", "p_ref", "pf_err"}
    if needed.issubset(per_utt.columns) and per_utt.p_err.notna().all() and per_utt.pf_err.notna().all():
        return per_utt
    df = per_utt.copy()
    df["ref"], df["hyp"] = df.ref.fillna("").astype(str), df.hyp.fillna("").astype(str)
    counts = pd.DataFrame([_error_counts(r, h) for r, h in zip(df.ref, df.hyp)], index=df.index)
    for col in counts.columns:
        df[col] = counts[col]
    return df


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
    per_utt: Any = None    # pandas DataFrame: see utterance_errors() for the columns
    wer_ci: tuple[float, float] | None = None
    per: float = float("nan")         # Phoneme error rate (NaN without g2p_en)
    wpfer: float = float("nan")       # Weighted phone feature error rate (NaN without g2p_en or panphon)
    sub_rate: float = float("nan")    # Substitutions / reference words
    del_rate: float = float("nan")    # Deletions / reference words
    ins_rate: float = float("nan")    # Insertions / reference words (sub + del + ins = WER)

    @classmethod
    def from_utterances(cls, condition: str, utt_ids: Sequence[str], refs: Sequence[str],
                        hyps: Sequence[str], loss: float = float("nan"), ci: bool = True) -> EvalResult:
        """Score a whole test set: per-utterance table -> corpus WER/CER -> optional 95% CI."""
        per_utt = utterance_errors(utt_ids, refs, hyps)
        result = cls.from_per_utt(condition, per_utt, loss=loss)
        if ci:
            result.wer_ci = bootstrap_ci(result)
        return result

    @classmethod
    def from_per_utt(cls, condition: str, per_utt: pd.DataFrame, loss: float = float("nan"),
                     ci: bool = False) -> EvalResult:
        """Corpus scores from an existing per-utterance table (e.g. a saved CSV)."""
        n_words = per_utt.w_ref.sum()
        result = cls(condition=condition,
                     wer=float(per_utt.w_err.sum() / n_words),
                     cer=float(per_utt.c_err.sum() / per_utt.c_ref.sum()),
                     loss=loss, n_utts=len(per_utt), per_utt=per_utt)
        if {"w_sub", "w_del", "w_ins"}.issubset(per_utt.columns):
            result.sub_rate = float(per_utt.w_sub.sum() / n_words)
            result.del_rate = float(per_utt.w_del.sum() / n_words)
            result.ins_rate = float(per_utt.w_ins.sum() / n_words)
        if "p_err" in per_utt.columns and per_utt.p_err.notna().all():
            result.per = float(per_utt.p_err.sum() / per_utt.p_ref.sum())
        if "pf_err" in per_utt.columns and per_utt.pf_err.notna().all():
            result.wpfer = float(per_utt.pf_err.sum() / per_utt.p_ref.sum())
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
    best_val_loss: float = float("inf")           # tie-breaker when two epochs have the same WER