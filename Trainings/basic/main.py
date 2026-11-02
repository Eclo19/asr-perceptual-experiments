"""
First experiment: log-mel + Transformer-CTC on clean LibriSpeech.

Every knob is in the KNOBS block below, written out explicitly (even where it equals the default),
so the experiment is fully visible in one place. The values mirror the DL_review notebook's recipe,
which makes this run the regression check: the library should learn the way the notebook did.

Set the RUN globals below, then press Run in VS Code (or `python main.py`):
    1. PREPARE = True  on a Quest LOGIN node (internet): downloads the data once, then exits
    2. PREPARE = False on a GPU node (interactive or sbatch): trains, then tests

Results: models/<name>.pt + <name>_meta.json, results/<name>/... (see asrlib.Workspace).
If the job dies, run the same command again: it resumes from the last finished epoch.

Author: Eric Oliveira
Date: 11/01/2026

* Written with the help of Claude *

"""

import sys
from pathlib import Path

from asrlib import ASR, ASRConfig, CLEAN, DataConfig, LogMelConfig, ModelConfig, TrainConfig, CarfacConfig

# Print immediately, line by line, even into a Slurm log
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# RUN: where and how to run (doesn't change the experiment itself)
# =============================================================================

# Workspace folder (datasets/, features/, noise/, models/, results/). Default: the repo root,
# i.e. the folder holding this file (asr-perceptual-experiments/)
root_path = ""
ROOT = Path().resolve()
print(f"ROOT={ROOT}")

# True: only download the data and exit . False: train and test
PREPARE = False

# "cuda", "cpu", or None for the best available (use "cpu" on the Mac: CTC has no MPS kernel)
DEVICE = None # will try cuda first 


# =============================================================================
# KNOBS: everything that defines this experiment
# =============================================================================

# ---- Front end: 80 log-mel bins, 25 ms windows (n_fft rounded up to 512), 10 ms hop
logmel_fe = LogMelConfig(
    n_mels=80,
    win_ms=25.0,
    log_eps=1e-6,
)

# Default CARFAC
carfac_fe = CarfacConfig(
    post="cuberoot"
)

FRONTEND = carfac_fe

# ---- Data: LibriSpeech "clean", from <root>/datasets/ 
N_TRAIN_SHARDS = None                         # 4 ≈ 28 h (regression check); None = all 14 ≈ 100 h

DATA = DataConfig(
    corpus="librispeech",
    train_split="train.100",
    n_train_shards=N_TRAIN_SHARDS,
    val_split="validation",                # dev-clean: model selection
    test_split="test",                     # test-clean: scored once, at the end
    source="auto",                         # local parquet first, else Hugging Face
    max_train_seconds=None,                # keep every training utterance (≤ ~25 s), as the notebook did
)

# ---- Model: Transformer-CTC (≈ 5.55 M parameters with 80 mel bins)
MODEL = ModelConfig(
    arch="transformer_ctc",
    d_model=384,
    n_heads=6,
    n_layers=4,
    d_ff=1024,
    dropout=0.1,
    subsample_channels=128,
    subsample_factor=4,
)

# ---- Training recipe 
TRAIN = TrainConfig(
    epochs=60,
    peak_lr=1e-3,
    warmup_steps=1000,
    weight_decay=0.01,
    max_frames_per_batch=160_000,          # ≈ 1,600 s of audio per batch ≈ the notebook's 128 utterances
    grad_clip=1.0,
    amp="auto",                            # automatic mix precision choice. 
    specaugment={"freq_masks": 2, "freq_width": 15, "time_masks": 2, "time_width": 40,
                 "scale_freq_to_n_feats": True},
    train_conditions=(CLEAN,),             # clean speech at 65 dB SPL, no noise
    n_noisy_variants=0,
    early_stop_patience=None,              # run all epochs, like the notebook
    num_workers=1,                         # match --cpus-per-task in the Slurm job (leave 2 spare)
)

SEED = 25                                  # the notebook's seed
NORM = "utt_cmvn"                          # per-utterance mean/variance normalization


# =============================================================================
# Run
# =============================================================================

def main():

    # Build Config Object
    cfg = ASRConfig(frontend=FRONTEND, model=MODEL, data=DATA, train=TRAIN, norm=NORM,
                    tokenizer="char", seed=SEED)

    # Instantiate class
    asr_basic = ASR(cfg, root=ROOT, device=DEVICE)
    print(asr_basic.summary())

    # Loading each split downloads it into <root>/datasets/hf_cache/ the first time
    if PREPARE:
        for split in (DATA.train_split, DATA.val_split, DATA.test_split):
            corpus = asr_basic.corpus(split)
            hours = corpus.durations().sum() / 16000 / 3600
            print(f"  {split:12s} {len(corpus):6d} utterances, {hours:6.1f} h")
        print("Data ready. Now run without --prepare on a GPU node.")
        return

    # Quick wiring check (a few seconds), then the real training run
    print("\nRunning sanity checks...\n")
    asr_basic.sanity_check(n_utts=16, steps=200)

    # Train
    print("\nStarting training...\n")
    asr_basic.fit()

    # Score on test-clean (clean speech at the training level), with a 95% bootstrap interval
    print("\nStarting evaluation...\n")
    result = asr_basic.evaluate()
    lo, hi = result.wer_ci
    print(f"\nTEST {asr_basic.name}\n  WER {result.wer:.2%}  (95% CI {lo:.2%}–{hi:.2%})  CER {result.cer:.2%}  "
          f"over {result.n_utts} utterances")


if __name__ == "__main__":
    main()