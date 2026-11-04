"""
CARFAC design sweep: Conformer-CTC on 3 shards of clean LibriSpeech, one CARFAC parameter changed per run.

Every knob is in the KNOBS block below, written out explicitly (even where it equals the default),
so the experiment is fully visible in one place. For the sweep, change ONE value in car_params,
AGC_PARAMS, IHC_PARAMS, OPEN_LOOP or CARFAC_POST per run, and keep everything else fixed.

Set the RUN globals below, then press Run in VS Code (or `python main.py`):
    1. PREPARE = True  on a Quest LOGIN node (internet): downloads the data once, then exits
    2. PREPARE = False on a GPU node (interactive or sbatch): trains, then scores on dev-clean

Selection is on dev-clean (clean, white noise at 10 and 5 dB), never on test-clean: test is kept
for the final, frozen configuration.

Results: models/<name>.pt + <name>_meta.json, results/<name>/... (see asrlib.Workspace).
If the job dies, run the same command again: it resumes from the last finished epoch.

Run	      Change	Channels
default	   none	       80
AGC off	OPEN_LOOP = True	80
AGC slower	time constants x2	80
AGC faster	time constants x0.5, decimation (3, 2, 2, 2)	80
AGC control	decimation (3, 2, 2, 2) only	80
velocity_scale	0.05 / 0.2	80
min_zeta	0.06 / 0.15	80
max_zeta	0.25 / 0.5	80
erb_per_step	0.5 / 0.33	67 / 102
tau_lpf	4e-5 / 1.6e-4	80
cube root	CARFAC_POST = "cuberoot" (reuses the default cache)	80

Author: Eric Oliveira
Date: 11/01/2026 (CARFAC sweep: 11/04/2026)

* Written with the help of Claude *

"""

import sys
from pathlib import Path
import math

from asrlib import ASR, ASRConfig, CLEAN, Condition, DataConfig, LogMelConfig, ModelConfig, TrainConfig, CarfacConfig

# Print immediately, line by line, even into a Slurm log
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# RUN: where and how to run (doesn't change the experiment itself)
# =============================================================================

# Workspace folder (datasets/, features/, noise/, models/, results/): the folder you launch from
ROOT = Path().resolve()
print(f"ROOT={ROOT}")

# True: only download the data and exit . False: train and evaluate
PREPARE = False

# "cuda", "cpu", or None for the best available (use "cpu" on the Mac: CTC has no MPS kernel)
DEVICE = None # will try cuda first


# =============================================================================
# KNOBS: everything that defines this experiment
# =============================================================================

# ---- Front end: 80 log-mel bins, 25 ms windows (n_fft rounded up to 512), 10 ms hop
logmel_fe = LogMelConfig(
    n_mels=80
)

# ---- CARFAC: the sweep changes ONE value per run, relative to these
car_params = {
    "velocity_scale": 0.1,                   # Fast compression strength. Sweep: 0.05, 0.2
    "v_offset": 0.04,                        # Offset gives a quadratic part
    "min_zeta": 0.10,                        # Damping at low levels (sharp tuning). Sweep: 0.06, 0.15
    "max_zeta": 0.35,                        # Damping at high levels (broad tuning). Sweep: 0.25, 0.5
    "first_pole_theta": 0.95 * math.pi,      # Highest pole freq in fs units (0.85π → 6.8 kHz, 0.95π → 7.6 kHz)
    "zero_ratio": math.sqrt(2),              # How far zero is above pole
    "high_f_damping_compression": 0.5,       # 0 to 1 to compress zeta
    "erb_per_step": 0.42,                    # Channel spacing: 0.42 → 80 channels. Sweep: 0.5 (67), 0.33 (~100)
    "min_pole_hz": 30,
    "erb_break_freq": 165.3,                 # Greenwood map's break freq
    "erb_q": 1000 / (24.7 * 4.37),           # Glasberg and Moore's high-cf ratio
    "ac_corner_hz": 20,                      # AC couple at 20 Hz corner
    "high_f_factor": 0.0,                    # Factor to include more high frequency channels
}

# AGC (slow gain control). Keep "decimation": (6, 2, 2, 2) in every run: without it CARFAC's
# default (8, 2, 2, 2) comes back, which the JAX port computes wrongly (asrlib refuses it).
# Time-constant sweep (default 0.002, 0.008, 0.032, 0.128 s):
#   ×2   → {"decimation": (6, 2, 2, 2), "time_constants": (0.004, 0.016, 0.064, 0.256)}
#   ×0.5 → {"decimation": (3, 2, 2, 2), "time_constants": (0.001, 0.004, 0.016, 0.064)}
#          Faster adaptation needs a smaller first decimation (the AGC updates more often), or asrlib
#          refuses the design. That's two changes at once, so add a control run with the default
#          time constants and decimation (3, 2, 2, 2) to separate the two effects.
AGC_PARAMS = {
    "decimation": (6, 2, 2, 2),
}

# Inner hair cell. Empty = CARFAC's defaults. tau_lpf sweep (default 8e-05 s): {"tau_lpf": 4e-05}, {"tau_lpf": 1.6e-04}
IHC_PARAMS = {}

OPEN_LOOP = False                              # True: AGC off ("carfac-noagc"): the key mechanism test
CARFAC_POST = "log"                          # "cuberoot": same cache, only retraining

carfac_fe = CarfacConfig(
    car_params=car_params,
    agc_params=AGC_PARAMS,
    ihc_params=IHC_PARAMS,
    open_loop=OPEN_LOOP,
    post=CARFAC_POST,
    pooling="hann25",
)

FRONTEND = carfac_fe

# ---- Data: LibriSpeech "clean", from <root>/datasets/
N_TRAIN_SHARDS = 3                           # 3 ≈ 21 h (sweep); None = all 14 ≈ 100 h

DATA = DataConfig(
    corpus="librispeech",
    train_split="train.100",
    n_train_shards=N_TRAIN_SHARDS,
    val_split="validation",                  # dev-clean: model selection
    test_split="test",                       # test-clean: kept for the final, frozen configuration
    source="auto",                           # local parquet first, else Hugging Face
    max_train_seconds=None,                  # keep every training utterance (≤ ~35 s)
)

# ---- Model
transformer_ctc_model_conv1d = ModelConfig(
    arch="transformer_ctc_conv1d", # "transformer_ctc", "transformer_ctc_conv1d"
    d_model=384,
    n_heads=6,
    n_layers=6,
    d_ff=1536,
    dropout=0.1,
    subsample_channels=128,
    subsample_factor=4,
)

conformer_ctc_model = ModelConfig(
    arch="conformer_ctc",
    d_model=256, n_heads=4,
    n_layers=12,
    d_ff=1024,
    dropout=0.1
)

MODEL = conformer_ctc_model

# ---- Training recipe (identical for every sweep run)
TRAIN = TrainConfig(
    epochs=100,
    peak_lr=5e-4,
    warmup_steps=10_000,
    weight_decay=0.01,
    max_frames_per_batch=35_000,             # ~220 steps per epoch on 3 shards
    grad_clip=1.0,
    amp="auto",
    specaugment={"freq_masks": 2, "freq_width": 27, "time_masks": 2, "time_width": 50,
                 "scale_freq_to_n_feats": True},
    train_conditions=(CLEAN,),
    n_noisy_variants=0,
    early_stop_patience=None,
    num_workers=6,
)

# Pick seed and normalization
SEED = 30
NORM = "global"

# Dev-clean conditions used to rank the sweep runs
SELECTION_CONDITIONS = (CLEAN, Condition("white", 10), Condition("white", 5))


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
        print("Data ready. Now run with PREPARE = False on a GPU node.")
        return

    # Quick wiring check (a few seconds), then the real training run
    print("\nRunning sanity checks...\n")
    asr_basic.sanity_check(n_utts=16, steps=200)

    # Train
    print("\nStarting training...\n")
    asr_basic.fit()

    # Score on dev-clean: clean, and white noise at 10 and 5 dB (saved in results/<name>/per_utt/validation__*.csv)
    print("\nStarting evaluation (dev-clean)...\n")
    wers = []
    for cond in SELECTION_CONDITIONS:
        r = asr_basic.evaluate(split="validation", condition=cond)
        lo, hi = r.wer_ci
        wers.append(r.wer)
        print(f"  dev {cond.name:12s} WER {r.wer:.2%} (95% CI {lo:.2%}–{hi:.2%})  CER {r.cer:.2%}  "
              f"PER {r.per:.2%}  WPFER {r.wpfer:.2%}")
    print(f"\nSWEEP {asr_basic.name}\n  {asr_basic.cfg.frontend.name}: mean dev WER over "
          f"{len(wers)} conditions {sum(wers) / len(wers):.2%}")


if __name__ == "__main__":
    main()