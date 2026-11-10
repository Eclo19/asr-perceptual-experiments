"""
Train a new ASR system with another model's exact config, changing only the front end.

Set BASE_MODEL and FRONTEND below, then press Run in VS Code (or `python retrain_frontend.py`)
from the repo root, on a GPU node. The new model has the same model, data, recipe, normalization
and seed as the base, so it shares the base's controlled hash and compare_models.py can pair them.

Results: models/<name>.pt + <name>_meta.json, results/<name>/... (see asrlib.Workspace).
If the job dies, run the same command again: it resumes from the last finished epoch.

Author: Eric Oliveira
Date: 11/09/2026

* Written with the help of Claude *

"""

import math
import sys
from pathlib import Path

from asrlib import ASR, CarfacConfig, LogMelConfig, MFCCConfig

# Print immediately, line by line, even into a Slurm log
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# SETTINGS
# =============================================================================

# Workspace folder (datasets/, features/, models/, results/): the folder you launch from
ROOT = Path().resolve()

# "cuda", "cpu", or None for the best available
DEVICE = None

# The model whose config is copied: a model name (looked up in models/) or a path to its .pt file
BASE_MODEL = "Backup_Models/Transformer_CTC_Models_Good/2026-11-03_carfac-agc-log_librispeech-train.100_clean_s30_64fdc8.pt"

# The new front end (the only change)
logmel_fe = LogMelConfig(n_mels=80)
mfcc_fe = MFCCConfig()
carfac_fe = CarfacConfig(
    car_params={
        "velocity_scale": 0.1, "v_offset": 0.04, "min_zeta": 0.10, "max_zeta": 0.35,
        "first_pole_theta": 0.95 * math.pi, "zero_ratio": math.sqrt(2), "high_f_damping_compression": 0.5,
        "erb_per_step": 0.42, "min_pole_hz": 30, "erb_break_freq": 165.3, "erb_q": 1000 / (24.7 * 4.37),
        "ac_corner_hz": 20, "high_f_factor": 0.0,
    },
    agc_params={"decimation": (6, 2, 2, 2)},
    post="log",
    pooling="hann25",
)

FRONTEND = mfcc_fe


# =============================================================================
# Run
# =============================================================================

def main():
    # Copy the base's config, change the front end: a new, untrained system
    base = ASR.load(BASE_MODEL, root=ROOT, device=DEVICE)
    asr = base.with_changes(frontend=FRONTEND)

    print(f"Base {base.name}\n  config {base.cfg.hash()} | controlled {base.cfg.controlled_hash()}")
    print(f"New  front end {FRONTEND.name}\n  config {asr.cfg.hash()} | controlled {asr.cfg.controlled_hash()}")
    if asr.cfg.controlled_hash() != base.cfg.controlled_hash():
        sys.exit("Controlled hashes differ: the new model would not be comparable to the base")
    print(asr.summary())

    # Quick wiring check (a few seconds), then the real training run
    asr.sanity_check(n_utts=16, steps=200)
    asr.fit()

    # Clean test score; run benchmark_model.py afterwards for the noise and level sweeps
    r = asr.evaluate(split="test")
    lo, hi = r.wer_ci
    print(f"\nTEST {asr.name}\n  WER {r.wer:.2%} (95% CI {lo:.2%}–{hi:.2%})  CER {r.cer:.2%}  "
          f"over {r.n_utts} utterances")


if __name__ == "__main__":
    main()