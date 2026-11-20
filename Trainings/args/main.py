"""
Train one ASR system, configured from the command line: front end, data, model, recipe, levels.

Every option defaults to the train.100 Conformer-CTC setup used so far (Trainings/basic/main.py and
the ablations), so a run only names what it changes:

    python Trainings/args/main.py --frontend carfac                    # CARFAC, train.100, clean
    python Trainings/args/main.py --frontend logmel --recipe train360  # Log-mel, train.360 recipe
    python Trainings/args/main.py --frontend mfcc --level-range 45 85 --variants 3 --mix sample
    python Trainings/args/main.py --frontend carfac --dry-run          # Print hashes, don't train

With the defaults, the configs are identical to the earlier train.100 models (same hashes):
CARFAC 0b1a55, log-mel 44568e, MFCC 9c4af7 (controlled a8c0bf).

Launch from the repo root.

Author: Eric Oliveira
Date: 11/19/2026

* Written with the help of Claude *

"""

import argparse
import math
import os
import sys
from pathlib import Path

from asrlib import (ASR, ASRConfig, CLEAN, CarfacConfig, Condition, DataConfig, LogMelConfig, MFCCConfig,
                    ModelConfig, TrainConfig)

# Print immediately, line by line, even into a Slurm log
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# Defaults: the front ends, model and recipes used so far
# =============================================================================

# CARFAC reference design (Table "Reference CARFAC configuration")
CAR_PARAMS = {
    "velocity_scale": 0.1,                   # Fast compression strength
    "v_offset": 0.04,                        # Offset gives a quadratic part
    "min_zeta": 0.10,                        # Damping at low levels (sharp tuning)
    "max_zeta": 0.35,                        # Damping at high levels (broad tuning)
    "first_pole_theta": 0.95 * math.pi,      # Highest pole: 0.95π → 7.6 kHz at 16 kHz
    "zero_ratio": math.sqrt(2),              # How far each zero sits above its pole
    "high_f_damping_compression": 0.5,       # 0 to 1 to compress zeta
    "erb_per_step": 0.42,                    # Channel spacing: 0.42 → 80 channels
    "min_pole_hz": 30,
    "erb_break_freq": 165.3,                 # Greenwood map's break frequency
    "erb_q": 1000 / (24.7 * 4.37),           # Glasberg and Moore's high-cf ratio
    "ac_corner_hz": 20,                      # AC coupling corner
    "high_f_factor": 0.0,
}

# AGC decimation (6, 2, 2, 2): the 3-tap smoother the JAX port needs (default (8, 2, 2, 2) is refused)
AGC_PARAMS = {"decimation": (6, 2, 2, 2)}

# Recipes: one per training set. --peak-lr, --warmup, --epochs, ... override single values
RECIPES = {
    # Every train.100 Conformer-CTC so far
    "train100": dict(split="train.100", peak_lr=5e-4, warmup_steps=10_000, masks=2, early_stop=None),
    # train.360: longer warmup and lower peak (the 5e-4 / 10k recipe didn't converge for CARFAC), more masks
    "train360": dict(split="train.360", peak_lr=4e-4, warmup_steps=30_000, masks=4, early_stop=10),
}

# Dev-clean conditions scored after training (the full sweep is Evaluations/basic/benchmark_model.py)
DEV_CONDITIONS = (CLEAN, Condition("white", 10), Condition("white", 5))


# =============================================================================
# Options
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one ASR system",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    fe = p.add_argument_group("front end and model")
    fe.add_argument("--frontend", choices=("logmel", "mfcc", "carfac"), default="logmel")
    fe.add_argument("--arch", choices=("conformer_ctc", "conformer_ctc_conv2d", "transformer_ctc",
                                       "transformer_ctc_conv1d"), default="conformer_ctc")

    rec = p.add_argument_group("recipe (defaults come from --recipe; set any of these to override)")
    rec.add_argument("--recipe", choices=tuple(RECIPES), default="train100")
    rec.add_argument("--shards", type=int, default=None, help="train.100/360 shards to use; default all")
    rec.add_argument("--epochs", type=int, default=100)
    rec.add_argument("--peak-lr", type=float, default=None)
    rec.add_argument("--warmup", type=int, default=None, help="warmup steps")
    rec.add_argument("--masks", type=int, default=None, help="SpecAugment frequency and time masks (each)")
    rec.add_argument("--early-stop", type=int, default=None, help="patience in epochs; default from the recipe")
    rec.add_argument("--seed", type=int, default=30)
    rec.add_argument("--norm", choices=("global", "utt_cmvn"), default="global")

    lvl = p.add_argument_group("training conditions")
    lvl.add_argument("--level-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                     help="clean speech at a random level from LOW to HIGH dB SPL per utterance "
                          "(default: every utterance at 65 dB SPL)")
    lvl.add_argument("--variants", type=int, default=0, help="draws per utterance (each cached for CARFAC)")
    lvl.add_argument("--mix", choices=("concat", "sample"), default="concat",
                     help="sample: one draw per utterance per epoch (epoch length as clean)")

    run = p.add_argument_group("run")
    run.add_argument("--root", default=os.environ.get("ASRLIB_ROOT", str(Path().resolve())),
                     help="workspace folder (default: $ASRLIB_ROOT, else the current folder)")
    run.add_argument("--device", default=None, help="cuda / cpu (default: best available)")
    run.add_argument("--workers", type=int, default=6, help="DataLoader workers (match --cpus-per-task - 2)")
    run.add_argument("--prepare", action="store_true", help="only download the data, then exit (login node)")
    run.add_argument("--dry-run", action="store_true", help="print the config and hashes, then exit")
    return p


def build_config(a: argparse.Namespace) -> ASRConfig:
    recipe = RECIPES[a.recipe]

    if a.frontend == "logmel":
        frontend = LogMelConfig(n_mels=80)
    elif a.frontend == "mfcc":
        frontend = MFCCConfig()
    else:
        frontend = CarfacConfig(car_params=CAR_PARAMS, agc_params=AGC_PARAMS, ihc_params={},
                                open_loop=False, post="log", pooling="hann25")

    if a.arch.startswith("conformer"):
        model = ModelConfig(arch=a.arch, d_model=256, n_heads=4, n_layers=12, d_ff=1024, dropout=0.1)
    else:
        model = ModelConfig(arch=a.arch, d_model=384, n_heads=6, n_layers=6, d_ff=1536, dropout=0.1,
                            subsample_channels=128, subsample_factor=4)

    data = DataConfig(corpus="librispeech", train_split=recipe["split"], n_train_shards=a.shards,
                      val_split="validation", test_split="test", source="auto", max_train_seconds=None)

    # Clean at 65 dB SPL, or clean at a random level per utterance
    conditions = (Condition(level_range=tuple(a.level_range)),) if a.level_range else (CLEAN,)
    masks = a.masks if a.masks is not None else recipe["masks"]
    train = TrainConfig(
        epochs=a.epochs,
        peak_lr=a.peak_lr if a.peak_lr is not None else recipe["peak_lr"],
        warmup_steps=a.warmup if a.warmup is not None else recipe["warmup_steps"],
        weight_decay=0.01,
        max_frames_per_batch=35_000,
        grad_clip=1.0,
        amp="auto",
        specaugment={"freq_masks": masks, "freq_width": 27, "time_masks": masks, "time_width": 50,
                     "scale_freq_to_n_feats": True},
        train_conditions=conditions,
        n_noisy_variants=a.variants,
        early_stop_patience=a.early_stop if a.early_stop is not None else recipe["early_stop"],
        num_workers=a.workers,
        mix=a.mix,
    )
    return ASRConfig(frontend=frontend, model=model, data=data, train=train, norm=a.norm,
                     tokenizer="char", seed=a.seed)


# =============================================================================
# Run
# =============================================================================

def main():
    a = build_parser().parse_args()
    cfg = build_config(a)
    t = cfg.train
    print(f"CONFIG {cfg.hash()} | controlled {cfg.controlled_hash()} | {cfg.frontend.name} | {cfg.model.arch} | "
          f"{cfg.data.train_split}" + (f" ({cfg.data.n_train_shards} shards)" if cfg.data.n_train_shards else ""))
    print(f"  recipe   peak lr {t.peak_lr:g}, warmup {t.warmup_steps}, {t.epochs} epochs, "
          f"SpecAugment {t.specaugment['freq_masks']}+{t.specaugment['time_masks']} masks, "
          f"early stop {t.early_stop_patience}, seed {cfg.seed}")
    print(f"  training {[c.name for c in t.train_conditions]} | variants {t.n_noisy_variants} | mix {t.mix}")

    if a.dry_run:
        if a.frontend == "carfac":
            # Build the CARFAC design without data or training: catches invalid parameters in seconds
            from asrlib.frontends import CARFAC
            fe = CARFAC(cfg.frontend)
            print(f"  CARFAC OK: {fe.n_feats} channels, {fe.cf_hz[0]:.0f}–{fe.cf_hz[-1]:.0f} Hz")
        return

    asr = ASR(cfg, root=a.root, device=a.device)
    print(asr.summary())

    # Loading each split downloads it into <root>/datasets/hf_cache/ the first time
    if a.prepare:
        for split in (cfg.data.train_split, cfg.data.val_split, cfg.data.test_split):
            corpus = asr.corpus(split)
            print(f"  {split:12s} {len(corpus):6d} utterances, {corpus.durations().sum() / 16000 / 3600:6.1f} h")
        print("Data ready. Now submit the training job (without --prepare).")
        return

    # Quick wiring check (a few seconds), then the real training run
    print("\nRunning sanity checks...\n")
    asr.sanity_check(n_utts=16, steps=200)
    print("\nStarting training...\n")
    asr.fit()

    # Dev-clean scores (model selection), then test-clean at the training level
    print("\nStarting evaluation...\n")
    for cond in DEV_CONDITIONS:
        r = asr.evaluate(split="validation", condition=cond)
        lo, hi = r.wer_ci
        print(f"  dev  {cond.name:12s} WER {r.wer:.2%} (95% CI {lo:.2%}–{hi:.2%})  CER {r.cer:.2%}")
    r = asr.evaluate(split="test")
    lo, hi = r.wer_ci
    print(f"  test {'clean':12s} WER {r.wer:.2%} (95% CI {lo:.2%}–{hi:.2%})  CER {r.cer:.2%}")
    print(f"\nDONE {asr.name}\n  Full noise and level sweep: Evaluations/basic/benchmark_model.py")


if __name__ == "__main__":
    main()