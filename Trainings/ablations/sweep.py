"""
CARFAC design sweep, one command per run: Conformer-CTC on 3 shards, one CARFAC parameter changed.

Every option defaults to the sweep's reference ("default") configuration, so a run only names what
it changes:

    python Trainings/carfac_sweep/sweep.py                          # The reference CARFAC
    python Trainings/carfac_sweep/sweep.py --velocity-scale 0.05    # One change
    python Trainings/carfac_sweep/sweep.py --open-loop              # AGC off
    python Trainings/carfac_sweep/sweep.py --frontend logmel        # Log-mel baseline, same recipe
    python Trainings/carfac_sweep/sweep.py --max-zeta 0.5 --dry-run # Check the design, don't train

Launch from the repo root (the workspace is the folder you launch from, or --root / $ASRLIB_ROOT).
On Quest, submit runs with Trainings/carfac_sweep/submit_sweep.sh, which wraps this in sbatch jobs.

The model, training recipe and data are fixed in the FIXED block below and identical for every run,
so every run shares one controlled hash and compare_models.py can compare any two of them.
Selection is on dev-clean (clean, white noise at 10 and 5 dB), never on test-clean.

With every option at its default, the configuration is identical to Trainings/basic/main.py's
CARFAC (same config hash, same caches).

Author: Eric Oliveira
Date: 11/04/2026

* Written with the help of Claude *

"""

import argparse
import math
import os
import sys
from pathlib import Path

from asrlib import ASR, ASRConfig, CLEAN, CarfacConfig, Condition, DataConfig, LogMelConfig, MFCCConfig, ModelConfig, TrainConfig

# Print immediately, line by line, even into a Slurm log
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# FIXED: identical for every sweep run
# =============================================================================

# Encoder size: the stem (arch) is an option below, for the stem ablation
MODEL_KW = dict(d_model=256, n_heads=4, n_layers=12, d_ff=1024, dropout=0.1)

TRAIN_KW = dict(
    peak_lr=5e-4,
    warmup_steps=10_000,
    weight_decay=0.01,
    max_frames_per_batch=35_000,
    grad_clip=1.0,
    amp="auto",
    specaugment={"freq_masks": 2, "freq_width": 27, "time_masks": 2, "time_width": 50,
                 "scale_freq_to_n_feats": True},
    train_conditions=(CLEAN,),
    n_noisy_variants=0,
    early_stop_patience=None,
    num_workers=6,                           # Match --cpus-per-task=8 in the sbatch script
)
NORM = "global"

# Dev-clean conditions used to rank the runs
SELECTION_CONDITIONS = (CLEAN, Condition("white", 10), Condition("white", 5))

# CARFAC defaults (commit 77d8a91) that the options below start from
AGC_TIME_CONSTANTS = (0.002, 0.008, 0.032, 0.128)
IHC_TAU_LPF = 8e-05


# =============================================================================
# Options: the sweep's reference values are the defaults
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CARFAC design sweep (one change per run)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    run = p.add_argument_group("run")
    run.add_argument("--frontend", choices=("carfac", "logmel", "mfcc"), default="carfac",
                     help="logmel / mfcc = baselines with the same recipe (CARFAC options are ignored)")
    run.add_argument("--arch", choices=("conformer_ctc", "conformer_ctc_conv2d"), default="conformer_ctc",
                     help="stem ablation: conformer_ctc = Conv1d stem, conformer_ctc_conv2d = Conv2d stem")
    run.add_argument("--subsample-channels", type=int, default=128,
                     help="Conv2d stem width (48 ≈ the Conv1d stem's parameter count); the Conv1d stem ignores it")
    run.add_argument("--pooling", choices=("hann25", "box10"), default="hann25", help="CARFAC frame pooling")
    run.add_argument("--f-max", type=float, default=None,
                     help="log-mel top filter edge in Hz (7600 = CARFAC's range); default Nyquist")
    run.add_argument("--shards", type=int, default=3, help="LibriSpeech train.100 shards (~7 h each); 0 = all 14")
    run.add_argument("--epochs", type=int, default=100)
    run.add_argument("--seed", type=int, default=30)
    run.add_argument("--root", default=os.environ.get("ASRLIB_ROOT", str(Path().resolve())),
                     help="workspace folder (default: $ASRLIB_ROOT, else the current folder)")
    run.add_argument("--device", default=None, help="cuda / cpu (default: best available)")
    run.add_argument("--prepare", action="store_true", help="only download the data, then exit (login node)")
    run.add_argument("--dry-run", action="store_true",
                     help="build the config and the CARFAC design, print them, and exit (no data, no training)")

    car = p.add_argument_group("CARFAC cascade (car_params)")
    car.add_argument("--velocity-scale", type=float, default=0.1, help="fast compression strength")
    car.add_argument("--v-offset", type=float, default=0.04)
    car.add_argument("--min-zeta", type=float, default=0.10, help="damping at low levels (sharp tuning)")
    car.add_argument("--max-zeta", type=float, default=0.35, help="damping at high levels (broad tuning)")
    car.add_argument("--first-pole-theta", type=float, default=0.95,
                     help="highest pole, in units of π (0.95 → 7.6 kHz, 0.85 → 6.8 kHz)")
    car.add_argument("--zero-ratio", type=float, default=math.sqrt(2))
    car.add_argument("--high-f-damping-compression", type=float, default=0.5)
    car.add_argument("--erb-per-step", type=float, default=0.42, help="channel spacing (0.42 → 80 channels)")
    car.add_argument("--min-pole-hz", type=float, default=30)

    agc = p.add_argument_group("CARFAC AGC (agc_params) and hair cell (ihc_params)")
    agc.add_argument("--open-loop", action="store_true", help="AGC off")
    agc.add_argument("--agc-tc-scale", type=float, default=1.0,
                     help="multiply all AGC time constants (0.5 = faster adaptation; needs --agc-decimation 3 2 2 2)")
    agc.add_argument("--agc-decimation", type=int, nargs=4, default=(6, 2, 2, 2), metavar="D",
                     help="AGC decimation per stage (6 2 2 2 keeps the 3-tap smoother the JAX port needs)")
    agc.add_argument("--tau-lpf", type=float, default=IHC_TAU_LPF, help="hair-cell low-pass time constant (s)")
    agc.add_argument("--post", choices=("log", "cuberoot", "none"), default="log",
                     help="compression after the cochlea (same cache for every choice)")
    return p


def build_frontend(a: argparse.Namespace):
    """The front-end config from the options. Only non-default AGC/IHC entries are passed, so the
    reference run's config (and hash, and caches) matches Trainings/basic/main.py exactly."""
    if a.frontend == "logmel":
        return LogMelConfig(n_mels=80) if a.f_max is None else LogMelConfig(n_mels=80, f_max=a.f_max)
    if a.frontend == "mfcc":
        return MFCCConfig()

    car_params = {
        "velocity_scale": a.velocity_scale, "v_offset": a.v_offset,
        "min_zeta": a.min_zeta, "max_zeta": a.max_zeta,
        "first_pole_theta": a.first_pole_theta * math.pi, "zero_ratio": a.zero_ratio,
        "high_f_damping_compression": a.high_f_damping_compression,
        "erb_per_step": a.erb_per_step, "min_pole_hz": a.min_pole_hz,
        "erb_break_freq": 165.3, "erb_q": 1000 / (24.7 * 4.37),
        "ac_corner_hz": 20, "high_f_factor": 0.0,
    }
    agc_params = {"decimation": tuple(a.agc_decimation)}
    if a.agc_tc_scale != 1.0:
        agc_params["time_constants"] = tuple(t * a.agc_tc_scale for t in AGC_TIME_CONSTANTS)
    ihc_params = {} if a.tau_lpf == IHC_TAU_LPF else {"tau_lpf": a.tau_lpf}

    return CarfacConfig(car_params=car_params, agc_params=agc_params, ihc_params=ihc_params,
                        open_loop=a.open_loop, post=a.post, pooling=a.pooling)


def changed_options(a: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """'velocity_scale=0.05, open_loop=True', or 'reference' when nothing changed: a readable run label."""
    skip = {"root", "device", "prepare", "dry_run"}
    diffs = []
    for key, value in vars(a).items():
        default = parser.get_default(key)
        # nargs options come back as lists, their defaults are tuples: compare as tuples
        same = tuple(value) == tuple(default) if isinstance(value, (list, tuple)) else value == default
        if key not in skip and not same:
            diffs.append(f"{key}={' '.join(map(str, value)) if isinstance(value, (list, tuple)) else value}")
    return ", ".join(diffs) or "reference"


def main():
    parser = build_parser()
    a = parser.parse_args()
    frontend = build_frontend(a)
    data = DataConfig(corpus="librispeech", train_split="train.100", n_train_shards=a.shards or None,
                      val_split="validation", test_split="test", source="auto", max_train_seconds=None)
    cfg = ASRConfig(frontend=frontend, model=ModelConfig(arch=a.arch, subsample_channels=a.subsample_channels, **MODEL_KW), data=data, train=TrainConfig(epochs=a.epochs, **TRAIN_KW),
                    norm=NORM, tokenizer="char", seed=a.seed)

    label = changed_options(a, parser)
    print(f"RUN  {label}\n  config {cfg.hash()} | controlled {cfg.controlled_hash()} | front end {frontend.name}")

    if a.dry_run:
        # Build the CARFAC design without data or training: catches invalid parameters in seconds
        if a.frontend == "carfac":
            from asrlib.frontends import CARFAC
            fe = CARFAC(frontend)
            print(f"  CARFAC OK: {fe.n_feats} channels, {fe.cf_hz[0]:.0f}–{fe.cf_hz[-1]:.0f} Hz | "
                  f"cache {frontend.cache_key()}")
        return

    asr = ASR(cfg, root=a.root, device=a.device)
    print(asr.summary())

    if a.prepare:
        for split in (data.train_split, data.val_split):
            corpus = asr.corpus(split)
            print(f"  {split:12s} {len(corpus):6d} utterances, {corpus.durations().sum() / 16000 / 3600:6.1f} h")
        return

    asr.sanity_check(n_utts=16, steps=200)
    asr.fit()

    # Dev-clean scores (saved in results/<name>/per_utt/validation__*.csv)
    wers = []
    for cond in SELECTION_CONDITIONS:
        r = asr.evaluate(split="validation", condition=cond)
        lo, hi = r.wer_ci
        wers.append(r.wer)
        print(f"  dev {cond.name:12s} WER {r.wer:.2%} (95% CI {lo:.2%}–{hi:.2%})  CER {r.cer:.2%}  "
              f"PER {r.per:.2%}  WPFER {r.wpfer:.2%}")

    # One greppable line per run: grep "^SWEEP" logs/*.out collects the whole sweep
    print(f"SWEEP | {label} | {asr.name} | clean {wers[0]:.2%} | 10dB {wers[1]:.2%} | 5dB {wers[2]:.2%} | "
          f"mean {sum(wers) / len(wers):.2%}")


if __name__ == "__main__":
    main()