"""
End-to-end smoke test of asrlib on a tiny synthetic corpus. No internet or datasets needed.

It builds 40 short recordings of four "words" (each word is its own tone), then runs the whole
library on them: registry, features, sanity check, training, save/load, transcription,
benchmarking with noise and level conditions, and the CARFAC path through the feature cache.
Every stage checks real expectations, prints PASS / FAIL / SKIP, and the run ends with a summary.

Run from anywhere (laptop, Quest login or compute node, Colab):
    python notes/run_e2e.py
    python notes/run_e2e.py --device cuda --epochs 150 --keep
    python notes/run_e2e.py --skip-carfac --root /scratch/<netid>/e2e

Exit code 0 = every stage passed or was skipped; 1 = at least one stage failed.

Author: Eric Oliveira
Date: 10/30/2026

* Written with the help of Claude *

"""

import argparse
import csv
import platform
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Print immediately, line by line, even when output goes to a Slurm log file
sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (3, 12)


# =============================================================================
# Small stage runner: PASS / FAIL / SKIP per stage, summary at the end
# =============================================================================

class SkipStage(Exception):
    """Raised inside a stage to skip it with a reason (e.g. CARFAC isn't installed)."""


class Runner:
    def __init__(self):
        self.results: dict[str, tuple[str, str, float]] = {}

    def run(self, name: str, fn, needs: tuple[str, ...] = ()):
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")

        # Don't run a stage whose prerequisites didn't pass
        blocked = [n for n in needs if self.results.get(n, ("",))[0] != "PASS"]
        if blocked:
            self.results[name] = ("SKIP", f"needs: {', '.join(blocked)}", 0.0)
            print(f"SKIP  (needs: {', '.join(blocked)})")
            return None

        t0 = time.perf_counter()
        try:
            out = fn()
            status, note = "PASS", ""
        except SkipStage as e:
            out, status, note = None, "SKIP", str(e)
        except Exception as e:
            out, status, note = None, "FAIL", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        seconds = time.perf_counter() - t0

        self.results[name] = (status, note, seconds)
        print(f"{status}  ({seconds:.1f} s){'  ' + note if note else ''}")
        return out

    def summary(self) -> bool:
        print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
        for name, (status, note, seconds) in self.results.items():
            print(f"  {status:4s}  {name:44s} {seconds:7.1f} s  {note[:60]}")
        failed = [n for n, (s, _, _) in self.results.items() if s == "FAIL"]
        print("\nALL STAGES PASSED (or skipped)" if not failed else f"\n{len(failed)} stage(s) FAILED")
        return not failed


def check(condition: bool, message: str) -> None:
    """Like assert, but always active and with a readable message."""
    if not condition:
        raise AssertionError(message)


# =============================================================================
# Arguments and import
# =============================================================================

parser = argparse.ArgumentParser(description="asrlib end-to-end smoke test")
parser.add_argument("--device", default=None, help="cpu, cuda, mps (default: best available)")
parser.add_argument("--epochs", type=int, default=150, help="training epochs per model (default 150)")
parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 = clearest errors)")
parser.add_argument("--root", default=None, help="workspace folder (default: a temporary folder)")
parser.add_argument("--keep", action="store_true", help="keep the workspace folder afterwards")
parser.add_argument("--skip-carfac", action="store_true", help="skip the CARFAC stages")
args = parser.parse_args()

# Import asrlib; if it isn't installed in this env, fall back to the repo's src/ and say so
try:
    import asrlib
    import_note = None
except ModuleNotFoundError as e:
    if e.name != "asrlib":
        sys.exit(f"Missing dependency {e.name!r}. From the repo root, run: pip install -r requirements.txt")
    sys.path.insert(0, str(REPO / "src"))
    import asrlib
    import_note = (f"asrlib is NOT installed in this environment; running it from {REPO / 'src'}.\n"
                   f"      Install it properly with:  cd {REPO} && pip install -r requirements.txt")

import numpy as np
import soundfile as sf
import torch

from asrlib import (ASR, ASRConfig, CarfacConfig, CLEAN, Condition, DataConfig, LogMelConfig,
                    ModelConfig, TrainConfig, assert_comparable, paired_bootstrap)
from asrlib.models import TransformerCTC
from asrlib.registry import _REGISTRY
from asrlib.utils import environment_info, pick_device

runner = Runner()
state: dict = {}


# =============================================================================
# Stages
# =============================================================================

def stage_environment():
    """Where we are running, with what. Nothing here can fail except a wrong Python version."""
    info = environment_info()
    #device = pick_device(args.device)
    device = pick_device("cpu")
    state["device"] = device

    print(f"  asrlib      {Path(asrlib.__file__).parent}  (commit {info.get('asrlib_commit')})")
    if import_note:
        print(f"  NOTE  {import_note}")
    print(f"  python      {platform.python_version()}  ({sys.executable})")
    print(f"  platform    {info['platform']}  | machine {info['machine']}")
    print(f"  torch       {torch.__version__}  | torchaudio {info.get('torchaudio')}")
    print(f"  cuda        {torch.cuda.is_available()}"
          + (f"  | {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}"
             if torch.cuda.is_available() else ""))
    mps = getattr(torch.backends, "mps", None)
    print(f"  mps         {bool(mps and mps.is_available())}")
    print(f"  jax         {info.get('jax')}  | carfac {info.get('carfac')}")
    print(f"  device      {device}  (choose with --device)")

    check(sys.version_info[:2] == EXPECTED_PYTHON,
          f"Python {platform.python_version()} here, but the reference env is "
          f"{'.'.join(map(str, EXPECTED_PYTHON))}: recreate the env with python=3.12")


def stage_registry():
    """Every component is registered, and the model size matches the hand-computed counts."""
    registered = {k: sorted(v) for k, v in _REGISTRY.items()}
    print(f"  registry    {registered}")
    check({"logmel", "mfcc", "carfac"} <= set(registered["frontend"]), "a front end is missing from the registry")
    check("transformer_ctc" in registered["model"], "transformer_ctc is missing from the registry")

    for n_feats, expected in ((65, 5_453_215), (80, 5_551_519)):
        got = sum(p.numel() for p in TransformerCTC(n_feats=n_feats, vocab_size=31).parameters())
        print(f"  params      n_feats={n_feats}: {got:,} (expected {expected:,})")
        check(got == expected, f"parameter count for n_feats={n_feats} is {got:,}, expected {expected:,}")


def stage_corpus():
    """A synthetic corpus: each word is a fixed tone, so a model can genuinely learn it."""
    root = Path(args.root).expanduser() if args.root else Path(tempfile.mkdtemp(prefix="asrlib_e2e_"))

    # resolve(): on macOS /var is a symlink to /private/var, and Workspace stores the resolved path
    root = root.resolve()
    state["root"] = root

    words = {"red": 300, "blue": 500, "green": 700, "black": 900}
    rng = np.random.default_rng(0)
    for split, n in (("train", 24), ("validation", 8), ("test", 8)):
        folder = root / "datasets" / "my_recordings" / split
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            # One or two words per recording: 0.4 s of tone, then 0.1 s of silence, per word
            spoken = list(rng.choice(list(words), size=rng.integers(1, 3)))
            audio = []
            for w in spoken:
                t = np.arange(int(0.4 * 16000)) / 16000
                audio += [0.3 * np.sin(2 * np.pi * words[w] * t), np.zeros(1600)]
            sf.write(folder / f"{split}_{i}.wav", np.concatenate(audio).astype(np.float32), 16000)
            rows.append({"path": f"{split}_{i}.wav", "text": " ".join(spoken).upper()})
        with (folder / "manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "text"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"  workspace   {root}")
    print("  corpus      24 train / 8 validation / 8 test recordings in datasets/my_recordings/")


def make_config(frontend) -> ASRConfig:
    """Tiny model and a toy-sized recipe. Utterances are 0.5–1 s, so LibriSpeech-sized SpecAugment
    masks (up to 0.4 s each) would erase them: SpecAugment is off, batches are small."""
    return ASRConfig(
        frontend=frontend,
        model=ModelConfig(d_model=64, n_heads=2, n_layers=2, d_ff=128),
        data=DataConfig(corpus="local_folder", train_split="train", val_split="validation", test_split="test"),
        train=TrainConfig(epochs=args.epochs, warmup_steps=50, peak_lr=2e-3, num_workers=args.workers,
                          max_frames_per_batch=600, early_stop_patience=None, specaugment={}),
        seed=1,
    )


def stage_build_mel():
    """Build an untrained log-mel system and check the frame-count contract on a real file."""
    asr = ASR(make_config(LogMelConfig()), root=state["root"], device=state["device"])
    print(asr.summary())

    wav = state["root"] / "datasets/my_recordings/test/test_0.wav"
    n_samples = sf.info(wav).frames
    feats = asr.featurize(wav)
    print(f"  featurize   {tuple(feats.shape)} for {n_samples} samples (expected ({n_samples // 160 + 1}, 80))")
    check(tuple(feats.shape) == (n_samples // 160 + 1, 80), "frame-count contract broken for log-mel")
    state["mel"] = asr


def stage_sanity():
    """Overfit 4 utterances: proves lengths, masks, CTC loss and decoding are wired correctly."""
    result = state["mel"].sanity_check(n_utts=4, steps=100)
    print(f"  result      initial loss {result['initial_loss']:.2f} → final {result['final_loss']:.3f}, "
          f"min frames per token {result['min_frames_per_token']:.2f}")
    check(result["min_frames_per_token"] > 1, "fewer output frames than tokens: CTC can't align")
    check(result["final_loss"] < 0.5, f"couldn't overfit 4 utterances (final loss {result['final_loss']:.3f})")


def stage_train_mel():
    """Train, then check the files in models/ and that the model actually learned."""
    asr = state["mel"]
    asr.fit()
    files = sorted(p.name for p in (state["root"] / "models").iterdir())
    print(f"  models/     {files}")
    print(f"  best val    {asr.history.best_val_wer:.1%} WER at epoch {asr.history.best_epoch}")
    check(f"{asr.name}.pt" in files and f"{asr.name}_meta.json" in files, "best model files missing")
    check(not any(f.endswith("_last.pt") for f in files), "resume file wasn't cleaned up after training")
    check(asr.history.best_val_wer < 0.25, f"didn't learn: best val WER {asr.history.best_val_wer:.1%}")


def stage_load_transcribe():
    """Reload from disk, then transcribe every test file and compare with the truth."""
    asr = state["mel"]
    loaded = ASR.load(asr.name, root=state["root"], device=state["device"])
    state["mel_loaded"] = loaded

    wavs = sorted((state["root"] / "datasets/my_recordings/test").glob("*.wav"),
                  key=lambda p: int(p.stem.split("_")[1]))
    hyps = loaded.transcribe(wavs)
    truths = [t.lower() for t in asr.corpus("test").texts()]
    for wav, hyp, truth in list(zip(wavs, hyps, truths))[:4]:
        print(f"  {wav.name:12s} hyp {hyp!r:22s} truth {truth!r}")

    # The reloaded model must transcribe exactly like the in-memory one
    check(hyps == asr.transcribe(wavs), "reloaded model transcribes differently from the in-memory model")
    correct = sum(h == t for h, t in zip(hyps, truths))
    print(f"  exact       {correct}/{len(wavs)} transcripts exactly right")
    check(correct >= len(wavs) // 2, "fewer than half the test transcripts are right")


def stage_benchmark():
    """Score clean speech, white noise and quiet speech; check the per-utterance CSVs."""
    loaded = state["mel_loaded"]
    df = loaded.benchmark([CLEAN, Condition("white", 10), Condition(speech_db_spl=45)])
    print(df[["condition", "wer", "cer", "wer_lo", "wer_hi"]].to_string(index=False))
    print("  note        a clean-trained toy model is expected to fail under noise and at low level")

    csvs = sorted(p.name for p in (state["root"] / "results" / loaded.name / "per_utt").glob("*.csv"))
    print(f"  results/    {csvs}")
    check(len(df) == 3 and len(csvs) == 3, "expected 3 benchmark rows and 3 per-utterance CSVs")
    check(df.loc[df.condition == "clean", "wer"].item() < 0.25, "clean test WER too high")


def stage_carfac():
    """CARFAC through the feature cache: precompute, train, evaluate."""
    if args.skip_carfac:
        raise SkipStage("--skip-carfac")
    try:
        import carfac  # noqa: F401
        import jax  # noqa: F401
    except ImportError as e:
        raise SkipStage(f"{e.name} not installed (pip install -r requirements.txt)")

    carfac_asr = state["mel"].with_changes(frontend=CarfacConfig())
    print(carfac_asr.summary())

    # One cache folder per split (train, validation, test), each finished with a DONE file
    locations = carfac_asr.precompute_features()
    for loc in locations:
        print(f"  cache       {loc.relative_to(state['root'])}  DONE={(loc / 'DONE').exists()}")
    check(len(locations) == 3 and all((loc / "DONE").exists() for loc in locations),
          "expected 3 finished cache folders")

    carfac_asr.fit()
    print(f"  best val    {carfac_asr.history.best_val_wer:.1%} WER at epoch {carfac_asr.history.best_epoch}")
    check(carfac_asr.history.best_val_wer < 0.25, f"CARFAC didn't learn: {carfac_asr.history.best_val_wer:.1%}")
    state["carfac"] = carfac_asr


def stage_compare():
    """The thesis comparison in miniature: same everything except the front end."""
    mel, cf = state["mel_loaded"], state["carfac"]
    assert_comparable(mel, cf)
    a, b = mel.evaluate(), cf.evaluate()
    delta, p = paired_bootstrap(a, b)
    print(f"  mel WER {a.wer:.1%}  vs  CARFAC WER {b.wer:.1%}  | difference {delta:+.1%}, p = {p:.2f}")
    print("  note        on toy data both should be near 0%, so p is expected to be large")


# =============================================================================
# Run
# =============================================================================

runner.run("1. Environment", stage_environment)
runner.run("2. Registry and model size", stage_registry)
runner.run("3. Synthetic corpus", stage_corpus)
runner.run("4. Build log-mel ASR + featurize", stage_build_mel, needs=("3. Synthetic corpus",))
runner.run("5. Sanity check (overfit 4 utterances)", stage_sanity, needs=("4. Build log-mel ASR + featurize",))
runner.run("6. Train log-mel", stage_train_mel, needs=("5. Sanity check (overfit 4 utterances)",))
runner.run("7. Save, load, transcribe", stage_load_transcribe, needs=("6. Train log-mel",))
runner.run("8. Benchmark conditions", stage_benchmark, needs=("7. Save, load, transcribe",))
runner.run("9. CARFAC: cache, train, evaluate", stage_carfac, needs=("6. Train log-mel",))
runner.run("10. Compare mel vs CARFAC", stage_compare,
           needs=("7. Save, load, transcribe", "9. CARFAC: cache, train, evaluate"))

ok = runner.summary()

# Clean up the temporary workspace unless asked to keep it
if "root" in state:
    if args.keep or args.root:
        print(f"\nWorkspace kept at {state['root']}")
    else:
        shutil.rmtree(state["root"], ignore_errors=True)

sys.exit(0 if ok else 1)