"""End-to-end smoke test of asrlib on synthetic local data (no Hugging Face)."""

import csv, sys, tempfile, dataclasses
from pathlib import Path
import numpy as np, soundfile as sf, torch

import asrlib
from asrlib import (ASR, ASRConfig, CarfacConfig, Condition, DataConfig, LogMelConfig,
                    ModelConfig, TrainConfig, CLEAN, assert_comparable, paired_bootstrap)
from asrlib.registry import _REGISTRY
from asrlib.models import TransformerCTC

print("registry:", {k: sorted(v) for k, v in _REGISTRY.items()})

# Parameter counts from the explanation
for n_feats, expected in ((65, 5_453_215), (80, 5_551_519)):
    got = sum(p.numel() for p in TransformerCTC(n_feats=n_feats, vocab_size=31).parameters())
    print(f"params n_feats={n_feats}: {got:,} (expected {expected:,})", "OK" if got == expected else "MISMATCH")

# Synthetic corpus: each "word" is a fixed tone pattern, so it's learnable
# resolve(): on macOS /var is a symlink to /private/var, and Workspace stores the resolved path
root = Path(tempfile.mkdtemp()).resolve()
words = {"red": 300, "blue": 500, "green": 700, "black": 900}
rng = np.random.default_rng(0)
def make_split(split, n):
    folder = root / "datasets" / "my_recordings" / split
    folder.mkdir(parents=True)
    rows = []
    for i in range(n):
        ws = list(rng.choice(list(words), size=rng.integers(1, 3)))
        audio = []
        for w in ws:
            t = np.arange(int(0.4 * 16000)) / 16000
            audio += [0.3 * np.sin(2 * np.pi * words[w] * t), np.zeros(1600)]
        sf.write(folder / f"{split}_{i}.wav", np.concatenate(audio).astype(np.float32), 16000)
        rows.append({"path": f"{split}_{i}.wav", "text": " ".join(ws).upper()})
    with (folder / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "text"]); w.writeheader(); w.writerows(rows)
for split, n in (("train", 24), ("validation", 8), ("test", 8)):
    make_split(split, n)

tiny = ModelConfig(d_model=64, n_heads=2, n_layers=2, d_ff=128)
data = DataConfig(corpus="local_folder", train_split="train", val_split="validation", test_split="test")
# Toy-sized recipe: utterances here are 0.5–1 s, so LibriSpeech-sized SpecAugment masks
# (up to 0.4 s each) would erase them; turn it off, use small batches and more epochs
train = TrainConfig(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 150, warmup_steps=50, peak_lr=2e-3,
                    num_workers=0, max_frames_per_batch=600, early_stop_patience=None, specaugment={})
cfg = ASRConfig(frontend=LogMelConfig(), model=tiny, data=data, train=train, seed=1)

asr = ASR(cfg, root=root, device="cpu")
print(asr.summary())
print("featurize:", tuple(asr.featurize(root / "datasets/my_recordings/test/test_0.wav").shape))
print("sanity:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asr.sanity_check(n_utts=4, steps=100).items()})
asr.fit()
print("models/:", sorted(p.name for p in (root / "models").iterdir()))

loaded = ASR.load(asr.name, root=root, device="cpu")
wav = root / "datasets/my_recordings/test/test_1.wav"
print("transcribe:", repr(loaded.transcribe(wav)), "| truth:", asr.corpus("test").texts()[1])
df = loaded.benchmark([CLEAN, Condition("white", 10), Condition(speech_db_spl=45)])
print(df[["condition", "wer", "cer", "wer_lo", "wer_hi"]].to_string(index=False))

# CARFAC through the cache
carfac = asr.with_changes(frontend=CarfacConfig())
print(carfac.summary())
print("precompute:", [str(p.relative_to(root)) for p in carfac.precompute_features()])
carfac.fit()
assert_comparable(loaded, carfac)
a = loaded.evaluate(); b = carfac.evaluate()
print("mel vs carfac WER:", round(a.wer, 3), round(b.wer, 3), "| paired:", paired_bootstrap(a, b))
print("ALL DONE")