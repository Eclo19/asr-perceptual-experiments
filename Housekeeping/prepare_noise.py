"""
Fill <root>/noise/ with the recorded noises used in the evaluation: DEMAND environments and babble.

Run once, on a Quest LOGIN node (it needs internet), from the repo root:

    python Trainings/noise/prepare_noise.py

Set the globals below first, VS Code style. Safe to re-run: finished parts are skipped.

Result:
    noise/cafeteria/PCAFETER_ch01.wav    DEMAND, channel 1 of 16, 16 kHz, about 5 min each
    noise/car/TCAR_ch01.wav
    noise/kitchen/DKITCHEN_ch01.wav
    noise/office/OOFFICE_ch01.wav
    noise/babble/babble_00.wav ...       8 talkers from LibriSpeech dev-other, 60 s each

Each folder name is the noise kind for Condition: Condition("cafeteria", 5), Condition("babble", 0).
NoiseBank splits every recording 60/40 into train/test parts on first use (noise/splits.json), and
evaluation only ever reads the test part.

Author: Eric Oliveira
Date: 11/19/2026

* Written with the help of Claude *

"""

import io
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from asrlib.constants import SAMPLE_RATE

# Print immediately, line by line
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# SETTINGS
# =============================================================================

# Workspace root (holds noise/): $ASRLIB_ROOT, else the folder you launch from
ROOT = Path(os.environ.get("ASRLIB_ROOT", Path().resolve()))

# DEMAND environment -> noise kind (folder name under noise/)
ENVIRONMENTS = {
    "PCAFETER": "cafeteria",
    "TCAR": "car",
    "DKITCHEN": "kitchen",
    "OOFFICE": "office",
}

# DEMAND record on Zenodo (16 kHz zips, 16 single-channel WAV files each)
DEMAND_URL = "https://zenodo.org/records/1227121/files/{env}_16k.zip?download=1"

# Which of the 16 microphones to keep (1 = first)
CHANNEL = 1

# Keep the downloaded zips in noise/_downloads/ (False: delete each one after extracting)
KEEP_ZIPS = False

# Babble: N_TRACKS tracks of TRACK_SECONDS, each the sum of N_TALKERS different speakers
BUILD_BABBLE = True
N_TALKERS = 8
N_TRACKS = 10
TRACK_SECONDS = 60
BABBLE_SEED = 1

# Speakers for babble: LibriSpeech dev-other, whose speakers appear in no other split
# (not in train.100, train.360 or test-clean)
BABBLE_HF_REPO = "openslr/librispeech_asr"
BABBLE_FILES = "other/validation/*.parquet"


# =============================================================================
# DEMAND
# =============================================================================

def download(url: str, dest: Path) -> None:
    """Download url to dest, via a .part file so an interrupted download is never mistaken for a finished one."""
    if dest.exists():
        print(f"  {dest.name} already downloaded")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, open(part, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    print()
    part.rename(dest)


def extract_channel(zip_path: Path, env: str, out_dir: Path, channel: int = CHANNEL) -> Path:
    """Copy one channel's WAV out of a DEMAND zip (e.g. DKITCHEN/ch01.wav) → out_dir/<ENV>_chNN.wav.

    The file is copied byte for byte (no re-encoding); NoiseBank reads it as mono 16 kHz.
    """
    name = f"ch{channel:02d}.wav"
    dest = out_dir / f"{env}_ch{channel:02d}.wav"
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(name)]
        if len(members) != 1:
            raise ValueError(f"Expected one {name} in {zip_path.name}, found {members}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with z.open(members[0]) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)

    info = sf.info(dest)
    print(f"  {dest.relative_to(ROOT)}: {info.duration:.0f} s at {info.samplerate} Hz, {info.channels} channel(s)")
    return dest


def prepare_demand() -> None:
    noise_dir = ROOT / "noise"
    downloads = noise_dir / "_downloads"
    for env, kind in ENVIRONMENTS.items():
        out_dir = noise_dir / kind
        if (out_dir / f"{env}_ch{CHANNEL:02d}.wav").exists():
            print(f"{kind} ({env}): done")
            continue
        print(f"{kind} ({env})")
        zip_path = downloads / f"{env}_16k.zip"
        download(DEMAND_URL.format(env=env), zip_path)
        extract_channel(zip_path, env, out_dir)
        if not KEEP_ZIPS:
            zip_path.unlink()
    if downloads.exists() and not any(downloads.iterdir()):
        downloads.rmdir()


# =============================================================================
# Babble
# =============================================================================

def load_speakers() -> dict[str, list[np.ndarray]]:
    """LibriSpeech dev-other → {speaker id: [f32 mono 16 kHz waveforms]}."""
    from datasets import Audio, load_dataset

    print(f"Loading {BABBLE_HF_REPO} {BABBLE_FILES} (about 300 MB the first time)...")
    rows = load_dataset(BABBLE_HF_REPO, data_files={"data": BABBLE_FILES}, split="data",
                        cache_dir=str(ROOT / "datasets" / "hf_cache"), verification_mode="no_checks")
    rows = rows.cast_column("audio", Audio(decode=False))

    speakers: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz LibriSpeech audio, got {sr} Hz")
        # LibriSpeech ids are <speaker>-<chapter>-<utterance>
        speakers[row["id"].split("-")[0]].append(audio[:, 0])
    print(f"  {sum(len(v) for v in speakers.values())} utterances from {len(speakers)} speakers")
    return dict(speakers)


def talker_stream(utterances: list[np.ndarray], n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """One speaker talking continuously for n_samples: their utterances in random order, looped,
    starting at a random offset → f32[n_samples] at unit RMS."""
    order = rng.permutation(len(utterances))
    stream = np.concatenate([utterances[i] for i in order])
    # Loop if the speaker has less audio than the track, then start anywhere
    reps = int(np.ceil(2 * n_samples / len(stream)))
    stream = np.tile(stream, reps)
    start = int(rng.integers(len(stream) - n_samples))
    out = stream[start:start + n_samples].astype(np.float64)
    return out / np.sqrt(np.mean(out ** 2))


def babble_track(speakers: dict[str, list[np.ndarray]], n_talkers: int, n_samples: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Sum of n_talkers different speakers, each at the same RMS → f32[n_samples], peak 0.9.

    The absolute level doesn't matter: Condition.apply rescales the noise to the requested SNR.
    """
    chosen = rng.choice(sorted(speakers), size=n_talkers, replace=False)
    mix = sum(talker_stream(speakers[s], n_samples, rng) for s in chosen)
    mix -= mix.mean()
    return (0.9 * mix / np.abs(mix).max()).astype(np.float32)


def prepare_babble() -> None:
    out_dir = ROOT / "noise" / "babble"
    paths = [out_dir / f"babble_{i:02d}.wav" for i in range(N_TRACKS)]
    if all(p.exists() for p in paths):
        print("babble: done")
        return

    speakers = load_speakers()
    if len(speakers) < N_TALKERS:
        raise ValueError(f"Need {N_TALKERS} speakers, found {len(speakers)}")

    # One generator for all tracks: same seed, same babble on every machine
    rng = np.random.default_rng(BABBLE_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        track = babble_track(speakers, N_TALKERS, TRACK_SECONDS * SAMPLE_RATE, rng)
        if not path.exists():
            sf.write(path, track, SAMPLE_RATE, subtype="PCM_16")
            print(f"  {path.relative_to(ROOT)}: {TRACK_SECONDS} s, {N_TALKERS} talkers")


# =============================================================================
# Run
# =============================================================================

def main():
    print(f"Workspace: {ROOT}\n")
    prepare_demand()
    if BUILD_BABBLE:
        print()
        prepare_babble()

    # A model's noise kinds must not change after evaluation starts: splits.json fixes the cut once
    print("\nNoise kinds now available:",
          sorted(p.name for p in (ROOT / "noise").iterdir() if p.is_dir() and not p.name.startswith("_")))
    print("If you ever replace a recording, delete its entry in noise/splits.json (or the file is refused).")


if __name__ == "__main__":
    main()