"""
Define the Workspace class for efficient, safe, and simple file management.

Author: Eric Oliveira 
Date: 10/27/2026

* Written with the help of Claude *

"""

import os
from pathlib import Path

class Workspace:
    """The project folder, with these subfolders:

        <root>/datasets/   speech corpora (local copies, plus the Hugging Face cache in hf_cache/)
        <root>/features/   cached features: <frontend cache key>/<corpus>/<split>/<condition>/
        <root>/noise/      noise recordings: noise/pub/*.wav, noise/babble/*.wav, noise/splits.json
        <root>/models/     <name>.pt + <name>_meta.json for every best model
        <root>/results/    <name>/per_utt/<condition>.csv from evaluate()/benchmark()

    The root is chosen in this order: the `root` argument, then the ASRLIB_ROOT
    environment variable, then the current working directory (where your notebook
    or script runs). It's deliberately NOT the folder this .py file lives in: once
    asrlib is pip-installed from GitHub, that folder is inside site-packages and
    gets wiped on every reinstall.

        Laptop: run from ASR/ folder (or set ASRLIB_ROOT=~/ASR)
        Colab:  Workspace("/content/drive/MyDrive/ASR")
        Quest:  export ASRLIB_ROOT=/projects/<allocation>/thesis_asr
    """

    # Directories to keep track of
    FOLDERS = ("datasets", "features", "noise", "models", "results")

    def __init__(self, root: str | Path | None = None, create: bool = True):
        # Resolve the root: explicit argument > environment variable > current directory
        chosen = root or os.environ.get("ASRLIB_ROOT") or Path.cwd()
        self.root = Path(chosen).expanduser().resolve()

        # Create any missing subfolders, so later code can assume they exist
        if create:
            for name in self.FOLDERS:
                (self.root / name).mkdir(parents=True, exist_ok=True)

    # Path Getters
    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def noise(self) -> Path:
        return self.root / "noise"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"


    def model_paths(self, name: str) -> tuple[Path, Path]:
        """Get model data (weights and metadata) from a model name:

        (<root>/models/<name>.pt, <root>/models/<name>_meta.json)
        
        """
        return self.models / f"{name}.pt", self.models / f"{name}_meta.json"

    def find_models(self, config_hash: str | None = None) -> list[Path]:
        """Meta files in models/, newest first; optionally only those trained from one config."""
        pattern = f"*_{config_hash}_meta.json" if config_hash else "*_meta.json"
        return sorted(self.models.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    def summary(self) -> str:
        """What's in each folder: a quick check that the root is the one you meant."""
        lines = [f"Workspace: {self.root}"]
        for name in self.FOLDERS:
            folder = self.root / name
            if folder.exists():
                lines.append(f"  {name + '/':10s} {len(list(folder.iterdir()))} item(s)")
            else:
                lines.append(f"  {name + '/':10s} missing")

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"