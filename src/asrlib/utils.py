
"""
Helper functions for the ASR class and its workflow. 

Author: Eric Oliveira 
Date: 10/27/2026

* Written with the help of Claude *

"""
from __future__ import annotations  
import random
import sys
import platform
import subprocess
from pathlib import Path
import os

import numpy as np
import torch

from .constants import CARFAC_COMMIT


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA).

    Called once in ASR.__init__, before any weights are built, so the same seed
    gives the same initial weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str | torch.device | None = None) -> torch.device:
    """Explicit choice, else CUDA (Colab/Quest), else MPS (your Mac's GPU), else CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_info() -> dict:
    """Where and with what code a model was trained; goes into every _meta.json."""
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": getattr(torch, "__version__", "?")}

    # Optional libraries: record their versions only if they're installed
    for lib in ("torchaudio", "jax", "carfac", "datasets"):
        try:
            info[lib] = getattr(__import__(lib), "__version__", "installed")
        except ImportError:
            info[lib] = None

    # Which machine: Colab and Slurm (Quest) both leave recognizable traces
    if "google.colab" in sys.modules:
        info["machine"] = "colab"
    elif "SLURM_JOB_ID" in os.environ:
        info["machine"] = f"quest (slurm job {os.environ['SLURM_JOB_ID']})"
    else:
        info["machine"] = platform.node()
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    # The exact asrlib code: the git commit of the folder this file lives in, if it's a repo
    try:
        info["asrlib_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["asrlib_commit"] = None
    info["carfac_commit"] = CARFAC_COMMIT
    return info

