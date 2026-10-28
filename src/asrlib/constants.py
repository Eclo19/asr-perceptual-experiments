"""
Constants for the ASR class. 

Author: Eric Oliveira 
Date: 10/27/2026

* Written with the help of Claude *

"""

from pathlib import Path
from typing import Union

import numpy as np
import torch


# Reference to a specific commit in the google/carfac repo
CARFAC_COMMIT = "77d8a91f24d2f5873783dc0ce767d32778b938bc"

# Sampling and hop
SAMPLE_RATE = 16_000
HOP_LENGTH = 160 # 10 ms

# Anything transcribe() accepts as audio
AudioLike = Union[str, Path, np.ndarray, torch.Tensor]

# Level calibration, CARFAC's convention (CARFAC_Test.m, test_spike_rates: "dbfs + 104" = dB SPL rms):
# a digital RMS of 1.0 (0 dBFS) corresponds to 104 dB SPL, i.e. about 3.2 Pa
FULL_SCALE_DB_SPL = 104.0

# Default speech level for every Condition: normal conversational speech (~ -39 dBFS)
DEFAULT_SPEECH_DB_SPL = 65.0