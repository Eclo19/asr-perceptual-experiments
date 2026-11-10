"""
One figure from several compare_models.py runs: ΔWER (A − B) vs SNR, one line per comparison.

Built for the stem question: CARFAC − log-mel under each sub-sampling stem, so you can see whether
the stem changes the front-end ranking. Each line is read from a compare_models.py output folder
(its comparison.csv), with the 95% paired-bootstrap interval as a shaded band.

Set COMPARISONS below and press Run in VS Code (or `python stem_delta_figure.py`). No GPU needed.
Saves <OUT>.png and <OUT>.pdf.

Author: Eric Oliveira
Date: 11/10/2026

* Written with the help of Claude *

"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # Draw to files: works on Quest without a display
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

# Workspace root: $ASRLIB_ROOT, else the folder you launch from
ROOT = Path(os.environ.get("ASRLIB_ROOT", Path().resolve()))

# Line label: compare_models.py output folder (in results/compare/), A = CARFAC, B = log-mel in each
COMPARISONS = {
    "1D stem":                "carfac-agc-log_vs_logmel80_test_a8c0bf_s30",
    "2D stem, 128 filters":   "carfac-agc-log_vs_logmel80_test_487987_s30",
    "2D stem, size-matched":  "carfac-agc-log_vs_logmel80_test_03394c_s31",
}

NOISE = "white"
METRIC = "wer"                              # "wer" (has an interval), or "cer" / "per" / "wpfer" (line only)
TITLE = "Δ = CARFAC − log-mel"
OUT = ROOT / "results" / "compare" / "stem_comparisons" / "stem_carfac_vs_logmel"


# =============================================================================
# Run
# =============================================================================

def main():
    fig, ax = plt.subplots(figsize=(7, 4))
    snr_ticks = set()
    for label, folder in COMPARISONS.items():
        df = pd.read_csv(ROOT / "results" / "compare" / folder / "comparison.csv")

        # Noise points at the training level, plus clean at the right end of the axis
        noisy = df[(df.noise == NOISE) & (df.speech_db_spl == 65)].sort_values("snr_db")
        clean = df[df.condition == "clean"]
        x_clean = noisy.snr_db.max() + 10
        x = list(noisy.snr_db) + ([x_clean] if len(clean) else [])
        part = pd.concat([noisy, clean])
        y = part[f"{METRIC}_delta"] * 100
        snr_ticks.update(noisy.snr_db)

        line, = ax.plot(x, y, marker="o", ms=4, label=label)
        if f"{METRIC}_delta_lo" in part:
            ax.fill_between(x, part[f"{METRIC}_delta_lo"] * 100, part[f"{METRIC}_delta_hi"] * 100,
                            color=line.get_color(), alpha=0.2, lw=0)

    ticks = sorted(snr_ticks)
    ax.set_xticks(ticks + [max(ticks) + 10], [f"{t:g}" for t in ticks] + ["clean"])
    ax.axhline(0, color="gray", lw=1)
    ax.set(xlabel=f"SNR (dB), {NOISE} noise", ylabel=f"Δ{METRIC.upper()} (points)",
           title=f"{TITLE} (below 0 where CARFAC is better)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200)
    print(f"Saved {OUT}.png / .pdf")


if __name__ == "__main__":
    main()