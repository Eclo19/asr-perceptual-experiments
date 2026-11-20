"""
Front ends across conditions: one line per front end, one figure per noise type, plus LaTeX tables.

Set the globals below and press Run in VS Code (or `python frontend_conditions.py`). Needs no GPU:
it finds each model's meta by config hash in models/ and, recursively, in Backup_Models/, and reads
the per-utterance results in results/<name>/per_utt/ (or the scores saved in the meta).

Outputs go to results/frontend_comparisons/<OUT_NAME>/ (default: <split>_<controlled hash>):
    snr_<noise>.png / .pdf    WER and CER vs SNR for one noise (e.g. snr_babble.pdf); high SNR on the
                              left, as in compare_models.py and benchmark_model.py; dashed lines are
                              each front end's clean score
    snr_all_wer.png / .pdf    every noise side by side (WER), shared y-axis
    level.png / .pdf          WER and CER vs presentation level (clean speech)
    tables.tex                WER and CER tables (rows = conditions grouped by noise, best per row in bold)
    scores.csv                every score used above, one row per front end and condition

Author: Eric Oliveira
Date: 11/05/2026 (one figure per noise: 11/19/2026)

* Written with the help of Claude *

"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # Draw to files: works on Quest without a display
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

# Workspace root (holds models/ and results/): $ASRLIB_ROOT, else the folder you launch from
ROOT = Path(os.environ.get("ASRLIB_ROOT", Path().resolve()))

# Models by config hash, with the label shown in plots and tables (order = legend and column order)
MODELS = {
    "984ff9": "CARFAC",
    "4e680e": "Log-mel",
    "1376d5": "MFCC",
}

# Output folder name under results/frontend_comparisons/. None: "<split>_<controlled hash>"
OUT_NAME = None

# Folders searched for <name>_meta.json, in order; subfolders included. A model found in several
# (e.g. models/ and a backup copy) is the same model: the first one is used
SEARCH_DIRS = ["models", "Backup_Models"]

SPLIT = "test"

# Noises to plot and tabulate, in this order. None: every noise all models were scored on
NOISES = ("white", "car", "cafeteria", "office", "kitchen", "babble")

# SNRs (dB) in the tables (the figures use every SNR all models share)
TABLE_SNRS = (20, 10, 5, 0, -5)

# Presentation levels (dB SPL) in the tables, besides the clean row
TABLE_LEVELS = (45, 55, 75, 85)

# Display names in titles and table rows
NOISE_NAMES = {"white": "White noise", "car": "Car", "cafeteria": "Cafeteria", "office": "Office",
               "kitchen": "Kitchen", "babble": "Babble"}

# Level of the clean condition (speech in training)
TRAIN_LEVEL_DB = 65

# Training data, for the table captions (e.g. "100\,h" or "360\,h")
TRAIN_HOURS = "360\\,h"


# =============================================================================
# Helpers
# =============================================================================

def load(config_hash: str) -> dict:
    """The meta of the model with this config hash, searched recursively in SEARCH_DIRS."""
    hits = []
    for folder in SEARCH_DIRS:
        hits += sorted((ROOT / folder).rglob(f"*_{config_hash}_meta.json"))
    if not hits:
        sys.exit(f"No model with config hash {config_hash} in {SEARCH_DIRS} (subfolders included)")

    # Copies of one model share a file name; different names would mean two different models
    names = {p.name for p in hits}
    if len(names) > 1:
        sys.exit(f"Config hash {config_hash} matches different models: {sorted(names)}")
    print(f"  {config_hash}: {hits[0].relative_to(ROOT)}" + (f" (+{len(hits) - 1} copies)" if len(hits) > 1 else ""))
    return json.loads(hits[0].read_text())


def scores(meta: dict) -> dict[str, dict]:
    """{condition: {wer, cer}} for SPLIT: from the per-utterance CSVs in results/<name>/per_utt/,
    plus any condition only recorded in meta["evaluations"]."""
    from asrlib.metrics import add_error_columns

    out = {}
    for path in sorted((ROOT / "results" / meta["name"] / "per_utt").glob(f"{SPLIT}__*.csv")):
        cond = path.stem.split("__", 1)[1]
        df = add_error_columns(pd.read_csv(path, dtype={"utt_id": str, "ref": str, "hyp": str},
                                           keep_default_na=False))
        out[cond] = {"wer": df.w_err.sum() / df.w_ref.sum(), "cer": df.c_err.sum() / df.c_ref.sum()}
    prefix = f"{SPLIT}/"
    for k, v in meta.get("evaluations", {}).items():
        if k.startswith(prefix):
            out.setdefault(k[len(prefix):], {"wer": v["wer"], "cer": v["cer"]})
    return out


def parse(cond: str) -> dict:
    """'babble@5dB' → noise babble, SNR 5, level 65; 'clean@45dBSPL' → clean, level 45; 'clean' → clean, 65."""
    if cond == "clean":
        return {"noise": None, "snr_db": None, "level": TRAIN_LEVEL_DB}
    m = re.fullmatch(r"clean@(-?\d+(?:\.\d+)?)dBSPL", cond)
    if m:
        return {"noise": None, "snr_db": None, "level": float(m.group(1))}
    # Noisy at the training level only (noisy conditions at other levels are left out)
    m = re.fullmatch(r"([A-Za-z_]\w*)@(-?\d+(?:\.\d+)?)dB", cond)
    if m:
        return {"noise": m.group(1), "snr_db": float(m.group(2)), "level": TRAIN_LEVEL_DB}
    return {"noise": "?", "snr_db": None, "level": None}


def pretty(cond: str) -> str:
    """Row label for the tables."""
    p = parse(cond)
    if p["noise"] is None:
        return f"Clean, {p['level']:g}\\,dB SPL" + (" (training)" if p["level"] == TRAIN_LEVEL_DB else "")
    return f"{NOISE_NAMES.get(p['noise'], p['noise'])}, {p['snr_db']:g}\\,dB"


def snr_axes(ax, df: pd.DataFrame, metric: str, clean: dict, colors: dict, legend: bool) -> None:
    """One noise: metric vs SNR per front end (high SNR on the left), dashed line = clean score."""
    for label, part in df.groupby("frontend", sort=False):
        part = part.sort_values("snr_db")
        ax.plot(part.snr_db, part[metric] * 100, marker="o", ms=4, color=colors[label], label=label)
        if label in clean:
            ax.axhline(clean[label][metric] * 100, color=colors[label], ls="--", lw=1, alpha=0.6)
    ax.set_xticks(sorted(df.snr_db.unique()))
    ax.invert_xaxis()                    # High SNR (easy) on the left, as in compare_models.py
    ax.set(xlabel="SNR (dB)", ylabel=f"{metric.upper()} (%)", ylim=(0, 100))
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(frameon=False, fontsize=8)


def save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


# =============================================================================
# Run
# =============================================================================

def main():
    metas = {label: load(h) for h, label in MODELS.items()}
    controlled = {m["controlled_hash"] for m in metas.values()}
    if len(controlled) > 1:
        print(f"WARNING: models differ in more than front end and seed (controlled hashes {sorted(controlled)})")
    tables = {label: scores(m) for label, m in metas.items()}
    common = set.intersection(*(set(t) for t in tables.values()))

    # What each model has, so a missing benchmark is obvious
    for label, t in tables.items():
        print(f"  {label:10s} {metas[label]['name']}: {len(t)} {SPLIT} conditions")
    print(f"  shared by all: {len(common)}")
    if not common:
        sys.exit(f"No {SPLIT} condition is scored for every model: run benchmark_model.py on the ones with 0 "
                 f"(same CONDITIONS list for all), then run this again")

    # One long table: front end × condition, with the parsed noise, SNR and level
    rows = [{"frontend": label, "condition": c, **parse(c), **tables[label][c]}
            for label in tables for c in sorted(common)]
    df = pd.DataFrame(rows)

    name = OUT_NAME or f"{SPLIT}_{'-'.join(sorted(controlled))}"
    out = ROOT / "results" / "frontend_comparisons" / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scores.csv", index=False)

    colors = {label: f"C{i}" for i, label in enumerate(tables)}
    clean = {label: tables[label]["clean"] for label in tables if "clean" in common}

    # Noises: as listed in NOISES (skipping any not scored for every model), else all shared ones
    shared_noises = [n for n in df.noise.dropna().unique() if n != "?"]
    noises = [n for n in NOISES if n in shared_noises] if NOISES else sorted(shared_noises)
    missing = [n for n in (NOISES or ()) if n not in shared_noises]
    if missing:
        print(f"Not scored for every model (no figure or table rows): {missing}")

    # 1. One figure per noise: WER | CER
    for noise in noises:
        part = df[df.noise == noise]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, metric in zip(axes, ("wer", "cer")):
            snr_axes(ax, part, metric, clean, colors, legend=metric == "wer")
        fig.suptitle(f"{NOISE_NAMES.get(noise, noise)} ({SPLIT}, speech at {TRAIN_LEVEL_DB} dB SPL; "
                     f"dashed: clean)", fontsize=10)
        save(fig, out, f"snr_{noise}")

    # 2. Every noise side by side (WER), shared y-axis
    if noises:
        cols = min(3, len(noises))
        n_rows = -(-len(noises) // cols)
        fig, axes = plt.subplots(n_rows, cols, figsize=(4.2 * cols, 3.4 * n_rows), sharey=True, squeeze=False)
        for i, noise in enumerate(noises):
            ax = axes[i // cols][i % cols]
            snr_axes(ax, df[df.noise == noise], "wer", clean, colors, legend=i == 0)
            ax.set_title(NOISE_NAMES.get(noise, noise), fontsize=10)
        for j in range(len(noises), n_rows * cols):
            axes[j // cols][j % cols].axis("off")
        save(fig, out, "snr_all_wer")

    # 3. Level sweep (clean speech): WER | CER
    levels = df[df.noise.isna() & df.level.notna()]
    if levels.level.nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, metric in zip(axes, ("wer", "cer")):
            for label, part in levels.groupby("frontend", sort=False):
                part = part.sort_values("level")
                ax.plot(part.level, part[metric] * 100, marker="o", ms=4, color=colors[label], label=label)
            ax.axvline(TRAIN_LEVEL_DB, color="gray", lw=0.8, ls=":")
            ax.set(xlabel="Speech level (dB SPL)", ylabel=f"{metric.upper()} (%)", ylim=(0, 100))
            ax.grid(alpha=0.3)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Clean speech at different levels ({SPLIT}; trained at {TRAIN_LEVEL_DB} dB SPL)", fontsize=10)
        save(fig, out, "level")

    # 4. LaTeX tables: clean, then each noise at TABLE_SNRS, then the level sweep; best per row in bold
    groups = [["clean"]]
    groups += [[f"{n}@{s:g}dB" for s in TABLE_SNRS] for n in noises]
    groups += [[f"clean@{lv:g}dBSPL" for lv in TABLE_LEVELS]]
    groups = [[c for c in g if c in common] for g in groups]
    groups = [g for g in groups if g]
    labels = list(tables)
    tex = []
    for metric in ("wer", "cer"):
        tex += [r"\begin{table}[t]", r"\centering", r"\small",
                rf"\caption{{{metric.upper()} (\%) on {SPLIT}-clean per condition, by front end "
                rf"(Conformer-CTC, {TRAIN_HOURS}, one seed). Best per row in bold.}}",
                rf"\label{{tab:frontends_{metric}}}",
                r"\begin{tabular}{l" + "r" * len(labels) + "}", r"\toprule",
                "Condition & " + " & ".join(labels) + r" \\"]
        for g in groups:
            tex.append(r"\midrule")
            for c in g:
                vals = [tables[l][c][metric] * 100 for l in labels]
                best = min(vals)
                cells = [rf"\textbf{{{v:.2f}}}" if abs(v - best) < 1e-9 else f"{v:.2f}" for v in vals]
                tex.append(f"{pretty(c)} & " + " & ".join(cells) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / "tables.tex").write_text("\n".join(tex))

    print(f"Saved to {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()