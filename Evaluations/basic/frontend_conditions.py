"""
Front ends across conditions: WER and CER vs SNR and vs level, one line per front end, plus LaTeX tables.

Set the globals below and press Run in VS Code (or `python frontend_conditions.py`). Needs no GPU:
it finds each model's meta by config hash in models/ and, recursively, in Backup_Models/, and reads
the per-utterance results in results/<name>/per_utt/ (or the scores saved in the meta).

Outputs go to results/compare/frontends_<split>_<controlled hash>/:
    wer_conditions.png / .pdf    WER vs SNR (white noise) | WER vs level, one line per front end
    cer_conditions.png / .pdf    the same for CER
    tables.tex                   WER and CER tables (rows = conditions, best per row in bold)

Author: Eric Oliveira
Date: 11/05/2026

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


# =============================================================================
# SETTINGS
# =============================================================================

# Workspace root (holds models/ and results/): $ASRLIB_ROOT, else the folder you launch from
ROOT = Path(os.environ.get("ASRLIB_ROOT", Path().resolve()))

# Models by config hash, with the label shown in plots and tables (order = legend and column order)
MODELS = {
    "2fd4a1": "CARFAC",
    "7215fa": "Log-mel",
    "19f103": "MFCC",
}

# Folders searched for <name>_meta.json, in order; subfolders included. A model found in several
# (e.g. models/ and a backup copy) is the same model: the first one is used
SEARCH_DIRS = ["models", "Backup_Models"]

SPLIT = "test"

# Conditions in the tables (the plots use every condition all models share)
TABLE_CONDITIONS = ["clean", "white@20dB", "white@10dB", "white@5dB", "white@0dB",
                    "clean@45dBSPL", "clean@55dBSPL", "clean@75dBSPL", "clean@85dBSPL"]

# Level of the clean condition (speech in training)
TRAIN_LEVEL_DB = 65


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
    import pandas as pd
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
            out.setdefault(k[len(prefix):], v)
    return out


def snr_of(cond: str) -> float | None:
    """'white@10dB' → 10; 'clean' → None. Only white noise, so the SNR axis has one noise type."""
    m = re.fullmatch(r"white@(-?\d+(?:\.\d+)?)dB", cond)
    return float(m.group(1)) if m else None


def level_of(cond: str) -> float | None:
    """'clean@45dBSPL' → 45; 'clean' → the training level."""
    if cond == "clean":
        return TRAIN_LEVEL_DB
    m = re.fullmatch(r"clean@(-?\d+(?:\.\d+)?)dBSPL", cond)
    return float(m.group(1)) if m else None


def pretty(cond: str) -> str:
    """Row label for the tables."""
    if cond == "clean":
        return f"Clean ({TRAIN_LEVEL_DB}\\,dB SPL)"
    if (s := snr_of(cond)) is not None:
        return f"White noise, {s:g}\\,dB SNR"
    if (l := level_of(cond)) is not None:
        return f"Clean, {l:g}\\,dB SPL"
    return cond


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

    out = ROOT / "results" / "compare" / f"frontends_{SPLIT}_{'-'.join(sorted(controlled))}"
    out.mkdir(parents=True, exist_ok=True)

    # Figures: one per metric, two panels (SNR, level)
    for metric in ("wer", "cer"):
        fig, (ax_snr, ax_lvl) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for label, t in tables.items():
            snr = sorted((snr_of(c), t[c][metric]) for c in common if snr_of(c) is not None)
            if "clean" in common:
                snr.append((float("inf"), t["clean"][metric]))
            lvl = sorted((level_of(c), t[c][metric]) for c in common if level_of(c) is not None)
            # Clean sits at the right of the SNR axis, as "∞"
            xs = [s for s, _ in snr if s != float("inf")]
            x_clean = (max(xs) + 10) if xs else 0
            ax_snr.plot([x_clean if s == float("inf") else s for s, _ in snr], [v * 100 for _, v in snr],
                        marker="o", ms=4, label=label)
            ax_lvl.plot([l for l, _ in lvl], [v * 100 for _, v in lvl], marker="o", ms=4, label=label)
        ticks = sorted({snr_of(c) for c in common if snr_of(c) is not None})
        if ticks and "clean" in common:
            ax_snr.set_xticks(ticks + [max(ticks) + 10], [f"{t:g}" for t in ticks] + ["clean"])
        ax_snr.set(xlabel="SNR (dB), white noise", ylabel=f"{metric.upper()} (%)", title="Noise")
        ax_lvl.set(xlabel="Speech level (dB SPL)", title=f"Level (trained at {TRAIN_LEVEL_DB} dB SPL)")
        ax_lvl.axvline(TRAIN_LEVEL_DB, color="gray", lw=0.8, ls=":")
        for ax in (ax_snr, ax_lvl):
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 100)
        ax_snr.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{metric}_conditions.{ext}", dpi=200)
        plt.close(fig)

    # LaTeX tables: rows = conditions, columns = front ends, best per row in bold
    labels = list(tables)
    rows_kept = [c for c in TABLE_CONDITIONS if c in common]
    skipped = [c for c in TABLE_CONDITIONS if c not in common]
    if skipped:
        print(f"Not scored for every model (left out of the tables): {skipped}")
    tex = []
    for metric in ("wer", "cer"):
        tex += [r"\begin{table}[t]", r"\centering", r"\small",
                rf"\caption{{{metric.upper()} (\%) on {SPLIT}-clean per condition, by front end "
                r"(Conformer-CTC, 100\,h, one seed). Best per row in bold.}",
                rf"\label{{tab:frontends_{metric}}}",
                r"\begin{tabular}{l" + "r" * len(labels) + "}", r"\toprule",
                "Condition & " + " & ".join(labels) + r" \\", r"\midrule"]
        group = None
        for c in rows_kept:
            g = "snr" if snr_of(c) is not None or c == "clean" else "level"
            if group is not None and g != group:
                tex.append(r"\midrule")
            group = g
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