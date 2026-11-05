"""
Summarize the CARFAC design sweep: every run against the reference, on dev-clean, in one table and one figure.

Set the globals below and press Run in VS Code (or `python summarize_sweep.py`). Needs no GPU: it
reads what training already saved (models/<name>_meta.json and results/<name>/per_utt/*.csv).

It finds every model in models/ with the sweep's controlled hash (same model, data and training),
labels each by what differs from the reference's front end (e.g. "velocity_scale 0.1 → 0.05"),
and reports for each run and condition (clean, white at 10 and 5 dB):

    WER, CER, PER, WPFER
    ΔWER vs the reference with a 95% paired-bootstrap interval and a Holm-corrected p-value

Outputs go to results/compare/carfac_sweep_<controlled hash>/: sweep.csv, sweep_delta_wer.png
(ΔWER per run and condition), report.txt.

Author: Eric Oliveira
Date: 11/04/2026

* Written with the help of Claude *

"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # Draw to files: works on Quest without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asrlib.metrics import add_error_columns

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# SETTINGS
# =============================================================================

# Workspace root (holds models/ and results/): $ASRLIB_ROOT, else the folder you launch from
ROOT = Path(os.environ.get("ASRLIB_ROOT", Path().resolve()))

# The sweep: every model with this controlled hash is included
CONTROLLED_HASH = "1db986"

# The reference run's config hash: every other run is compared against it
REFERENCE = "aebde1"

# Where and what the runs were scored on
SPLIT = "validation"
CONDITIONS = ("clean", "white@10dB", "white@5dB")

# Paired bootstrap draws (fixed seed: same answer every run)
N_BOOT = 1000
BOOT_SEED = 0


# =============================================================================
# Helpers
# =============================================================================

class Tee:
    """Write to the terminal and keep a copy, so the report can be saved."""

    def __init__(self, stream):
        self.stream, self.lines = stream, []

    def write(self, text):
        self.stream.write(text)
        self.lines.append(text)

    def flush(self):
        self.stream.flush()


def same(a, b) -> bool:
    """Equal values, with floats compared to 1e-9 (π multiples, √2, ...) and lists vs tuples alike."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def fmt(v) -> str:
    """Short readable value: 0.9500000001·π → 2.985, (0.004, 0.016, ...) → 0.004/0.016/..."""
    if isinstance(v, (list, tuple)):
        return "/".join(fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def label_run(frontend: dict, reference: dict) -> str:
    """What this run's front end changes relative to the reference, e.g. 'velocity_scale 0.1 → 0.05'."""
    if frontend.get("kind") != reference.get("kind"):
        return frontend.get("name", frontend.get("kind", "?"))
    changes = []
    for key in sorted(set(frontend) | set(reference)):
        if key in ("name", "cache_key", "n_feats"):
            continue
        a, b = reference.get(key), frontend.get(key)
        if isinstance(a, dict) or isinstance(b, dict):
            a, b = a or {}, b or {}
            for sub in sorted(set(a) | set(b)):
                if not same(a.get(sub), b.get(sub)):
                    changes.append(f"{sub} {fmt(a.get(sub, 'default'))} → {fmt(b.get(sub, 'default'))}")
        elif not same(a, b):
            changes.append(f"{key} {fmt(a)} → {fmt(b)}")
    return ", ".join(changes) or "reference"


def load_per_utt(name: str, condition: str) -> pd.DataFrame | None:
    """A run's per-utterance results for one condition (S/D/I, PER, WPFER filled in if missing)."""
    path = ROOT / "results" / name / "per_utt" / f"{SPLIT}__{condition}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"utt_id": str, "ref": str, "hyp": str}, keep_default_na=False)
    return add_error_columns(df).sort_values("utt_id").reset_index(drop=True)


def rates(df: pd.DataFrame) -> dict:
    """Corpus-level WER, CER, PER and WPFER."""
    out = {"wer": df.w_err.sum() / df.w_ref.sum(), "cer": df.c_err.sum() / df.c_ref.sum()}
    out["per"] = df.p_err.sum() / df.p_ref.sum() if df.p_err.notna().all() else np.nan
    out["wpfer"] = df.pf_err.sum() / df.p_ref.sum() if "pf_err" in df and df.pf_err.notna().all() else np.nan
    return out


def paired_delta(run: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """ΔWER = run − reference, with a paired bootstrap over utterances (same utterances in every draw)."""
    if not np.array_equal(run.utt_id, ref.utt_id):
        raise ValueError("Runs were scored on different utterances")
    e_run, e_ref, n = (x.to_numpy(dtype=float) for x in (run.w_err, ref.w_err, ref.w_ref))
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(n), size=(N_BOOT, len(n)))
    deltas = (e_run[idx].sum(1) - e_ref[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return {"d_wer": (e_run.sum() - e_ref.sum()) / n.sum(), "d_lo": lo, "d_hi": hi, "p": p}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values across every test in the table."""
    order = np.argsort(p_values)
    adjusted, running, m = np.empty(len(p_values)), 0.0, len(p_values)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


# =============================================================================
# Run
# =============================================================================

def main():
    # Every model of the sweep, by its meta file
    metas = []
    for path in sorted((ROOT / "models").glob("*_meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("controlled_hash") == CONTROLLED_HASH:
            metas.append(meta)
    by_hash = {m["config_hash"]: m for m in metas}
    if REFERENCE not in by_hash:
        sys.exit(f"Reference {REFERENCE} not found among {len(metas)} models with controlled hash {CONTROLLED_HASH}")

    out = ROOT / "results" / "compare" / f"carfac_sweep_{CONTROLLED_HASH}"
    out.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout)
    sys.stdout = tee

    try:
        ref_meta = by_hash[REFERENCE]
        ref_frontend = ref_meta["config"]["frontend"]
        print(f"CARFAC sweep {CONTROLLED_HASH}: {len(metas)} runs | reference {ref_meta['name']} | split {SPLIT}")

        ref_tables = {c: load_per_utt(ref_meta["name"], c) for c in CONDITIONS}
        missing_ref = [c for c, t in ref_tables.items() if t is None]
        if missing_ref:
            sys.exit(f"The reference has no {SPLIT} results for {missing_ref}: evaluate it on those first")

        rows = []
        for meta in metas:
            label = label_run(meta["config"]["frontend"], ref_frontend)
            for cond in CONDITIONS:
                table = load_per_utt(meta["name"], cond)
                if table is None:
                    print(f"  {label}: no {SPLIT} results for {cond} (skipped)")
                    continue
                row = {"run": label, "config_hash": meta["config_hash"], "name": meta["name"],
                       "n_feats": meta["frontend"].get("n_feats"), "condition": cond, **rates(table)}
                if meta["config_hash"] != REFERENCE:
                    row.update(paired_delta(table, ref_tables[cond]))
                rows.append(row)

        df = pd.DataFrame(rows)
        tested = df.d_wer.notna() if "d_wer" in df else pd.Series(False, index=df.index)
        df.loc[tested, "p_holm"] = holm(df.loc[tested, "p"].tolist())

        # One row per run: WER per condition, ΔWER vs the reference, mean over conditions
        wide = df.pivot_table(index=["run", "config_hash", "n_feats"], columns="condition",
                              values=["wer", "d_wer"], aggfunc="first")
        wide["mean_wer"] = wide["wer"][list(CONDITIONS)].mean(axis=1)
        wide = wide.sort_values("mean_wer")

        print(f"\nDev WER by run, best mean first (Δ = run − reference; * = p_holm < 0.05)")
        header = f"{'run':52s} {'ch':>3s} " + " ".join(f"{c:>17s}" for c in CONDITIONS) + f" {'mean':>7s}"
        print(header)
        for (run, h, nf), r in wide.iterrows():
            cells = []
            for c in CONDITIONS:
                w = r[("wer", c)]
                if run == "reference":
                    cells.append(f"{w:7.2%}{'':>10s}")
                else:
                    sub = df[(df.config_hash == h) & (df.condition == c)].iloc[0]
                    star = "*" if sub.p_holm < 0.05 else " "
                    cells.append(f"{w:7.2%} ({sub.d_wer * 100:+5.2f}{star})")
            print(f"{run[:52]:52s} {int(nf) if pd.notna(nf) else '?':>3} " + " ".join(cells) + f" {r['mean_wer'].iloc[0]:7.2%}")

        print(f"\nAll metrics per run and condition (CER, PER, WPFER: lower is better)")
        print(f"{'run':52s} {'condition':>11s} {'WER':>7s} {'CER':>7s} {'PER':>7s} {'WPFER':>7s} "
              f"{'ΔWER':>7s} {'95% CI of Δ':>17s} {'p_holm':>7s}")
        for _, r in df.sort_values(["run", "condition"]).iterrows():
            delta = (f"{r.d_wer * 100:+6.2f}  [{r.d_lo * 100:+6.2f}, {r.d_hi * 100:+6.2f}] {r.p_holm:7.3f}"
                     if pd.notna(r.get("d_wer")) else f"{'—':>7s} {'':>17s} {'':>7s}")
            print(f"{r.run[:52]:52s} {r.condition:>11s} {r.wer:7.2%} {r.cer:7.2%} {r.per:7.2%} {r.wpfer:7.2%} {delta}")

        print("\nOne seed per run: Δ under ~1 point may be seed noise; re-run finalists with a second seed.")
        df.to_csv(out / "sweep.csv", index=False)

        # Figure: ΔWER vs the reference, one panel per condition, runs sorted by mean WER
        runs = [run for (run, _, _), _ in wide.iterrows() if run != "reference"]
        if runs:
            fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(5 * len(CONDITIONS), 0.45 * len(runs) + 1.8),
                                     sharey=True, squeeze=False)
            for ax, cond in zip(axes[0], CONDITIONS):
                sub = df[(df.condition == cond) & df.run.isin(runs)].set_index("run").reindex(runs)
                y = np.arange(len(runs))
                colors = ["tab:green" if d < 0 else "tab:red" for d in sub.d_wer]
                ax.barh(y, sub.d_wer * 100, color=colors, alpha=0.75)
                ax.errorbar(sub.d_wer * 100, y, xerr=[(sub.d_wer - sub.d_lo) * 100, (sub.d_hi - sub.d_wer) * 100],
                            fmt="none", ecolor="black", capsize=2, lw=0.8)
                ax.axvline(0, color="gray", lw=1)
                ax.set(title=f"{cond} ({SPLIT})", xlabel="ΔWER vs reference (points)")
                ax.set_yticks(y, [r[:45] for r in runs], fontsize=8)
                ax.grid(alpha=0.3, axis="x")
            axes[0][0].invert_yaxis()            # Best mean WER at the top
            fig.suptitle("CARFAC sweep: left of 0 = better than the reference", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "sweep_delta_wer.png", dpi=150)
            plt.close(fig)

        print(f"\nSaved to {out}")
        for f in sorted(out.iterdir()):
            print(f"  {f.name}")
    finally:
        sys.stdout = tee.stream
        (out / "report.txt").write_text("".join(tee.lines))


if __name__ == "__main__":
    main()