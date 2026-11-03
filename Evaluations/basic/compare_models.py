"""
Compare two front ends condition by condition, with paired significance tests.

Set the globals below and press Run in VS Code (or `python compare_models.py`). Needs no GPU:
it reads the per-utterance results that benchmark_model.py (or ASR.evaluate) already saved in
results/<name>/per_utt/<split>__<condition>.csv. Run benchmark_model.py on every model first.

For every condition that all models were scored on, it reports:

    WER of A and B, ΔWER = WER_A - WER_B with a 95% interval, and a two-sided p-value

using the paired bootstrap (Bisani & Ney, 2004): resample the test UTTERANCES 1,000 times, score
both systems on the same resampled set each time, and look at the spread of the difference.

Several seeds: list one model per seed on each side, in the same order (A[i] pairs with B[i]).
The pooled test resamples utterances once per draw and scores every seed on them, so it covers
the test-set sampling; the per-seed signs show whether training randomness could flip the result.
A claim needs both: every seed agrees on the sign, and the pooled p-value (Holm-corrected for the
number of conditions) is below 0.05.

Outputs go to results/compare/<A>_vs_<B>_<split>/: comparison.csv, delta_vs_snr.png,
delta_vs_level.png and report.txt.

Author: Eric Oliveira
Date: 11/02/2026

* Written with the help of Claude *

"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # Draw to files: works on Quest without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# SETTINGS
# =============================================================================

# System A and system B: .pt files, _meta.json files or model names. One per seed, same seed order
MODELS_A = [
    "models/2026-11-03_carfac-agc-log_librispeech-train.100_clean_s25_af2b85.pt",
]
MODELS_B = [
    "models/2026-11-02_logmel80_librispeech-train.100_clean_s25_95b1a7_meta.json",
]

# Workspace root. None: the folder that holds models/ (inferred from the first model), else the current folder
ROOT = None

# Which test split's results to compare
SPLIT = "test"

# Conditions to compare (names as saved, e.g. "clean", "white@10dB", "clean@45dBSPL").
# None: every condition that ALL models were scored on
CONDITIONS = None

# Bootstrap draws, and the seed of the resampling (fixed: same answer every run)
N_BOOT = 1000
BOOT_SEED = 0

# Refuse to compare models that differ in more than front end and seed (see assert_comparable)
REQUIRE_COMPARABLE = True


# =============================================================================
# Loading
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


def resolve(model: str, root: Path | None) -> tuple[str, dict, Path]:
    """A .pt, a _meta.json or a name → (model name, its meta dict, workspace root)."""
    path = Path(model).expanduser()
    if path.name.endswith("_meta.json"):
        meta_path = path
    elif path.name.endswith(".pt"):
        meta_path = path.with_name(path.name[: -len(".pt")] + "_meta.json")
    else:
        # A bare name (endswith, not .suffix: names contain dots like "train.100")
        meta_path = Path(root or Path.cwd()) / "models" / f"{path.name}_meta.json"

    meta_path = meta_path.resolve()
    if not meta_path.exists():
        sys.exit(f"No meta file for {model}: {meta_path}")

    if root is None:
        root = meta_path.parent.parent if meta_path.parent.name == "models" else Path.cwd()
    meta = json.loads(meta_path.read_text())
    return meta["name"], meta, Path(root).resolve()


def saved_conditions(root: Path, name: str, split: str) -> set[str]:
    """Conditions this model has per-utterance results for, on this split."""
    folder = root / "results" / name / "per_utt"
    return {p.stem.split("__", 1)[1] for p in folder.glob(f"{split}__*.csv")}


def load_per_utt(root: Path, name: str, split: str, condition: str) -> pd.DataFrame:
    """utt_id, w_err, w_ref (+ c_err, c_ref) for every utterance, sorted by utt_id."""
    path = root / "results" / name / "per_utt" / f"{split}__{condition}.csv"
    df = pd.read_csv(path, dtype={"utt_id": str})
    return df.sort_values("utt_id").reset_index(drop=True)


def parse_condition(name: str) -> dict:
    """'white@10dB@45dBSPL' → noise 'white', snr 10, level 45 (65 dB SPL when not written)."""
    out = {"noise": None, "snr_db": np.nan, "speech_db_spl": 65.0}
    base = name.removesuffix("-trainnoise")
    level = re.search(r"@(-?[\d.]+)dBSPL", base)
    if level:
        out["speech_db_spl"] = float(level.group(1))
        base = base[: level.start()]
    noisy = re.fullmatch(r"(.+)@(-?[\d.]+)dB", base)
    if noisy:
        out["noise"], out["snr_db"] = noisy.group(1), float(noisy.group(2))
    return out


# =============================================================================
# Statistics
# =============================================================================

def paired_stats(pairs: list[tuple[pd.DataFrame, pd.DataFrame]], n_boot: int, seed: int) -> dict:
    """ΔWER = WER_A - WER_B over one or more seed pairs, with a paired bootstrap over utterances.

    Each draw resamples the utterances ONCE and scores every seed of both systems on them, so the
    comparison stays paired (same utterances for A and B) and the seeds are averaged, not pooled
    as if they were extra utterances. With one pair this is exactly asrlib's paired_bootstrap.
    """
    # Line every seed up on the same utterances: (n_seeds, n_utts) error counts
    ids = pairs[0][0].utt_id.to_numpy()
    for a, b in pairs:
        if not (np.array_equal(a.utt_id, ids) and np.array_equal(b.utt_id, ids)):
            raise ValueError("Models were scored on different utterances; re-run their benchmarks")
    err_a = np.stack([a.w_err.to_numpy() for a, _ in pairs]).astype(float)
    err_b = np.stack([b.w_err.to_numpy() for _, b in pairs]).astype(float)
    ref = pairs[0][0].w_ref.to_numpy().astype(float)

    # Corpus WER averaged over seeds: total errors / (seeds × total reference words)
    def wer(err, idx=None):
        if idx is None:
            return err.sum() / (len(err) * ref.sum())
        return err[:, idx].sum(axis=(0, 2)) / (len(err) * ref[idx].sum(axis=1))

    # Resample utterance indices: (n_boot, n_utts), the same rows for A and B
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref), size=(n_boot, len(ref)))
    deltas = wer(err_a, idx) - wer(err_b, idx)

    # Per-seed differences: do all seeds agree on the sign?
    per_seed = [(a.w_err.sum() - b.w_err.sum()) / ref.sum() for a, b in pairs]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return {"wer_a": wer(err_a), "wer_b": wer(err_b), "delta": wer(err_a) - wer(err_b),
            "delta_lo": lo, "delta_hi": hi, "p": p, "per_seed": per_seed,
            "seeds_agree": all(d > 0 for d in per_seed) or all(d < 0 for d in per_seed)}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values: controls false positives across many conditions."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


# =============================================================================
# Plots
# =============================================================================

def plot_delta(df: pd.DataFrame, x: str, group: str | None, xlabel: str, title: str,
               out: Path, reverse_x: bool = False) -> None:
    """ΔWER with its 95% interval; below zero = A better, above zero = B better."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = df.groupby(group, sort=False) if group else [("clean speech", df)]
    for label, g in groups:
        g = g.sort_values(x)
        y = g.delta * 100
        err = np.vstack([y - g.delta_lo * 100, g.delta_hi * 100 - y])
        ax.errorbar(g[x], y, yerr=err, marker="o", capsize=3, label=label)
    ax.axhline(0, color="gray", lw=1)
    if reverse_x:
        ax.invert_xaxis()
    ax.set(xlabel=xlabel, ylabel="ΔWER = A − B (points)", title=title)
    ax.text(0.01, 0.02, "below 0: A better   above 0: B better", transform=ax.transAxes, fontsize=9,
            color="gray")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =============================================================================
# Run
# =============================================================================

def main():
    if len(MODELS_A) != len(MODELS_B) or not MODELS_A:
        sys.exit("MODELS_A and MODELS_B need the same number of models (one per seed, same order)")

    # Resolve every model; the workspace comes from the first one unless ROOT is set
    root = Path(ROOT).expanduser().resolve() if ROOT else None
    a_models, b_models = [], []
    for spec_a, spec_b in zip(MODELS_A, MODELS_B):
        name_a, meta_a, root = resolve(spec_a, root)
        name_b, meta_b, root = resolve(spec_b, root)
        a_models.append((name_a, meta_a))
        b_models.append((name_b, meta_b))

    label_a = a_models[0][1]["frontend"]["name"]
    label_b = b_models[0][1]["frontend"]["name"]
    out = root / "results" / "compare" / f"{label_a}_vs_{label_b}_{SPLIT}"
    out.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout)
    sys.stdout = tee

    try:
        print(f"A = {label_a}   B = {label_b}   split {SPLIT}   {len(a_models)} seed pair(s)")
        for (na, ma), (nb, mb) in zip(a_models, b_models):
            print(f"  A {na}\n  B {nb}")
            if ma["seed"] != mb["seed"]:
                print(f"    note: seeds differ in this pair ({ma['seed']} vs {mb['seed']})")

        # Fair comparison: everything except front end and seed must match
        hashes = {m["controlled_hash"] for _, m in a_models + b_models}
        if len(hashes) > 1:
            message = (f"Models differ in more than front end and seed (controlled hashes {sorted(hashes)}). "
                       "Diff their configs before comparing.")
            if REQUIRE_COMPARABLE:
                sys.exit(message)
            print(f"  WARNING: {message}")
        else:
            print(f"  controlled hash {hashes.pop()}: comparable")

        # Conditions every model has results for
        names = [n for n, _ in a_models + b_models]
        common = set.intersection(*(saved_conditions(root, n, SPLIT) for n in names))
        conditions = sorted(common) if CONDITIONS is None else [c for c in CONDITIONS if c in common]
        missing = [] if CONDITIONS is None else [c for c in CONDITIONS if c not in common]
        if missing:
            print(f"  Skipping {missing}: not scored for every model (run benchmark_model.py on each)")
        if not conditions:
            sys.exit("No condition was scored for every model. Run benchmark_model.py on each model first.")

        # One paired test per condition
        rows = []
        for cond in conditions:
            pairs = [(load_per_utt(root, na, SPLIT, cond), load_per_utt(root, nb, SPLIT, cond))
                     for (na, _), (nb, _) in zip(a_models, b_models)]
            stats = paired_stats(pairs, N_BOOT, BOOT_SEED)
            rows.append({"condition": cond, **parse_condition(cond), **stats})

        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df.p.tolist())
        df["verdict"] = np.where(
            (df.p_holm < 0.05) & df.seeds_agree,
            np.where(df.delta < 0, f"{label_a} better", f"{label_b} better"),
            "no clear difference")

        # Order: clean first, then each noise by SNR (high to low), then the level sweep
        df["_kind"] = np.select([df.noise.notna(), df.speech_db_spl != 65], [1, 2], 0)
        df = df.sort_values(["_kind", "noise", "snr_db", "speech_db_spl"],
                            ascending=[True, True, False, True], na_position="first").drop(columns="_kind")

        print(f"\n{'condition':22s} {'WER A':>7s} {'WER B':>7s} {'ΔWER':>7s} {'95% CI of Δ':>17s} "
              f"{'p':>6s} {'p_holm':>7s} {'seeds':>6s}  verdict")
        for _, r in df.iterrows():
            agree = "agree" if r.seeds_agree else "split"
            print(f"{r.condition:22s} {r.wer_a:7.2%} {r.wer_b:7.2%} {r.delta * 100:+6.2f}  "
                  f"[{r.delta_lo * 100:+6.2f}, {r.delta_hi * 100:+6.2f}] {r.p:6.3f} {r.p_holm:7.3f} "
                  f"{agree:>6s}  {r.verdict}")
        print("\nΔWER in percentage points; negative = A better. A difference counts only when every seed "
              "agrees on its sign and p_holm < 0.05.")
        if len(a_models) == 1:
            print("One seed per side: the test covers which utterances were tested, not training randomness.")

        # Save the table (per-seed deltas as text, so the CSV stays flat)
        df.assign(per_seed=df.per_seed.apply(lambda d: ";".join(f"{x:+.4f}" for x in d))) \
          .to_csv(out / "comparison.csv", index=False)

        # Plots: Δ vs SNR per noise, Δ vs speech level
        noisy = df[df.noise.notna() & (df.speech_db_spl == 65)]
        if not noisy.empty:
            plot_delta(noisy, "snr_db", "noise", "SNR (dB)", f"ΔWER, {label_a} − {label_b}",
                       out / "delta_vs_snr.png", reverse_x=True)
        levels = df[df.noise.isna()]
        if len(levels) > 1:
            plot_delta(levels, "speech_db_spl", None, "speech level (dB SPL)",
                       f"ΔWER, {label_a} − {label_b}", out / "delta_vs_level.png")

        print(f"\nSaved to {out}")
    finally:
        sys.stdout = tee.stream
        (out / "report.txt").write_text("".join(tee.lines))


if __name__ == "__main__":
    main()