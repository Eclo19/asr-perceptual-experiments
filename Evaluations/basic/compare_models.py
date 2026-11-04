"""
Compare two front ends condition by condition, with paired significance tests and side-by-side plots.

Set the globals below and press Run in VS Code (or `python compare_models.py`). Needs no GPU:
it reads the per-utterance results that benchmark_model.py (or ASR.evaluate) already saved in
results/<name>/per_utt/<split>__<condition>.csv. Run benchmark_model.py on every model first.

For every condition that all models were scored on, it reports:

    WER of A and B, ΔWER = WER_A - WER_B with a 95% interval, and a two-sided p-value
    PER of A and B and ΔPER, tested the same way (did the model hear the right sounds?)
    WPFER of A and B and ΔWPFER: PER with partial credit for acoustically close phones (panphon)
    the word errors of each side split into substitutions / deletions / insertions

using the paired bootstrap (Bisani & Ney, 2004): resample the test UTTERANCES 1,000 times, score
both systems on the same resampled set each time, and look at the spread of the difference.

Several seeds: list one model per seed on each side, in the same order (A[i] pairs with B[i]).
The pooled test resamples utterances once per draw and scores every seed on them, so it covers
the test-set sampling; the per-seed signs show whether training randomness could flip the result.
A claim needs both: every seed agrees on the sign, and the pooled p-value (Holm-corrected for the
number of conditions) is below 0.05. WER is the primary metric; PER is reported alongside.

Results saved before these metrics existed are upgraded on the fly: S/D/I, PER and WPFER are recomputed from their
ref/hyp text (needs g2p_en and panphon, see benchmark_model.py). Nothing is rewritten on disk.

Outputs go to results/compare/<A>_vs_<B>_<split>/:
    comparison.csv                       every number in the tables below
    {wer,per,wpfer}_vs_snr.png           A and B on the same axes, one panel per noise
    {wer,per,wpfer}_vs_level.png         A and B on the same axes
    delta_vs_snr.png, delta_vs_level.png ΔWER with its 95% interval (below 0 = A better)
    report.txt                           a copy of everything printed

Author: Eric Oliveira
Date: 11/02/2026 (PER, S/D/I and side-by-side plots: 11/03/2026)

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

from asrlib.metrics import _g2p, _phone_table, add_error_columns

sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# SETTINGS
# =============================================================================

# System A and system B: .pt files, _meta.json files or model names. One per seed, same seed order
MODELS_A = [
    "models/2026-11-03_carfac-agc-log_librispeech-train.100_clean_s30_64fdc8_meta.json",
]
MODELS_B = [
    "models/2026-11-03_logmel80_librispeech-train.100_clean_s30_fc49ec_meta.json",
]

# Workspace root. None: the folder that holds models/ (inferred from the first model), else the current folder
ROOT = None

# Which test split's results to compare
SPLIT = "test"

# Conditions to compare (names as saved, e.g. "clean", "white@10dB", "clean@45dBSPL").
# None: every condition that ALL models were scored on
# Conditions to compare (names as saved by benchmark_model.py's default sweep)
CONDITIONS = (
    ["clean"]
    + [f"white@{snr:g}dB" for snr in (40, 30, 20, 10, 5, 0, -5, -10)]          # White noise, speech at 65 dB SPL
    + [f"clean@{level:g}dBSPL" for level in (35, 45, 55, 75, 85, 95)]          # Level sweep (65 dB SPL = "clean")
)

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


_UPGRADED: list[str] = []                  # Tables that needed S/D/I or PER recomputed (reported once)


def load_per_utt(root: Path, name: str, split: str, condition: str) -> pd.DataFrame:
    """Every utterance's error counts (words, characters, phonemes, S/D/I), sorted by utt_id."""
    path = root / "results" / name / "per_utt" / f"{split}__{condition}.csv"

    # keep_default_na=False: an empty hypothesis (the model output nothing) stays "", not NaN
    df = pd.read_csv(path, dtype={"utt_id": str, "ref": str, "hyp": str}, keep_default_na=False)

    # Older results lack S/D/I or PER: recompute them from the saved text
    full = add_error_columns(df)
    if full is not df:
        _UPGRADED.append(f"{name}/{split}__{condition}")
    return full.sort_values("utt_id").reset_index(drop=True)


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

def paired_stats(pairs: list[tuple[pd.DataFrame, pd.DataFrame]], n_boot: int, seed: int,
                 err: str = "w_err", ref: str = "w_ref") -> dict:
    """Error rate of A and B and their difference A - B, with a paired bootstrap over utterances.

    err/ref pick the unit: ("w_err", "w_ref") = WER, ("p_err", "p_ref") = PER, ("pf_err", "p_ref") = WPFER.
    Each draw resamples the utterances ONCE and scores every seed of both systems on them, so the
    comparison stays paired (same utterances for A and B) and the seeds are averaged, not pooled
    as if they were extra utterances. With one pair, WER is exactly asrlib's paired_bootstrap.
    """
    # Line every seed up on the same utterances: (n_seeds, n_utts) error counts
    ids = pairs[0][0].utt_id.to_numpy()
    for a, b in pairs:
        if not (np.array_equal(a.utt_id, ids) and np.array_equal(b.utt_id, ids)):
            raise ValueError("Models were scored on different utterances; re-run their benchmarks")
    err_a = np.stack([a[err].to_numpy() for a, _ in pairs]).astype(float)
    err_b = np.stack([b[err].to_numpy() for _, b in pairs]).astype(float)
    n_ref = pairs[0][0][ref].to_numpy().astype(float)

    # Corpus rate averaged over seeds: total errors / (seeds × total reference units)
    def rate(e, idx=None):
        if idx is None:
            return e.sum() / (len(e) * n_ref.sum())
        return e[:, idx].sum(axis=(0, 2)) / (len(e) * n_ref[idx].sum(axis=1))

    # Resample utterance indices: (n_boot, n_utts), the same rows for A and B
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n_ref), size=(n_boot, len(n_ref)))
    boot_a, boot_b = rate(err_a, idx), rate(err_b, idx)
    deltas = boot_a - boot_b

    # Per-seed differences: do all seeds agree on the sign?
    per_seed = [(a[err].sum() - b[err].sum()) / n_ref.sum() for a, b in pairs]
    q = [0.025, 0.975]
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return {"a": rate(err_a), "b": rate(err_b), "delta": rate(err_a) - rate(err_b),
            "a_lo": np.quantile(boot_a, q[0]), "a_hi": np.quantile(boot_a, q[1]),
            "b_lo": np.quantile(boot_b, q[0]), "b_hi": np.quantile(boot_b, q[1]),
            "delta_lo": np.quantile(deltas, q[0]), "delta_hi": np.quantile(deltas, q[1]),
            "p": p, "per_seed": per_seed,
            "seeds_agree": all(d > 0 for d in per_seed) or all(d < 0 for d in per_seed)}


def error_types(frames: list[pd.DataFrame]) -> dict:
    """Substitution / deletion / insertion rates (fractions of reference words), averaged over seeds."""
    words = sum(f.w_ref.sum() for f in frames)
    return {k: sum(f[c].sum() for f in frames) / words
            for k, c in (("sub", "w_sub"), ("del", "w_del"), ("ins", "w_ins"))}


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

def plot_two_curves(df: pd.DataFrame, metric: str, x: str, group: str | None, labels: tuple[str, str],
                    clean: pd.Series | None, xlabel: str, out: Path, reverse_x: bool = False) -> None:
    """A and B on the same axes, one panel per noise (or one panel for the level sweep).

    metric "wer" or "per": columns <metric>_a / <metric>_b with their 95% bootstrap intervals.
    Dashed lines: each system's clean score, the reference both curves start from.
    """
    groups = list(df.groupby(group, sort=False)) if group else [(None, df)]
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), squeeze=False, sharey=True)
    colors = ("tab:blue", "tab:orange")
    for ax, (name, g) in zip(axes[0], groups):
        g = g.sort_values(x)
        for side, label, color, marker in (("a", labels[0], colors[0], "o"), ("b", labels[1], colors[1], "s")):
            y = g[f"{metric}_{side}"] * 100
            err = np.vstack([y - g[f"{metric}_{side}_lo"] * 100, g[f"{metric}_{side}_hi"] * 100 - y])
            ax.errorbar(g[x], y, yerr=err, marker=marker, color=color, capsize=3, label=label)
            if clean is not None:
                ax.axhline(clean[f"{metric}_{side}"] * 100, color=color, ls="--", lw=1, alpha=0.7,
                           label=f"{label}, clean ({clean[f'{metric}_{side}']:.1%})")
        if reverse_x:
            ax.invert_xaxis()                # High SNR (easy) on the left, as in most papers
        title = f"{metric.upper()}" + (f": {name} noise" if name else "")
        ax.set(xlabel=xlabel, title=title)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"{metric.upper()} (%)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_delta(df: pd.DataFrame, x: str, group: str | None, xlabel: str, title: str,
               out: Path, reverse_x: bool = False) -> None:
    """ΔWER with its 95% interval; below zero = A better, above zero = B better."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = df.groupby(group, sort=False) if group else [("clean speech", df)]
    for label, g in groups:
        g = g.sort_values(x)
        y = g.wer_delta * 100
        err = np.vstack([y - g.wer_delta_lo * 100, g.wer_delta_hi * 100 - y])
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
        print(f"  PER: {'on (g2p_en)' if _g2p() is not None else 'off (g2p_en missing)'}   "
              f"WPFER: {'on (panphon)' if _phone_table() is not None else 'off (panphon missing)'}")

        # Conditions every model has results for
        names = [n for n, _ in a_models + b_models]
        common = set.intersection(*(saved_conditions(root, n, SPLIT) for n in names))
        conditions = sorted(common) if CONDITIONS is None else [c for c in CONDITIONS if c in common]
        missing = [] if CONDITIONS is None else [c for c in CONDITIONS if c not in common]
        if missing:
            print(f"  Skipping {missing}: not scored for every model (run benchmark_model.py on each)")
        if not conditions:
            sys.exit("No condition was scored for every model. Run benchmark_model.py on each model first.")

        # One paired test per condition, for WER (primary) and PER
        rows = []
        for cond in conditions:
            pairs = [(load_per_utt(root, na, SPLIT, cond), load_per_utt(root, nb, SPLIT, cond))
                     for (na, _), (nb, _) in zip(a_models, b_models)]
            row = {"condition": cond, **parse_condition(cond)}
            row.update({f"wer_{k}": v for k, v in paired_stats(pairs, N_BOOT, BOOT_SEED).items()})
            has_per = all(f.p_err.notna().all() for pair in pairs for f in pair)
            if has_per:
                row.update({f"per_{k}": v for k, v in
                            paired_stats(pairs, N_BOOT, BOOT_SEED, err="p_err", ref="p_ref").items()})
            if all(f.pf_err.notna().all() for pair in pairs for f in pair):
                row.update({f"wpfer_{k}": v for k, v in
                            paired_stats(pairs, N_BOOT, BOOT_SEED, err="pf_err", ref="p_ref").items()})
            row.update({f"{k}_a": v for k, v in error_types([a for a, _ in pairs]).items()})
            row.update({f"{k}_b": v for k, v in error_types([b for _, b in pairs]).items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        has_per = "per_a" in df and df.per_a.notna().all()
        df["wer_p_holm"] = holm(df.wer_p.tolist())
        has_wpfer = "wpfer_a" in df and df.wpfer_a.notna().all()
        if has_per:
            df["per_p_holm"] = holm(df.per_p.tolist())
        if has_wpfer:
            df["wpfer_p_holm"] = holm(df.wpfer_p.tolist())
        df["verdict"] = np.where(
            (df.wer_p_holm < 0.05) & df.wer_seeds_agree,
            np.where(df.wer_delta < 0, f"{label_a} better", f"{label_b} better"),
            "no clear difference")

        # Order: clean first, then each noise by SNR (high to low), then the level sweep
        df["_kind"] = np.select([df.noise.notna(), df.speech_db_spl != 65], [1, 2], 0)
        df = df.sort_values(["_kind", "noise", "snr_db", "speech_db_spl"],
                            ascending=[True, True, False, True], na_position="first").drop(columns="_kind")

        if _UPGRADED:
            print(f"  Recomputed S/D/I, PER and WPFER from saved text for {len(_UPGRADED)} older result table(s) "
                  "(re-run benchmark_model.py to store them)")

        # ---- Table 1: WER, the primary metric
        print(f"\nWER   (A = {label_a}, B = {label_b})")
        print(f"{'condition':22s} {'A':>7s} {'B':>7s} {'ΔWER':>7s} {'95% CI of Δ':>17s} "
              f"{'p':>6s} {'p_holm':>7s} {'seeds':>6s}  verdict")
        for _, r in df.iterrows():
            agree = "agree" if r.wer_seeds_agree else "split"
            print(f"{r.condition:22s} {r.wer_a:7.2%} {r.wer_b:7.2%} {r.wer_delta * 100:+6.2f}  "
                  f"[{r.wer_delta_lo * 100:+6.2f}, {r.wer_delta_hi * 100:+6.2f}] {r.wer_p:6.3f} "
                  f"{r.wer_p_holm:7.3f} {agree:>6s}  {r.verdict}")

        # ---- Table 2: phoneme-level metrics
        def phone_cols(r, m, ok):
            if not ok:
                return f"{'—':>7s} {'—':>7s} {'—':>7s} {'—':>7s}"
            return f"{r[f'{m}_a']:7.2%} {r[f'{m}_b']:7.2%} {r[f'{m}_delta'] * 100:+6.2f}  {r[f'{m}_p_holm']:7.3f}"

        print("\nPhoneme metrics: PER, and WPFER (PER with partial credit for close phones)")
        print(f"{'condition':22s} {'PER A':>7s} {'PER B':>7s} {'ΔPER':>7s} {'p_holm':>7s}   "
              f"{'WPFER A':>7s} {'WPFER B':>7s} {'ΔWPFER':>7s} {'p_holm':>7s}")
        for _, r in df.iterrows():
            print(f"{r.condition:22s} {phone_cols(r, 'per', has_per)}   {phone_cols(r, 'wpfer', has_wpfer)}")

        # ---- Table 3: how the word errors split
        print("\nWord error types (S / D / I, % of reference words; they add up to WER)")
        print(f"{'condition':22s} {'A: S / D / I':>18s}   {'B: S / D / I':>18s}")
        for _, r in df.iterrows():
            types = [f"{r[f'sub_{s}'] * 100:5.1f} /{r[f'del_{s}'] * 100:5.1f} /{r[f'ins_{s}'] * 100:5.1f}"
                     for s in ("a", "b")]
            print(f"{r.condition:22s} {types[0]:>18s}   {types[1]:>18s}")

        print("\nΔ in percentage points; negative = A better. A WER difference counts only when every "
              "seed agrees on its sign and p_holm < 0.05.")
        if len(a_models) == 1:
            print("One seed per side: the test covers which utterances were tested, not training randomness.")

        # Save the table (per-seed deltas as text, so the CSV stays flat)
        flat = df.copy()
        for col in [c for c in flat.columns if c.endswith("per_seed")]:
            flat[col] = flat[col].apply(lambda d: ";".join(f"{x:+.4f}" for x in d))
        flat.to_csv(out / "comparison.csv", index=False)

        # ---- Plots: both systems on the same axes, then the differences
        clean = df[df.condition == "clean"].iloc[0] if (df.condition == "clean").any() else None
        noisy = df[df.noise.notna() & (df.speech_db_spl == 65)]
        levels = df[df.noise.isna()]
        for metric in ["wer"] + (["per"] if has_per else []) + (["wpfer"] if has_wpfer else []):
            if not noisy.empty:
                plot_two_curves(noisy, metric, "snr_db", "noise", (label_a, label_b), clean, "SNR (dB)",
                                out / f"{metric}_vs_snr.png", reverse_x=True)
            if len(levels) > 1:
                plot_two_curves(levels, metric, "speech_db_spl", None, (label_a, label_b), None,
                                "speech level (dB SPL)", out / f"{metric}_vs_level.png")
        if not noisy.empty:
            plot_delta(noisy, "snr_db", "noise", "SNR (dB)", f"ΔWER, {label_a} − {label_b}",
                       out / "delta_vs_snr.png", reverse_x=True)
        if len(levels) > 1:
            plot_delta(levels, "speech_db_spl", None, "speech level (dB SPL)",
                       f"ΔWER, {label_a} − {label_b}", out / "delta_vs_level.png")

        print(f"\nSaved to {out}")
        for f in sorted(out.iterdir()):
            print(f"  {f.name}")
    finally:
        sys.stdout = tee.stream
        (out / "report.txt").write_text("".join(tee.lines))


if __name__ == "__main__":
    main()