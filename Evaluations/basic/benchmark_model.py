"""
Benchmark one trained model: everything you need to know about it, in one run.

Point MODEL at a trained model (its .pt, its _meta.json, or just its name), set the globals
below, then press Run in VS Code (or `python benchmark_model.py`). The script:

    1. prints what the model is: front end, model, data, training recipe, environment
    2. summarizes and plots its training history (loss, WER/CER, learning rate, epoch time)
    3. scores clean speech on the validation and test sets, with 95% confidence intervals,
       and shows its best and worst transcriptions
    4. sweeps noise type x SNR (every noise found in noise/, plus white noise)
    5. sweeps the speech level (dB SPL)
    6. saves tables and plots to results/<name>/benchmark/, and a copy of everything printed

Every evaluation is also recorded in models/<name>_meta.json and results/<name>/per_utt/.

Metrics for every condition: WER (with a 95% bootstrap interval), CER, PER (phoneme error rate:
did the model hear the right sounds, even when it misspelled the word?) and the word errors split
into substitutions / deletions / insertions (S/D/I, which add up to WER). PER needs g2p_en:
    pip install --no-deps g2p_en nltk inflect
Without it, everything else still runs and PER shows as "—".

Cost: log-mel and MFCC are computed on the fly, so each condition takes about a minute on a GPU.
CARFAC needs a feature cache for EVERY condition (one full pass over test-clean, about 5.4 h of
audio, each), so the full grid can take hours the first time; caches are reused afterwards.
MAX_UTTS shortens scoring but not CARFAC cache building: shrink SNRS / LEVELS / NOISES for a quick look.

Author: Eric Oliveira
Date: 11/02/2026

* Written with the help of Claude *

"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # Draw to files: works on Quest without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from asrlib import ASR, CLEAN, Condition, NoiseBank
from asrlib.metrics import _g2p

# Print immediately, line by line, even into a Slurm log
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# SETTINGS
# =============================================================================

# The model: a .pt file, a _meta.json file, or a model name (looked up in <ROOT>/models/)
MODEL = "models/2026-11-03_carfac-agc-log_librispeech-train.100_clean_s30_64fdc8_meta.json"

# Workspace root. None: the folder that holds models/ (inferred from MODEL), else the current folder
ROOT = None

# "cuda", "cpu", or None: CUDA if available, else CPU (never MPS: it has no CTC loss)
DEVICE = None

# Clean speech is scored on these splits (names as in DataConfig)
CLEAN_SPLITS = ("validation", "test")

# Noise sweep, on the test split. Kinds without recordings in noise/ are skipped ("white" is always there)
NOISES = ("white",) #("white", "pub", "babble", "pub_whisper")
SNRS = (40, 30, 20, 10, 5, 0, -5, -10)      # dB, the same grid as Whisper's noise robustness figure

# Level sweep of clean speech, dB SPL (65 = the default training level)
LEVELS = (35, 45, 55, 65, 75, 85, 95)

# Score only the first N utterances of each condition (None = all). Quick looks only
MAX_UTTS = None

# How many best / worst transcriptions to show
N_EXAMPLES = 5


# =============================================================================
# Helpers
# =============================================================================

class Tee:
    """Write to the terminal and to a list at the same time, so the report can be saved."""

    def __init__(self, stream):
        self.stream = stream
        self.lines: list[str] = []

    def write(self, text):
        self.stream.write(text)
        self.lines.append(text)

    def flush(self):
        self.stream.flush()


def header(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def resolve_model(model: str, root: str | None) -> tuple[Path, Path]:
    """MODEL (a .pt, a _meta.json or a name) → (path to .pt, workspace root)."""
    path = Path(model).expanduser()

    # A meta file: its .pt sits next to it
    if path.name.endswith("_meta.json"):
        path = path.with_name(path.name[: -len("_meta.json")] + ".pt")

    # A bare name: look in <root>/models/. (endswith, not .suffix: names contain dots like "train.100")
    elif not path.name.endswith(".pt"):
        path = Path(root or Path.cwd()).expanduser() / "models" / f"{path.name}.pt"

    path = path.resolve()
    if not path.exists():
        sys.exit(f"Model not found: {path}")

    # The workspace is the folder that holds models/, unless ROOT says otherwise
    if root is None:
        root = path.parent.parent if path.parent.name == "models" else Path.cwd()
    return path, Path(root).expanduser().resolve()


def pick_device(device: str | None) -> str:
    """Explicit choice, else CUDA, else CPU. MPS is skipped: CTC loss doesn't exist there."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_dict(d: dict, indent: int = 2, skip: tuple = ()) -> None:
    """One aligned 'key  value' line per field."""
    width = max((len(k) for k in d if k not in skip), default=0)
    for k, v in d.items():
        if k not in skip:
            print(f"{' ' * indent}{k:<{width}}  {v}")


def pct(x) -> str:
    """0.1234 → '12.34%', NaN → '—' (PER without g2p_en)."""
    return "—" if pd.isna(x) else f"{x:.2%}"


def fmt_types(row) -> str:
    """'S/D/I 20.1/5.3/1.2' (percent of reference words; they add up to WER)."""
    if pd.isna(row.get("sub_rate")):
        return ""
    return f"S/D/I {row['sub_rate'] * 100:4.1f}/{row['del_rate'] * 100:4.1f}/{row['ins_rate'] * 100:4.1f}"


def fmt_ci(row) -> str:
    """'12.34%  [11.80–12.90]' from a benchmark row (CI missing → just the WER)."""
    if pd.notna(row.get("wer_lo")) and pd.notna(row.get("wer_hi")):
        return f"{row['wer']:6.2%}  [{row['wer_lo']:.2%}–{row['wer_hi']:.2%}]"
    return f"{row['wer']:6.2%}"


# =============================================================================
# 1. What the model is
# =============================================================================

def describe(asr: ASR, meta: dict) -> None:
    header(f"MODEL  {asr.name}")
    print(asr.summary())

    header("IDENTITY")
    print_dict({"status": meta.get("status"), "saved_at": meta.get("saved_at"),
                "config_hash": meta.get("config_hash"),
                "controlled_hash": f"{meta.get('controlled_hash')}   (must match across a front-end comparison)",
                "seed": meta.get("seed")})

    header("FRONT END")
    print_dict(meta["frontend"])

    header("MODEL")
    print_dict({**meta["model"], "n_params": f"{meta['model']['n_params']:,}"})

    header("DATA")
    print_dict(meta["dataset"])
    for split in (asr.cfg.data.train_split, *CLEAN_SPLITS):
        corpus = asr.corpus(split)
        hours = corpus.durations().sum() / 16000 / 3600
        print(f"  {split:<12s} {len(corpus):6d} utterances, {hours:6.1f} h")

    header("TRAINING RECIPE")
    print_dict(meta["training"], skip=("history", "best_epoch", "best_val_wer", "epochs_run"))
    print_dict({"normalization": meta.get("normalization"), "tokenizer": meta.get("tokenizer")})

    header("ENVIRONMENT (where it was trained)")
    print_dict(meta.get("environment", {}))


# =============================================================================
# 2. Training history
# =============================================================================

def summarize_history(history: pd.DataFrame, meta: dict) -> None:
    header("TRAINING HISTORY")
    if history.empty:
        print("  No history recorded.")
        return

    best = history.loc[history.epoch == meta["training"]["best_epoch"]].iloc[0]
    last = history.iloc[-1]
    planned = meta["training"]["epochs"]

    print_dict({
        "epochs run": f"{len(history)} of {planned}"
                      + ("" if len(history) == planned else "  (stopped early or unfinished)"),
        "best epoch": f"{int(best.epoch)}  →  val WER {best.val_wer:.2%}, CER {best.val_cer:.2%}, "
                      f"loss {best.val_loss:.3f}",
        "last epoch": f"{int(last.epoch)}  →  val WER {last.val_wer:.2%}, CER {last.val_cer:.2%}, "
                      f"loss {last.val_loss:.3f}",
        "train loss": f"{history.train_loss.iloc[0]:.3f} → {last.train_loss:.3f}",
        "val - train loss (last)": f"{last.val_loss - last.train_loss:+.3f}   "
                                   "(large and growing = overfitting)",
        "peak lr": f"{history.lr.max():.2e}",
        "time": f"{history.seconds.sum() / 3600:.2f} h total, {history.seconds.mean():.0f} s per epoch",
    })

    # Did the last epochs still help? Compare the best of the last 5 with the best before them
    if len(history) > 10:
        before, recent = history.val_wer.iloc[:-5].min(), history.val_wer.iloc[-5:].min()
        trend = "still improving" if recent < before else "plateaued"
        print(f"  last 5 epochs           best val WER {recent:.2%} vs {before:.2%} before → {trend}")

    print("\n  epoch   train    val     WER      CER      lr        s")
    for _, r in history.iterrows():
        mark = "  *best" if r.epoch == best.epoch else ""
        print(f"  {int(r.epoch):5d}  {r.train_loss:6.3f}  {r.val_loss:6.3f}  {r.val_wer:6.2%}  "
              f"{r.val_cer:6.2%}  {r.lr:.2e}  {r.seconds:5.0f}{mark}")


def plot_history(history: pd.DataFrame, best_epoch: int, name: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (ax_loss, ax_wer), (ax_lr, ax_time) = axes

    # Loss: a growing gap between the curves means overfitting
    ax_loss.plot(history.epoch, history.train_loss, label="train")
    ax_loss.plot(history.epoch, history.val_loss, label="validation")
    ax_loss.set(title="CTC loss", xlabel="epoch", ylabel="loss")
    ax_loss.legend()

    # Validation error rates, with the chosen (best) epoch marked
    ax_wer.plot(history.epoch, history.val_wer * 100, label="WER")
    ax_wer.plot(history.epoch, history.val_cer * 100, label="CER")
    ax_wer.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best (epoch {best_epoch})")
    ax_wer.set(title="Validation error", xlabel="epoch", ylabel="%")
    ax_wer.legend()

    # Learning rate at the start of each epoch: warmup, then cosine
    ax_lr.plot(history.epoch, history.lr)
    ax_lr.set(title="Learning rate", xlabel="epoch", ylabel="lr")

    ax_time.bar(history.epoch, history.seconds / 60)
    ax_time.set(title="Time per epoch", xlabel="epoch", ylabel="minutes")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =============================================================================
# 3. Clean speech
# =============================================================================

def clean_scores(asr: ASR) -> pd.DataFrame:
    header("CLEAN SPEECH (at the training level)")
    rows = []
    for split in CLEAN_SPLITS:
        r = asr.evaluate(split=split, condition=asr.clean, max_utts=MAX_UTTS)
        lo, hi = r.wer_ci
        print(f"\n  {split}: WER {r.wer:.2%}  (95% CI {lo:.2%}–{hi:.2%})  CER {r.cer:.2%}  PER {pct(r.per)}  "
              f"loss {r.loss:.3f}  over {r.n_utts} utterances")
        print(f"    word errors            substitutions {r.sub_rate:.2%}, deletions {r.del_rate:.2%}, "
              f"insertions {r.ins_rate:.2%}  (sum = WER)")

        # How the errors are spread over utterances
        per = r.per_utt.assign(utt_wer=lambda d: d.w_err / d.w_ref.clip(lower=1))
        print(f"    perfect utterances     {(per.w_err == 0).mean():.1%}")
        print(f"    median utterance WER   {per.utt_wer.median():.2%}")
        print(f"    utterances above 50%   {(per.utt_wer > 0.5).mean():.1%}")

        # PER per utterance: how often the sounds were right even when the words weren't
        if per.p_err.notna().all():
            per = per.assign(utt_per=lambda d: d.p_err / d.p_ref.clip(lower=1))
            print(f"    median utterance PER   {per.utt_per.median():.2%}  "
                  f"(vs median WER {per.utt_wer.median():.2%}: the gap is mostly spelling)")

        # Only the test split gets examples, to keep the report readable
        if split == CLEAN_SPLITS[-1]:
            show_examples(per)

        rows.append({"split": split, "condition": "clean", **r.row()})
    return pd.DataFrame(rows)


def show_examples(per: pd.DataFrame) -> None:
    """The worst transcriptions (by utterance WER, ties → more errors) and a few typical ones."""
    worst = per.sort_values(["utt_wer", "w_err"], ascending=False).head(N_EXAMPLES)
    print(f"\n    {N_EXAMPLES} worst utterances")
    for _, r in worst.iterrows():
        per_txt = f", PER {r.utt_per:.0%}" if "utt_per" in r and pd.notna(r.utt_per) else ""
        print(f"      {r.utt_id}  WER {r.utt_wer:.0%} ({r.w_err}/{r.w_ref} words){per_txt}")
        print(f"        ref: {r.ref[:110]}")
        print(f"        hyp: {r.hyp[:110]}")

    # Typical: closest to the median utterance WER, among utterances with some errors
    with_errors = per[per.w_err > 0]
    if not with_errors.empty:
        median = with_errors.utt_wer.median()
        typical = with_errors.iloc[(with_errors.utt_wer - median).abs().argsort()[:N_EXAMPLES]]
        print(f"\n    {N_EXAMPLES} typical utterances (with errors)")
        for _, r in typical.iterrows():
            print(f"      {r.utt_id}  WER {r.utt_wer:.0%}")
            print(f"        ref: {r.ref[:110]}")
            print(f"        hyp: {r.hyp[:110]}")


# =============================================================================
# 4–5. Noise and level sweeps
# =============================================================================

def noise_sweep(asr: ASR) -> pd.DataFrame:
    header(f"NOISE SWEEP (test split, speech at {asr.clean.speech_db_spl:g} dB SPL)")
    available = NoiseBank(asr.workspace.noise).available()
    kinds = [k for k in NOISES if k in available]
    missing = [k for k in NOISES if k not in available]
    if missing:
        print(f"  Skipping {missing}: no recordings in {asr.workspace.noise}/<kind>/")
    if not kinds:
        return pd.DataFrame()

    conditions = [Condition(k, s, speech_db_spl=asr.clean.speech_db_spl) for k in kinds for s in SNRS]
    return asr.benchmark(conditions, max_utts=MAX_UTTS)


def level_sweep(asr: ASR) -> pd.DataFrame:
    header("LEVEL SWEEP (test split, clean speech)")
    return asr.benchmark([Condition(speech_db_spl=level) for level in LEVELS], max_utts=MAX_UTTS)


def summarize_noise(noise: pd.DataFrame, clean_wer: float) -> None:
    if noise.empty:
        return
    header("NOISE SUMMARY: WER [95% CI] by noise and SNR")
    for kind, group in noise.groupby("noise", sort=False):
        print(f"\n  {kind}")
        for _, r in group.sort_values("snr_db", ascending=False).iterrows():
            print(f"    {r.snr_db:5g} dB   {fmt_ci(r)}   ×{r.wer / clean_wer:4.1f} clean   "
                  f"PER {pct(r.per):>6s}   {fmt_types(r)}")

        # Two robustness numbers: where the error rate doubles, and how far down it stays usable
        g = group.sort_values("snr_db", ascending=False)
        doubled = g[g.wer >= 2 * clean_wer].snr_db
        usable = g[g.wer < 0.5].snr_db
        print(f"    WER doubles at {'≤ ' + format(doubled.max(), 'g') + ' dB' if len(doubled) else 'no SNR tested'}; "
              f"WER < 50% down to {format(usable.min(), 'g') + ' dB' if len(usable) else 'no SNR tested'}")

    # Compact tables: rows = noise, columns = SNR
    for metric in ("wer", "per"):
        if noise[metric].isna().all():
            continue
        table = noise.pivot_table(index="noise", columns="snr_db", values=metric, sort=False) * 100
        table = table[sorted(table.columns, reverse=True)]
        print(f"\n  {metric.upper()} % (columns: SNR in dB)")
        print("  " + table.round(1).to_string().replace("\n", "\n  "))


def summarize_levels(levels: pd.DataFrame, clean_wer: float) -> None:
    header("LEVEL SUMMARY: WER [95% CI] by speech level")
    for _, r in levels.sort_values("speech_db_spl").iterrows():
        print(f"    {r.speech_db_spl:5g} dB SPL   {fmt_ci(r)}   ×{r.wer / clean_wer:4.1f} clean   "
              f"PER {pct(r.per):>6s}   {fmt_types(r)}")


def plot_sweep(df: pd.DataFrame, x: str, group: str | None, clean: pd.Series,
               xlabel: str, title: str, out: Path, reverse_x: bool = False) -> None:
    """WER (with 95% bootstrap bars) and PER vs x, side by side; dashed lines = clean speech."""
    metrics = [("wer", "WER (%)")] + ([("per", "PER (%)")] if df["per"].notna().all() else [])
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.5), squeeze=False)
    for ax, (metric, ylabel) in zip(axes[0], metrics):
        groups = df.groupby(group, sort=False) if group else [("clean speech", df)]
        for label, g in groups:
            g = g.sort_values(x)
            y = g[metric] * 100
            # Error bars from the bootstrap interval (computed for WER only)
            if metric == "wer" and g.wer_lo.notna().all():
                err = np.vstack([y - g.wer_lo * 100, g.wer_hi * 100 - y])
                ax.errorbar(g[x], y, yerr=err, marker="o", capsize=3, label=label)
            else:
                ax.plot(g[x], y, marker="o", label=label)
        ax.axhline(clean[metric] * 100, color="gray", ls="--", lw=1, label=f"clean ({clean[metric]:.1%})")
        if reverse_x:
            ax.invert_xaxis()                    # High SNR (easy) on the left, as in most papers
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"{metric.upper()} — {title}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_error_types(df: pd.DataFrame, x: str, group: str | None, xlabel: str, title: str,
                     out: Path, reverse_x: bool = False) -> None:
    """Stacked bars of substitutions, deletions and insertions (their sum = WER), one panel per noise.

    Shows HOW a model fails: wrong words (S), missing words (D: typical of a model that falls silent
    in heavy noise) or made-up words (I: typical of a model that hears speech in the noise).
    """
    groups = list(df.groupby(group, sort=False)) if group else [("clean speech", df)]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.2), squeeze=False, sharey=True)
    for ax, (label, g) in zip(axes[0], groups):
        g = g.sort_values(x, ascending=not reverse_x)
        pos = np.arange(len(g))
        bottom = np.zeros(len(g))
        for col, name in (("sub_rate", "substitutions"), ("del_rate", "deletions"), ("ins_rate", "insertions")):
            vals = g[col].to_numpy() * 100
            ax.bar(pos, vals, bottom=bottom, label=name)
            bottom += vals
        ax.set_xticks(pos, [f"{v:g}" for v in g[x]])
        ax.set(xlabel=xlabel, title=str(label))
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("% of reference words (sum = WER)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))     # Outside the bars
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =============================================================================
# Run
# =============================================================================

def main():

    print("\nStarting evaluation and benchmarking...\n")
    pt_path, root = resolve_model(MODEL, ROOT)
    device = pick_device(DEVICE)

    asr = ASR.load(pt_path, root=root, device=device)
    meta_path = pt_path.with_name(f"{pt_path.stem}_meta.json")
    meta = json.loads(meta_path.read_text())

    # Everything goes to results/<name>/benchmark/, including a copy of this printout
    out = asr.workspace.results / asr.name / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout)
    sys.stdout = tee

    try:
        print(f"Benchmarking {pt_path}\n  workspace {root}\n  device    {device}\n  output    {out}")
        print(f"  PER       {'on (g2p_en)' if _g2p() is not None else 'off: pip install --no-deps g2p_en nltk inflect'}")

        # CARFAC builds one test-set cache per condition the first time: say what's coming
        if asr.frontend.expensive:
            n_conditions = len(NOISES) * len(SNRS) + len(LEVELS)
            print(f"\n  NOTE: {asr.frontend.name} needs a feature cache per condition. Up to "
                  f"{n_conditions} test-set caches will be built if they don't exist yet.")

        # 1–2. What it is and how it trained
        describe(asr, meta)
        history = pd.DataFrame(meta["training"]["history"])
        summarize_history(history, meta)
        if not history.empty:
            plot_history(history, meta["training"]["best_epoch"], asr.name, out / "history.png")

        # 3. Clean speech
        clean = clean_scores(asr)
        clean_row = clean.loc[clean.split == asr.cfg.data.test_split].iloc[0]
        clean_wer = float(clean_row.wer)

        # 4. Noise
        noise = noise_sweep(asr)
        summarize_noise(noise, clean_wer)
        if not noise.empty:
            plot_sweep(noise, "snr_db", "noise", clean_row, "SNR (dB)",
                       asr.frontend.name, out / "wer_vs_snr.png", reverse_x=True)
            plot_error_types(noise, "snr_db", "noise", "SNR (dB)",
                             f"Word error types vs SNR — {asr.frontend.name}", out / "error_types_vs_snr.png",
                             reverse_x=True)

        # 5. Level
        levels = level_sweep(asr)
        summarize_levels(levels, clean_wer)
        plot_sweep(levels, "speech_db_spl", None, clean_row, "speech level (dB SPL)",
                   asr.frontend.name, out / "wer_vs_level.png")
        plot_error_types(levels, "speech_db_spl", None, "speech level (dB SPL)",
                         f"Word error types vs level — {asr.frontend.name}", out / "error_types_vs_level.png")

        # 6. Save the tables: benchmark.csv concatenates across models with pd.concat
        pd.concat([noise, levels], ignore_index=True).to_csv(out / "benchmark.csv", index=False)
        clean.to_csv(out / "clean.csv", index=False)
        history.to_csv(out / "history.csv", index=False)

        header("FILES")
        for f in sorted(out.iterdir()):
            print(f"  {f}")
        print(f"  {out / 'report.txt'}")
    finally:
        # Restore the terminal and save the report, even if something failed halfway
        sys.stdout = tee.stream
        (out / "report.txt").write_text("".join(tee.lines))


if __name__ == "__main__":
    main()