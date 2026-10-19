#!/usr/bin/env python
"""Exploratory data analysis: saves figures to reports/figures/eda/.

    python scripts/eda.py [--data data/processed/heart_clean.csv]

Figures
    01_class_balance.png      target distribution
    02_numeric_histograms.png histograms by class (KDE overlay)
    03_categorical_rates.png  disease rate per category level
    04_correlation_heatmap.png
    05_boxplots.png           numeric features vs. target (outlier view)
    06_missing_values.png     missing-value counts
Also prints a text summary (shape, missing values, describe) to stdout and
saves it to reports/eda_summary.txt.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart import config  # noqa: E402

PALETTE = {0: "#4c78a8", 1: "#e45756"}
LABELS = {0: "No disease", 1: "Disease"}


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=150)
    plt.close(fig)


def class_balance(df, out):
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(
        [LABELS[i] for i in counts.index], counts.values, color=[PALETTE[i] for i in counts.index]
    )
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_ylabel("patients")
    ax.set_title("Class balance")
    save(fig, out, "01_class_balance.png")


def numeric_histograms(df, out):
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(
            data=df,
            x=col,
            hue=config.TARGET,
            palette=PALETTE,
            kde=True,
            bins=20,
            ax=ax,
            stat="count",
            common_norm=False,
            legend=False,
        )
        ax.set_title(col)
    # One explicit legend built from the same PALETTE/LABELS used for the colours,
    # so labels can never drift out of sync with the plotted classes.
    handles = [Patch(color=PALETTE[k], label=LABELS[k], alpha=0.8) for k in (0, 1)]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Numeric feature distributions by class", y=1.0, x=0.4)
    save(fig, out, "02_numeric_histograms.png")


def categorical_rates(df, out):
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.ravel(), cols):
        rate = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        ax.bar(rate.index.astype(str), rate["mean"], color="#e45756")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{col}: disease rate")
    for ax in axes.ravel()[len(cols) :]:
        ax.axis("off")
    save(fig, out, "03_categorical_rates.png")


def correlation_heatmap(df, out):
    corr = df[config.FEATURES + [config.TARGET]].corr()
    mask = None
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        mask=mask,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation heatmap (Pearson)")
    save(fig, out, "04_correlation_heatmap.png")


def boxplots(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC_FEATURES):
        sns.boxplot(
            data=df, x=config.TARGET, y=col, hue=config.TARGET, palette=PALETTE, legend=False, ax=ax
        )
        ax.set_xticks([0, 1], ["No disease", "Disease"])
        ax.set_xlabel("")
        ax.set_title(col)
    save(fig, out, "05_boxplots.png")


def missing_values(df, out):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if len(miss):
        ax.bar(miss.index, miss.values, color="#f58518")
        for i, v in enumerate(miss.values):
            ax.text(i, v + 0.05, str(v), ha="center")
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    ax.set_title("Missing values per column")
    save(fig, out, "06_missing_values.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=config.CLEAN_FILE)
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR / "eda")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(args.data)

    summary = [
        f"Shape: {df.shape}",
        f"\nMissing values:\n{df.isna().sum().to_string()}",
        f"\nClass balance:\n{df[config.TARGET].value_counts(normalize=True).round(3).to_string()}",
        f"\nDescribe:\n{df.describe().T.round(2).to_string()}",
        "\nTop correlations with target:\n"
        + df.corr()[config.TARGET]
        .drop(config.TARGET)
        .sort_values(key=abs, ascending=False)
        .round(3)
        .to_string(),
    ]
    text = "\n".join(summary)
    print(text)
    (config.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "eda_summary.txt").write_text(text)

    for fn in (
        class_balance,
        numeric_histograms,
        categorical_rates,
        correlation_heatmap,
        boxplots,
        missing_values,
    ):
        fn(df, args.out)
    print(f"\nFigures saved in {args.out}")


if __name__ == "__main__":
    main()
