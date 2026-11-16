"""Plot the README figures from the test summary written by `just evaluate split=test`.

Usage: just plot-results [path/to/test_summary.csv]
Writes docs/figures/test_top1.png and docs/figures/ablations.png.
"""

# matplotlib is not a project dependency: the just recipe runs this with `uv run --with`.
# pyright: reportMissingImports=false

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

TEXT, IMAGE, FUSION, BEST = "#2B6CB0", "#DD6B20", "#0F766E", "#B91C1C"
INK, MUTED = "#23373B", "#6B7280"

# (method, condition, label, colour), bottom to top.
METHODS = [
    ("image_mlp", "image_only", "Image only (frozen ViT)", IMAGE),
    ("text_mlp", "text_only", "Text only (frozen BERT)", TEXT),
    ("tfidf_logreg", "text_only", "Text only (TF-IDF)", TEXT),
    ("text_bert_ft", "text_only", "Text only (fine-tuned BERT)", TEXT),
    ("early_concat", "both", "Early fusion", FUSION),
    ("cross_attention", "both", "Cross-attention", FUSION),
    ("late_calibrated", "both", "Late, frozen BERT + image", FUSION),
    ("late_calibrated_tfidf", "both", "Late, TF-IDF + image", FUSION),
    ("late_calibrated_ft", "both", "Late, fine-tuned BERT + image", BEST),
]
REFERENCE = 93.8  # public repository: logit sum tuned on the test split, one seed

ABLATED = ["early_concat", "cross_attention", "late_calibrated"]
ABLATION_LABELS = ["Early", "Cross-attention", "Late"]
CONDITIONS = [
    ("both", "Text + image", "#5B6B70"),
    ("image_mean", "Image removed", TEXT),
    ("text_mean", "Text removed", IMAGE),
]


def read_top1(path: Path) -> dict[tuple[str, str], float]:
    with path.open(encoding="utf-8") as f:
        return {
            (row["method"], row["condition"]): 100 * float(row["mean"])
            for row in csv.DictReader(f)
            if row["subset"] == "all" and row["metric"] == "top1"
        }


def style(ax: plt.Axes) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(colors=INK, length=0)


def plot_methods(top1: dict[tuple[str, str], float], out: Path) -> None:
    rows = [(label, top1[(m, c)], colour) for m, c, label, colour in METHODS if (m, c) in top1]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    for y, (_, value, colour) in enumerate(rows):
        ax.hlines(y, 70, value, color="#EEEEEE", lw=1, zorder=0)
        ax.scatter(value, y, s=90 if colour == BEST else 55, color=colour, zorder=2)
        ax.text(
            value + 0.4,
            y,
            f"{value:.1f}",
            va="center",
            fontsize=9,
            color=INK,
            zorder=3,
            fontweight="bold" if colour == BEST else "normal",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        )
    ax.axvline(REFERENCE, color=MUTED, ls="--", lw=1, zorder=1)
    ax.text(
        REFERENCE - 0.3,
        len(rows) - 0.35,
        "public reference 93.8\n(tuned on test, 1 seed)",
        ha="right",
        va="top",
        fontsize=7.5,
        color=MUTED,
    )
    ax.axhline(3.5, color="#BBBBBB", lw=0.8)
    ax.set_yticks(range(len(rows)), [label for label, _, _ in rows], fontsize=9)
    ax.set_xlim(70, 97)
    ax.set_xlabel("Test top-1 accuracy (%), mean of 3 seeds", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_ablations(top1: dict[tuple[str, str], float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    width = 0.26
    for k, (condition, label, colour) in enumerate(CONDITIONS):
        for i, method in enumerate(ABLATED):
            value = top1.get((method, condition))
            if value is None:
                continue
            x = i + (k - 1) * width
            ax.bar(x, value, width * 0.92, color=colour, label=label if i == 0 else None, zorder=2)
            ax.text(
                x,
                value + 1,
                f"{value:.1f}",
                ha="center",
                fontsize=8,
                color=INK,
                zorder=3,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            )
    for value, colour, label in [
        (top1.get(("text_mlp", "text_only")), TEXT, "text only"),
        (top1.get(("image_mlp", "image_only")), IMAGE, "image only"),
    ]:
        if value is not None:
            ax.axhline(value, color=colour, ls="--", lw=1, zorder=1)
            ax.text(2.45, value, f"{label} {value:.1f}", va="center", fontsize=7.5, color=colour)
    ax.set_xticks(range(len(ABLATED)), ABLATION_LABELS)
    ax.set_xlim(-0.5, 2.42)
    ax.set_ylim(0, 105)
    ax.set_yticks([])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    summary = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/results/test_summary.csv")
    top1 = read_top1(summary)
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    plot_methods(top1, out / "test_top1.png")
    plot_ablations(top1, out / "ablations.png")
    print(f"wrote {out}/test_top1.png and {out}/ablations.png")


if __name__ == "__main__":
    main()
