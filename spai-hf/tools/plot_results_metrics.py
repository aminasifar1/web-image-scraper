#!/usr/bin/env python3
"""Generate score, accuracy, and confusion matrix plots from a results CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def _pick_ground_truth_column(df: pd.DataFrame) -> str:
    for column in ("gt_label", "label", "gt_label_raw"):
        if column in df.columns:
            return column
    raise ValueError("Could not find a ground-truth label column")


def _pick_prediction_column(df: pd.DataFrame) -> str:
    for column in ("predicted_label", "prediction", "pred"):
        if column in df.columns:
            return column
    raise ValueError("Could not find a prediction column")


def _pick_score_column(df: pd.DataFrame) -> str:
    for column in ("score", "spai", "confidence", "prediction"):
        if column in df.columns:
            return column
    raise ValueError("Could not find a score column")


def _label_name_map(df: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {0: "real", 1: "ai-generated"}
    for label_col, name_col in (("gt_label", "gt_label_name"), ("predicted_label", "predicted_label_name")):
        if label_col not in df.columns or name_col not in df.columns:
            continue
        named = df[[label_col, name_col]].dropna().drop_duplicates()
        for _, row in named.iterrows():
            try:
                mapping[int(row[label_col])] = str(row[name_col])
            except Exception:
                continue

    return mapping


def _save_score_histogram(df: pd.DataFrame, score_col: str, threshold: float, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scores = df[score_col].astype(float)

    ax.hist(scores, bins=30, color="#1971c2", edgecolor="white", alpha=0.9)
    ax.axvline(threshold, color="#c92a2a", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    ax.set_title("Score histogram")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "score_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_accuracy_plot(accuracy: float, total: int, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["Test accuracy"], [accuracy], color="#2f9e44", width=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Test accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0, accuracy + 0.03, f"{accuracy:.3f}\n(n={total})", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()

    path = output_dir / "test_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_real_vs_ai_bars(
    y_pred: np.ndarray,
    label_map: dict[int, str],
    output_dir: Path,
) -> Path:
    counts = pd.Series(y_pred).value_counts().reindex([0, 1], fill_value=0)
    real_label = label_map.get(0, "real")
    ai_label = label_map.get(1, "ai-generated")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(
        [real_label, ai_label],
        [int(counts.loc[0]), int(counts.loc[1])],
        color=["#51cf66", "#ff6b6b"],
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_title("Images reales vs IA")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(1, int(counts.max()) + max(5, int(counts.max() * 0.1))))

    total = max(1, int(counts.sum()))
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(1, total * 0.01),
            f"{int(height)}\n({100 * height / total:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    path = output_dir / "real_vs_ai_bars.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[int, str],
    output_dir: Path,
) -> Path:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels([label_map.get(label, str(label)) for label in labels], rotation=15)
    ax.set_yticklabels([label_map.get(label, str(label)) for label in labels])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")

    max_value = max(1, int(cm.max()))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > max_value / 2 else "black",
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_overview(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    accuracy: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[int, str],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    scores = df[score_col].astype(float)
    axes[0].hist(scores, bins=30, color="#1971c2", edgecolor="white", alpha=0.9)
    axes[0].axvline(threshold, color="#c92a2a", linestyle="--", linewidth=2)
    axes[0].set_title("Score histogram")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.2)

    axes[1].bar(["Test accuracy"], [accuracy], color="#2f9e44", width=0.6)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(0, accuracy + 0.03, f"{accuracy:.3f}", ha="center", va="bottom", fontweight="bold")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    image = axes[2].imshow(cm, cmap="Blues", interpolation="nearest")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels([label_map.get(0, "0"), label_map.get(1, "1")], rotation=15)
    axes[2].set_yticklabels([label_map.get(0, "0"), label_map.get(1, "1")])
    axes[2].set_title("Confusion matrix")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    max_value = max(1, int(cm.max()))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > max_value / 2 else "black",
                fontweight="bold",
            )
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Results overview", fontsize=16, fontweight="bold")
    fig.tight_layout()

    path = output_dir / "results_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_plots(csv_path: Path, output_dir: Path, threshold: float | None = None) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")

    score_col = _pick_score_column(df)
    pred_col = _pick_prediction_column(df)
    gt_col = _pick_ground_truth_column(df)

    if threshold is None:
        if "threshold" in df.columns and not df["threshold"].dropna().empty:
            threshold = float(df["threshold"].dropna().iloc[0])
        else:
            threshold = 0.5

    y_true = df[gt_col].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy()
    scores = df[score_col].astype(float).to_numpy()
    accuracy = accuracy_score(y_true, y_pred)

    label_map = _label_name_map(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    histogram_path = _save_score_histogram(df, score_col, threshold, output_dir)
    accuracy_path = _save_accuracy_plot(accuracy, len(df), output_dir)
    bars_path = _save_real_vs_ai_bars(y_pred, label_map, output_dir)
    cm_path = _save_confusion_matrix(y_true, y_pred, label_map, output_dir)
    overview_path = _save_overview(df, score_col, threshold, accuracy, y_true, y_pred, label_map, output_dir)

    summary = {
        "csv_path": str(csv_path),
        "rows": int(len(df)),
        "score_column": score_col,
        "prediction_column": pred_col,
        "ground_truth_column": gt_col,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "predicted_counts": {str(k): int(v) for k, v in pd.Series(y_pred).value_counts().sort_index().items()},
        "ground_truth_counts": {str(k): int(v) for k, v in pd.Series(y_true).value_counts().sort_index().items()},
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "artifacts": {
            "score_histogram": str(histogram_path),
            "test_accuracy": str(accuracy_path),
            "real_vs_ai_bars": str(bars_path),
            "confusion_matrix": str(cm_path),
            "overview": str(overview_path),
        },
    }

    summary_path = output_dir / "results_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results.csv"), help="Path to the results CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("results_plots"), help="Directory for plots")
    parser.add_argument("--threshold", type=float, default=None, help="Optional decision threshold for the histogram")
    args = parser.parse_args()

    summary = generate_plots(args.csv, args.output_dir, args.threshold)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()