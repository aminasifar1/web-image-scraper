#!/usr/bin/env python3
"""Classify Wayback image folders year by year and generate comparison plots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import EndpointHandler


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _collect_year_folders(root_dir: Path) -> list[Path]:
    year_folders = [p for p in sorted(root_dir.iterdir()) if p.is_dir() and p.name.isdigit()]
    return year_folders


def _collect_images(year_dir: Path) -> list[Path]:
    return sorted(
        p for p in year_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    )


def _classify_images(image_paths: list[Path], model_dir: Path, threshold: float) -> pd.DataFrame:
    os.environ["SPAI_THRESHOLD"] = str(threshold)
    handler = EndpointHandler(path=str(model_dir))

    rows: list[dict[str, Any]] = []
    failures = 0

    print(f"    Classifying {len(image_paths)} images", flush=True)

    for index, img_path in enumerate(image_paths, start=1):
        try:
            pred = handler({"inputs": str(img_path)})
            rows.append(
                {
                    "image_input": str(img_path),
                    "score": float(pred["score"]),
                    "predicted_label": int(pred["predicted_label"]),
                    "predicted_label_name": pred["predicted_label_name"],
                    "threshold": float(pred["threshold"]),
                }
            )
        except Exception:
            failures += 1

        if index == 1 or index % 20 == 0 or index == len(image_paths):
            print(f"      Processed {index}/{len(image_paths)} images", flush=True)

    if not rows:
        raise RuntimeError("No images could be classified")

    df = pd.DataFrame(rows)
    df.attrs["failures"] = failures
    return df


def _save_year_histogram(df: pd.DataFrame, year: str, threshold: float, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    scores = df["score"].astype(float)
    ax.hist(scores, bins=20, color="#1971c2", edgecolor="white", alpha=0.9)
    ax.axvline(threshold, color="#c92a2a", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    ax.set_title(f"Histogram of scores - {year}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    path = output_dir / f"histogram_{year}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_year_bars(df: pd.DataFrame, year: str, output_dir: Path) -> Path:
    counts = df["predicted_label"].value_counts().reindex([0, 1], fill_value=0)
    real_count = int(counts.loc[0])
    ai_count = int(counts.loc[1])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(["Real", "IA"], [real_count, ai_count], color=["#51cf66", "#ff6b6b"], edgecolor="black")
    ax.set_title(f"Real vs IA - {year}")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(1, max(real_count, ai_count) + max(3, int(max(real_count, ai_count) * 0.15))))

    total = max(1, real_count + ai_count)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(1, total * 0.02),
            f"{int(height)}\n({100 * height / total:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    path = output_dir / f"bars_{year}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_overall_comparison(yearly_summary: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    year_labels = yearly_summary["year"].astype(str).tolist()
    real_counts = yearly_summary["real_count"].to_numpy()
    ai_counts = yearly_summary["ai_count"].to_numpy()
    total_counts = yearly_summary["total_images"].to_numpy()
    ai_share = yearly_summary["ai_share"].to_numpy()
    score_means = yearly_summary["score_mean"].to_numpy()

    plots: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(year_labels))
    ax.bar(x, real_counts, label="Real", color="#51cf66", edgecolor="black")
    ax.bar(x, ai_counts, bottom=real_counts, label="IA", color="#ff6b6b", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels)
    ax.set_ylabel("Count")
    ax.set_title("Images reales vs IA por año")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for idx, total in enumerate(total_counts):
        ax.text(idx, total + max(1, int(total * 0.03)), str(int(total)), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    path = output_dir / "comparison_real_vs_ai_by_year.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots["comparison_real_vs_ai_by_year"] = path

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(year_labels, ai_share * 100.0, marker="o", linewidth=2.5, color="#ff6b6b")
    ax.set_ylim(0, 100)
    ax.set_ylabel("AI share (%)")
    ax.set_title("Crecimiento relativo de imágenes IA por año")
    ax.grid(alpha=0.25)
    for x_label, y_value in zip(year_labels, ai_share * 100.0):
        ax.text(x_label, y_value + 1.2, f"{y_value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    path = output_dir / "ai_share_by_year.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots["ai_share_by_year"] = path

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(year_labels, score_means, color="#1971c2", edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean score")
    ax.set_title("Mean SPAI score por año")
    ax.grid(axis="y", alpha=0.25)
    for year_label, mean_score in zip(year_labels, score_means):
        ax.text(year_label, mean_score + 0.02, f"{mean_score:.3f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    path = output_dir / "mean_score_by_year.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plots["mean_score_by_year"] = path

    fig, axes = plt.subplots(len(year_labels), 1, figsize=(10, max(3.5, 2.8 * len(year_labels))), sharex=True)
    if len(year_labels) == 1:
        axes = [axes]
    for ax, year_label, year_scores in zip(axes, year_labels, yearly_summary["scores_list"]):
        scores = np.asarray(year_scores, dtype=float)
        ax.hist(scores, bins=20, color="#0b7285", edgecolor="white", alpha=0.9)
        ax.axvline(0.6, color="#c92a2a", linestyle="--", linewidth=1.5)
        ax.set_ylabel(year_label)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Score")
    fig.suptitle("Histogramas de score por año", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "histograms_by_year.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots["histograms_by_year"] = path

    return plots


def classify_wayback_years(root_dir: Path, output_dir: Path, model_dir: Path, threshold: float) -> dict[str, Any]:
    year_folders = _collect_year_folders(root_dir)
    if not year_folders:
        raise ValueError(f"No year folders found under: {root_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    year_rows: list[dict[str, Any]] = []
    all_rows: list[pd.DataFrame] = []

    for year_dir in year_folders:
        year = year_dir.name
        image_paths = _collect_images(year_dir)
        if not image_paths:
            continue

        year_output_dir = output_dir / year
        year_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[YEAR {year}] {len(image_paths)} images", flush=True)

        df = _classify_images(image_paths, model_dir=model_dir, threshold=threshold)
        df.insert(0, "year", int(year))
        df.insert(1, "image_path", df.pop("image_input"))
        failures = int(df.attrs.get("failures", 0))

        pred_csv = year_output_dir / f"predictions_{year}.csv"
        df.to_csv(pred_csv, index=False)

        histogram_path = _save_year_histogram(df, year, threshold, year_output_dir)
        bars_path = _save_year_bars(df, year, year_output_dir)

        total = int(len(df))
        real_count = int((df["predicted_label"] == 0).sum())
        ai_count = int((df["predicted_label"] == 1).sum())
        ai_share = float(ai_count / total) if total else 0.0

        year_rows.append(
            {
                "year": int(year),
                "total_images": total,
                "real_count": real_count,
                "ai_count": ai_count,
                "ai_share": ai_share,
                "real_share": float(real_count / total) if total else 0.0,
                "score_mean": float(df["score"].mean()),
                "score_median": float(df["score"].median()),
                "score_min": float(df["score"].min()),
                "score_max": float(df["score"].max()),
                "failures": failures,
                "predictions_csv": str(pred_csv),
                "histogram_png": str(histogram_path),
                "bars_png": str(bars_path),
                "scores_list": df["score"].astype(float).tolist(),
            }
        )
        all_rows.append(df)

    if not year_rows:
        raise RuntimeError("No images were classified from any year folder")

    yearly_summary = pd.DataFrame(year_rows).sort_values("year").reset_index(drop=True)
    comparison_paths = _save_overall_comparison(yearly_summary, output_dir)

    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_csv = output_dir / "wayback_all_years_predictions.csv"
    combined_df.to_csv(combined_csv, index=False)

    yearly_summary_for_json = yearly_summary.drop(columns=["scores_list"]).copy()
    summary_json = {
        "root_dir": str(root_dir),
        "output_dir": str(output_dir),
        "threshold": float(threshold),
        "years": yearly_summary_for_json.to_dict(orient="records"),
        "combined_predictions_csv": str(combined_csv),
        "artifacts": {
            **{k: str(v) for k, v in comparison_paths.items()},
        },
    }
    (output_dir / "wayback_yearly_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    yearly_summary_for_json.to_csv(output_dir / "wayback_yearly_summary.csv", index=False)

    return summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Wayback root folder containing year subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where predictions and plots will be written",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/fhome/aaasidar/spai-hf"),
        help="SPAI model root directory",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="Decision threshold")
    args = parser.parse_args()

    summary = classify_wayback_years(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        threshold=args.threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()