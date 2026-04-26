#!/usr/bin/env python3
"""
Complete classification and analysis pipeline for crawled images.
Classifies by category, generates per-category and global statistics, and creates comprehensive plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import os

from plot_style import (
    CATEGORY_COLORS,
    COLOR_AI,
    COLOR_REAL,
    COLOR_THRESHOLD,
    STANDARD_DPI,
    apply_plot_style,
)

apply_plot_style()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import EndpointHandler

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CATEGORIES = ["news", "social_media", "arts_illustration", "education_institution", "corporate"]


def classify_images_dir(
    images_dir: Path,
    output_dir: Path,
    model_dir: Path,
    threshold: float,
    category_name: str = "all",
) -> dict[str, Any]:
    """Classify all images in a directory and generate predictions + plots."""
    
    os.environ["SPAI_THRESHOLD"] = str(threshold)
    handler = EndpointHandler(path=str(model_dir))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted([
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    ])
    
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"[{category_name}] Found {len(image_paths)} images to classify...")
    
    rows = []
    per_image_events = []
    failures = 0
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{category_name}] [{idx}/{len(image_paths)}] {img_path}")
        
        try:
            pred = handler({"inputs": str(img_path)})
            rows.append({
                "image_path": str(img_path),
                "score": float(pred["score"]),
                "predicted_label": int(pred["predicted_label"]),
                "predicted_label_name": pred["predicted_label_name"],
                "threshold": float(pred["threshold"]),
            })
            per_image_events.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "category": category_name,
                    "index": idx,
                    "total": len(image_paths),
                    "image_path": str(img_path),
                    "status": "ok",
                    "score": float(pred["score"]),
                    "predicted_label": int(pred["predicted_label"]),
                    "predicted_label_name": pred["predicted_label_name"],
                    "threshold": float(pred["threshold"]),
                    "error": None,
                }
            )
        except Exception as e:
            failures += 1
            err_msg = str(e)
            print(f"  [WARNING] Failed to classify {img_path}: {err_msg}")
            per_image_events.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "category": category_name,
                    "index": idx,
                    "total": len(image_paths),
                    "image_path": str(img_path),
                    "status": "error",
                    "score": None,
                    "predicted_label": None,
                    "predicted_label_name": None,
                    "threshold": float(threshold),
                    "error": err_msg,
                }
            )

    # Save per-image processing logs
    per_image_log_csv = output_dir / f"{category_name}_per_image_log.csv"
    pd.DataFrame(per_image_events).to_csv(per_image_log_csv, index=False)

    per_image_log_jsonl = output_dir / f"{category_name}_per_image_log.jsonl"
    with per_image_log_jsonl.open("w", encoding="utf-8") as f:
        for event in per_image_events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"[{category_name}] Saved per-image log to {per_image_log_csv}")
    
    if not rows:
        raise RuntimeError(f"No images could be classified from {images_dir}")
    
    pred_df = pd.DataFrame(rows)
    
    # Save predictions CSV
    pred_csv = output_dir / f"{category_name}_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"[{category_name}] Saved predictions to {pred_csv}")
    
    # Generate plots
    _plot_score_curve(pred_df, output_dir, category_name)
    _plot_score_histogram(pred_df, output_dir, category_name)
    _plot_testing_graphics(pred_df, output_dir, category_name)
    
    # Compute statistics (assuming all real, count false positives)
    ai_count = int((pred_df["predicted_label"] == 1).sum())
    real_count = int((pred_df["predicted_label"] == 0).sum())
    total = len(pred_df)
    fpr = ai_count / total if total > 0 else 0  # False positive rate (classifying real as AI)
    
    summary = {
        "category": category_name,
        "total_images": int(total),
        "classified_success": int(len(pred_df)),
        "classified_failed": int(failures),
        "threshold": float(threshold),
        "predicted_ai_count": int(ai_count),
        "predicted_real_count": int(real_count),
        "false_positive_count": int(ai_count),  # Assuming GT=all real
        "false_positive_rate": float(fpr),
        "accuracy_on_real": float(real_count / total) if total > 0 else 0,
        "score_stats": {
            "mean": float(pred_df["score"].mean()),
            "median": float(pred_df["score"].median()),
            "std": float(pred_df["score"].std()),
            "min": float(pred_df["score"].min()),
            "max": float(pred_df["score"].max()),
            "q25": float(pred_df["score"].quantile(0.25)),
            "q75": float(pred_df["score"].quantile(0.75)),
        },
        "artifacts": {
            "predictions_csv": str(pred_csv),
            "per_image_log_csv": str(per_image_log_csv),
            "per_image_log_jsonl": str(per_image_log_jsonl),
            "score_plot": str(output_dir / f"{category_name}_score.png"),
            "score_histogram": str(output_dir / f"{category_name}_score_histogram.png"),
            "testing_graphics": str(output_dir / f"{category_name}_testing_graphics.png"),
        },
    }
    
    summary_json = output_dir / f"{category_name}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return summary


def _plot_score_curve(pred_df: pd.DataFrame, output_dir: Path, category: str) -> None:
    """Plot score curve ordered by descending score."""
    ordered = pred_df.sort_values("score", ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(12, 5))
    x = np.arange(1, len(ordered) + 1)
    plt.plot(x, ordered["score"].to_numpy(), color="#0b7285", linewidth=2.0)
    plt.axhline(float(ordered["threshold"].iloc[0]), color=COLOR_THRESHOLD, linestyle="--", linewidth=1.8, label="Threshold")
    plt.title(f"SPAI Score Distribution - {category.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.xlabel("Image Rank (high to low score)", fontsize=11)
    plt.ylabel("SPAI Score", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{category}_score.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()


def _plot_score_histogram(pred_df: pd.DataFrame, output_dir: Path, category: str) -> None:
    """Plot histogram of score distribution."""
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        pred_df["score"].to_numpy(),
        bins=35,
        color="#1971c2",
        edgecolor="white",
        alpha=0.85
    )
    threshold = float(pred_df["threshold"].iloc[0])
    plt.axvline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=2.0, label=f"Threshold = {threshold}")
    
    # Color bars by threshold
    for i, patch in enumerate(patches):
        if bins[i] < threshold < bins[i + 1] or bins[i] >= threshold:
            patch.set_facecolor(COLOR_AI)
    
    plt.title(f"Score Histogram - {category.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.xlabel("SPAI Score", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / f"{category}_score_histogram.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()


def _plot_testing_graphics(pred_df: pd.DataFrame, output_dir: Path, category: str) -> None:
    """Plot multi-panel testing graphics: class distribution, CDF, box plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    ai_count = int((pred_df["predicted_label"] == 1).sum())
    real_count = int((pred_df["predicted_label"] == 0).sum())
    
    # Panel 1: Class distribution
    axes[0].bar(["Real", "AI Predicted"], [real_count, ai_count], color=[COLOR_REAL, COLOR_AI], width=0.5, edgecolor="black", linewidth=1.5)
    axes[0].set_title("Predicted Classes", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate([real_count, ai_count]):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")
    
    # Panel 2: CDF
    cdf_scores = np.sort(pred_df["score"].to_numpy())
    cdf = np.arange(1, len(cdf_scores) + 1) / len(cdf_scores)
    axes[1].plot(cdf_scores, cdf, color="#5f3dc4", linewidth=2.5, label="CDF")
    threshold = float(pred_df["threshold"].iloc[0])
    axes[1].axvline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=2.0, label=f"Threshold = {threshold}")
    axes[1].set_title("Cumulative Distribution Function", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("SPAI Score", fontsize=11)
    axes[1].set_ylabel("Cumulative Probability", fontsize=11)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    
    # Panel 3: Box plot and violin
    score_data = pred_df["score"].to_numpy()
    parts = axes[2].violinplot([score_data], positions=[0], showmeans=True, showmedians=True)
    axes[2].set_title("Score Distribution (Violin)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("SPAI Score", fontsize=11)
    axes[2].set_xticks([0])
    axes[2].set_xticklabels([category.replace('_', ' ').title()])
    axes[2].axhline(threshold, color=COLOR_THRESHOLD, linestyle="--", linewidth=2.0)
    axes[2].grid(alpha=0.3, axis="y")
    
    plt.suptitle(f"Classification Results - {category.replace('_', ' ').title()}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"{category}_testing_graphics.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()


def aggregate_results(summaries: dict[str, dict]) -> dict[str, Any]:
    """Aggregate statistics across all categories."""
    
    total_images = sum(s["total_images"] for s in summaries.values())
    total_ai = sum(s["predicted_ai_count"] for s in summaries.values())
    total_real = sum(s["predicted_real_count"] for s in summaries.values())
    total_fp = sum(s["false_positive_count"] for s in summaries.values())
    global_fpr = total_ai / total_images if total_images > 0 else 0
    
    all_scores = []
    for cat_name in CATEGORIES:
        if cat_name in summaries:
            pred_csv = Path(summaries[cat_name]["artifacts"]["predictions_csv"])
            if pred_csv.exists():
                pred_df = pd.read_csv(pred_csv)
                all_scores.extend(pred_df["score"].tolist())
    
    aggregated = {
        "evaluation_type": "crawled_images_global",
        "total_images_all_categories": int(total_images),
        "threshold": float(list(summaries.values())[0]["threshold"]),
        "predicted_ai_count_global": int(total_ai),
        "predicted_real_count_global": int(total_real),
        "false_positive_count_global": int(total_fp),
        "false_positive_rate_global": float(global_fpr),
        "accuracy_on_all_real": float(total_real / total_images) if total_images > 0 else 0,
        "score_stats_global": {
            "mean": float(np.mean(all_scores)) if all_scores else 0.0,
            "median": float(np.median(all_scores)) if all_scores else 0.0,
            "std": float(np.std(all_scores)) if all_scores else 0.0,
            "min": float(np.min(all_scores)) if all_scores else 0.0,
            "max": float(np.max(all_scores)) if all_scores else 0.0,
            "q25": float(np.quantile(all_scores, 0.25)) if all_scores else 0.0,
            "q75": float(np.quantile(all_scores, 0.75)) if all_scores else 0.0,
        },
        "by_category": {}
    }
    
    for cat_name, summary in summaries.items():
        aggregated["by_category"][cat_name] = {
            "total_images": summary["total_images"],
            "predicted_ai_count": summary["predicted_ai_count"],
            "predicted_real_count": summary["predicted_real_count"],
            "false_positive_count": summary["false_positive_count"],
            "false_positive_rate": summary["false_positive_rate"],
            "accuracy_on_real": summary["accuracy_on_real"],
            "score_mean": summary["score_stats"]["mean"],
            "score_median": summary["score_stats"]["median"],
            "score_std": summary["score_stats"]["std"],
        }
    
    return aggregated


def create_comparison_plots(summaries: dict[str, dict], output_dir: Path) -> None:
    """Create comparison plots across all categories."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = list(summaries.keys())
    ai_counts = [summaries[c]["predicted_ai_count"] for c in categories]
    real_counts = [summaries[c]["predicted_real_count"] for c in categories]
    fprs = [summaries[c]["false_positive_rate"] * 100 for c in categories]  # Convert to percentage
    accs = [summaries[c]["accuracy_on_real"] * 100 for c in categories]  # Convert to percentage
    score_means = [summaries[c]["score_stats"]["mean"] for c in categories]
    
    # Plot 1: AI vs Real by Category
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, real_counts, width, label="Real Classified", color=COLOR_REAL, edgecolor="black", linewidth=1.2)
    bars2 = ax.bar(x + width/2, ai_counts, width, label="AI Classified", color=COLOR_AI, edgecolor="black", linewidth=1.2)
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Image Count", fontsize=12, fontweight="bold")
    ax.set_title("Predicted Classes by Category", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "01_predictions_by_category.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    
    # Plot 2: False Positive Rate by Category
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, fprs, color="#ee5a6f", edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("False Positive Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("False Positive Rate by Category (Assuming GT=All Real)", fontsize=14, fontweight="bold")
    ax.set_xticklabels([c.replace('_', '\n') for c in categories])
    ax.axhline(np.mean(fprs), color="black", linestyle="--", linewidth=2, label=f"Mean FPR: {np.mean(fprs):.1f}%")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, fpr in zip(bars, fprs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{fpr:.1f}%', ha='center', va='bottom', fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "02_false_positive_rate_by_category.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    
    # Plot 3: Accuracy by Category
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, accs, color="#4dabf7", edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy on Real (%)", fontsize=12, fontweight="bold")
    ax.set_title("Classification Accuracy by Category (Assuming GT=All Real)", fontsize=14, fontweight="bold")
    ax.set_xticklabels([c.replace('_', '\n') for c in categories])
    ax.axhline(np.mean(accs), color="black", linestyle="--", linewidth=2, label=f"Mean Accuracy: {np.mean(accs):.1f}%")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "03_accuracy_by_category.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    
    # Plot 4: Mean Score by Category
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, score_means, color="#7950f2", edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean SPAI Score", fontsize=12, fontweight="bold")
    ax.set_title("Mean SPAI Score by Category", fontsize=14, fontweight="bold")
    ax.set_xticklabels([c.replace('_', '\n') for c in categories])
    ax.axhline(np.mean(score_means), color="black", linestyle="--", linewidth=2, label=f"Overall Mean: {np.mean(score_means):.3f}")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, score_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "04_mean_score_by_category.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    
    print(f"[PLOTS] Generated 4 comparison plots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete classification pipeline for crawled images by category."
    )
    parser.add_argument(
        "--crawl-dir",
        type=Path,
        required=True,
        help="Path to crawl directory with 'images' subdirectory containing category folders"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to SPAI model directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Decision threshold for SPAI classifier"
    )
    
    args = parser.parse_args()
    
    images_root = args.crawl_dir / "images"
    if not images_root.exists():
        print(f"ERROR: Images directory not found: {images_root}")
        sys.exit(1)
    
    print(f"Starting complete classification pipeline...")
    print(f"  Crawl dir: {args.crawl_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Threshold: {args.threshold}")
    
    summaries = {}
    
    # Classify each category
    for category in CATEGORIES:
        category_dir = images_root / category
        if not category_dir.exists():
            print(f"[{category}] Directory not found, skipping...")
            continue
        
        cat_output_dir = args.output_dir / category
        try:
            summary = classify_images_dir(
                images_dir=category_dir,
                output_dir=cat_output_dir,
                model_dir=args.model_dir,
                threshold=args.threshold,
                category_name=category
            )
            summaries[category] = summary
            print(f"[{category}] Completed: {summary['classified_success']} images classified")
        except Exception as e:
            print(f"[{category}] ERROR: {e}")
    
    # Aggregate results
    print("\nAggregating results across all categories...")
    aggregated = aggregate_results(summaries)
    
    agg_json = args.output_dir / "aggregated_results.json"
    agg_json.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    print(f"Saved aggregated results to {agg_json}")
    
    # Create summary CSV
    summary_rows = []
    for cat_name in CATEGORIES:
        if cat_name in summaries:
            s = summaries[cat_name]
            summary_rows.append({
                "category": cat_name,
                "total_images": s["total_images"],
                "predicted_ai": s["predicted_ai_count"],
                "predicted_real": s["predicted_real_count"],
                "false_positives": s["false_positive_count"],
                "fpr_percent": f"{s['false_positive_rate']*100:.2f}",
                "accuracy_percent": f"{s['accuracy_on_real']*100:.2f}",
                "score_mean": f"{s['score_stats']['mean']:.4f}",
                "score_median": f"{s['score_stats']['median']:.4f}",
                "score_std": f"{s['score_stats']['std']:.4f}",
            })
    
    # Add global row
    summary_rows.append({
        "category": "GLOBAL",
        "total_images": aggregated["total_images_all_categories"],
        "predicted_ai": aggregated["predicted_ai_count_global"],
        "predicted_real": aggregated["predicted_real_count_global"],
        "false_positives": aggregated["false_positive_count_global"],
        "fpr_percent": f"{aggregated['false_positive_rate_global']*100:.2f}",
        "accuracy_percent": f"{aggregated['accuracy_on_all_real']*100:.2f}",
        "score_mean": f"{aggregated['score_stats_global']['mean']:.4f}",
        "score_median": f"{aggregated['score_stats_global']['median']:.4f}",
        "score_std": f"{aggregated['score_stats_global']['std']:.4f}",
    })
    
    summary_csv = args.output_dir / "summary_by_category.csv"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(summaries, args.output_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nRESULTS SUMMARY:")
    print(f"  Total images classified: {aggregated['total_images_all_categories']}")
    print(f"  Predicted as AI: {aggregated['predicted_ai_count_global']}")
    print(f"  Predicted as Real: {aggregated['predicted_real_count_global']}")
    print(f"  Global FPR: {aggregated['false_positive_rate_global']*100:.2f}%")
    print(f"  Global Accuracy (on real): {aggregated['accuracy_on_all_real']*100:.2f}%")
    print(f"  Mean score (global): {aggregated['score_stats_global']['mean']:.4f}")
    print(f"\nOUTPUT FILES:")
    print(f"  - Per-category predictions CSV: <category>_predictions.csv")
    print(f"  - Per-category summaries: <category>_summary.json")
    print(f"  - Per-category plots: <category>_*.png (3 files each)")
    print(f"  - Aggregated results: aggregated_results.json")
    print(f"  - Summary table: summary_by_category.csv")
    print(f"  - Comparison plots: 01_*.png through 04_*.png")


if __name__ == "__main__":
    main()
