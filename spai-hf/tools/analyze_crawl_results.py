#!/usr/bin/env python3
"""
Post-execution analysis script: Read classification results and generate advanced plots.
Run this after classify_crawl_complete.py finishes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_style import (
    CATEGORY_COLORS,
    COLOR_AI,
    COLOR_REAL,
    STANDARD_DPI,
    apply_plot_style,
)

apply_plot_style()

CATEGORIES = ["news", "social_media", "arts_illustration", "education_institution", "corporate"]


def load_results(results_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Load aggregated results and per-category prediction CSVs."""
    
    agg_json = results_dir / "aggregated_results.json"
    if not agg_json.exists():
        raise FileNotFoundError(f"Missing aggregated_results.json in {results_dir}")
    
    aggregated = json.loads(agg_json.read_text())
    
    predictions = {}
    for category in CATEGORIES:
        pred_csv = results_dir / category / f"{category}_predictions.csv"
        if pred_csv.exists():
            predictions[category] = pd.read_csv(pred_csv)
        else:
            print(f"[WARNING] Missing predictions CSV for {category}")
    
    return aggregated, predictions


def plot_score_distribution_by_category(predictions: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create overlaid density plots of score distributions by category."""
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    colors = CATEGORY_COLORS
    
    for category in CATEGORIES:
        if category in predictions:
            scores = predictions[category]["score"].values
            ax.hist(scores, bins=25, alpha=0.4, label=category.replace('_', ' ').title(), 
                   color=colors.get(category, "#555"), edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("SPAI Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("Score Distribution Overlay - All Categories", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_score_distribution_overlay.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    print(f"Generated: 05_score_distribution_overlay.png")


def plot_category_statistics_table(aggregated: dict[str, Any], output_dir: Path) -> None:
    """Create a visual table of key statistics by category."""
    
    categories = list(aggregated["by_category"].keys())
    
    data = []
    for cat in categories:
        stats = aggregated["by_category"][cat]
        data.append([
            cat.replace('_', ' ').title(),
            f"{stats['total_images']}",
            f"{stats['predicted_ai_count']}",
            f"{stats['false_positive_count']}",
            f"{stats['false_positive_rate']*100:.1f}%",
            f"{stats['accuracy_on_real']*100:.1f}%",
            f"{stats['score_mean']:.3f}",
        ])
    
    # Add global row
    global_stats = aggregated["score_stats_global"]
    data.append([
        "GLOBAL",
        f"{aggregated['total_images_all_categories']}",
        f"{aggregated['predicted_ai_count_global']}",
        f"{aggregated['false_positive_count_global']}",
        f"{aggregated['false_positive_rate_global']*100:.1f}%",
        f"{aggregated['accuracy_on_all_real']*100:.1f}%",
        f"{global_stats['mean']:.3f}",
    ])
    
    columns = ["Category", "Total Images", "Pred AI", "FP Count", "FPR (%)", "Accuracy (%)", "Mean Score"]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(cellText=data, colLabels=columns, cellLoc="center", loc="center",
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#4472c4")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Style data rows with alternating colors
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i == len(data):  # Global row
                table[(i, j)].set_facecolor("#ffe699")
                table[(i, j)].set_text_props(weight="bold")
            elif i % 2 == 0:
                table[(i, j)].set_facecolor("#e7e6e6")
            else:
                table[(i, j)].set_facecolor("#f2f2f2")
    
    plt.title("Classification Results Summary Table", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_dir / "06_results_summary_table.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    print(f"Generated: 06_results_summary_table.png")


def plot_threshold_analysis_curve(predictions: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Plot accuracy at different thresholds for each category."""
    
    thresholds = np.arange(0.0, 1.01, 0.05)
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    colors = CATEGORY_COLORS
    
    for category in CATEGORIES:
        if category in predictions:
            pred_df = predictions[category]
            accuracies = []
            for th in thresholds:
                # Assuming all real (GT=0), accuracy = % of images below threshold
                acc = (pred_df["score"] < th).sum() / len(pred_df)
                accuracies.append(acc * 100)
            
            ax.plot(thresholds, accuracies, marker="o", linewidth=2.5, markersize=6,
                   label=category.replace('_', ' ').title(), color=colors.get(category, "#555"))
    
    ax.axvline(0.35, color="red", linestyle="--", linewidth=2.5, label="Threshold = 0.35 (used)")
    ax.set_xlabel("Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy on Real Images (%)", fontsize=12, fontweight="bold")
    ax.set_title("Threshold Analysis - Accuracy vs Decision Threshold", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "07_threshold_analysis_curve.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    print(f"Generated: 07_threshold_analysis_curve.png")


def plot_roc_style_analysis(predictions: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create a confusion-matrix-style heatmap showing classifications by category."""
    
    # Create a grid showing AI predicted vs category
    categories = sorted([c for c in CATEGORIES if c in predictions])
    
    # Count AI predictions by category
    ai_by_category = []
    total_by_category = []
    
    for cat in categories:
        pred_df = predictions[cat]
        ai_count = (pred_df["predicted_label"] == 1).sum()
        total = len(pred_df)
        ai_by_category.append(ai_count)
        total_by_category.append(total)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Plot 1: Stacked bar chart
    real_counts = [total - ai for total, ai in zip(total_by_category, ai_by_category)]
    
    x_pos = np.arange(len(categories))
    ax1.bar(x_pos, real_counts, label="Classified as Real", color=COLOR_REAL, edgecolor="black", linewidth=1.2)
    ax1.bar(x_pos, ai_by_category, bottom=real_counts, label="Classified as AI", color=COLOR_AI, edgecolor="black", linewidth=1.2)
    
    ax1.set_xlabel("Category", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax1.set_title("Classification Distribution (Stacked)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([c.replace('_', '\n') for c in categories])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    
    # Plot 2: Percentage heatmap style
    percentages = [[real*100/total, ai*100/total] for real, ai, total in zip(real_counts, ai_by_category, total_by_category)]
    
    im = ax2.imshow(percentages, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Real", "AI"])
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels([c.replace('_', ' ').title() for c in categories])
    ax2.set_title("Classification Distribution (%)", fontsize=12, fontweight="bold")
    
    # Add percentages to heatmap
    for i in range(len(categories)):
        for j in range(2):
            text = ax2.text(j, i, f"{percentages[i][j]:.1f}%", ha="center", va="center",
                          color="black", fontweight="bold", fontsize=10)
    
    plt.colorbar(im, ax=ax2, label="Percentage (%)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "08_classification_distribution_heatmap.png", dpi=STANDARD_DPI, bbox_inches="tight")
    plt.close()
    print(f"Generated: 08_classification_distribution_heatmap.png")


def generate_markdown_report(aggregated: dict[str, Any], output_dir: Path) -> None:
    """Generate a markdown report with key findings."""
    
    global_stats = aggregated["score_stats_global"]
    by_cat = aggregated["by_category"]
    
    report = f"""# Classification Results Report

## Overview
- **Total Images Classified:** {aggregated['total_images_all_categories']}
- **Threshold Used:** {aggregated['threshold']}
- **Evaluation Setting:** Assuming all images are real (GT=Real)

## Global Results
- **Predicted as Real:** {aggregated['predicted_real_count_global']} ({aggregated['accuracy_on_all_real']*100:.2f}%)
- **Predicted as AI (False Positives):** {aggregated['predicted_ai_count_global']} ({aggregated['false_positive_rate_global']*100:.2f}%)
- **Global Accuracy (on real):** {aggregated['accuracy_on_all_real']*100:.2f}%
- **Global False Positive Rate:** {aggregated['false_positive_rate_global']*100:.2f}%

### Global Score Statistics
- **Mean:** {global_stats['mean']:.4f}
- **Median:** {global_stats['median']:.4f}
- **Std Dev:** {global_stats['std']:.4f}
- **Min:** {global_stats['min']:.4f}
- **Max:** {global_stats['max']:.4f}
- **Q25:** {global_stats['q25']:.4f}
- **Q75:** {global_stats['q75']:.4f}

## Results by Category

"""
    
    for cat in CATEGORIES:
        if cat in by_cat:
            stats = by_cat[cat]
            report += f"""### {cat.replace('_', ' ').title()}
- **Total Images:** {stats['total_images']}
- **Classified as Real:** {stats['predicted_real_count']} ({stats['accuracy_on_real']*100:.2f}%)
- **Classified as AI (FP):** {stats['false_positive_count']} ({stats['false_positive_rate']*100:.2f}%)
- **Mean Score:** {stats['score_mean']:.4f}
- **Median Score:** {stats['score_median']:.4f}
- **Std Dev:** {stats['score_std']:.4f}

"""
    
    report += """## Key Findings & Interpretation

### Per-Category Performance
The classifier shows varying performance across different content categories:
- Categories with higher FPR may have images with characteristics that trigger AI detection
- Categories with lower FPR align better with real image signatures in the trained model

### Score Distribution Insights
The SPAI classifier outputs continuous scores in [0, 1]. The threshold of 0.35:
- Scores below 0.35 are classified as Real
- Scores above 0.35 are classified as AI
- Check the threshold analysis curve to see how accuracy changes with different thresholds

### Recommendations
1. Review images with highest AI scores (potential misclassifications)
2. Analyze score distributions per website within each category
3. Consider category-specific thresholds if FPR varies significantly
4. Validate on ground-truth labeled subset if available

## Output Files Generated
- `summary_by_category.csv` - Tabular summary of results
- `aggregated_results.json` - Complete statistical results in JSON
- Per-category results in `<category>/` directories:
  - `<category>_predictions.csv` - Full predictions for all images
  - `<category>_summary.json` - Per-category statistics
  - `<category>_*.png` - Visualizations (score curve, histogram, testing graphics)
- Comparison plots:
  - `01_predictions_by_category.png` - AI vs Real counts
  - `02_false_positive_rate_by_category.png` - FPR comparison
  - `03_accuracy_by_category.png` - Accuracy comparison
  - `04_mean_score_by_category.png` - Score means
  - `05_score_distribution_overlay.png` - Overlaid distributions
  - `06_results_summary_table.png` - Visual summary table
  - `07_threshold_analysis_curve.png` - Accuracy vs threshold
  - `08_classification_distribution_heatmap.png` - Classification breakdown

---
*Report generated by classify_crawl_complete.py and analyze_crawl_results.py*
"""
    
    report_file = output_dir / "RESULTS_REPORT.md"
    report_file.write_text(report, encoding="utf-8")
    print(f"Generated: RESULTS_REPORT.md")


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval")
    
    print(f"Loading results from: {results_dir}")
    
    try:
        aggregated, predictions = load_results(results_dir)
    except Exception as e:
        print(f"ERROR: Failed to load results: {e}")
        sys.exit(1)
    
    print(f"Loaded aggregated results and {len(predictions)} category prediction CSVs")
    
    print("\nGenerating advanced analysis plots...")
    plot_score_distribution_by_category(predictions, results_dir)
    plot_category_statistics_table(aggregated, results_dir)
    plot_threshold_analysis_curve(predictions, results_dir)
    plot_roc_style_analysis(predictions, results_dir)
    generate_markdown_report(aggregated, results_dir)
    
    print("\n" + "="*70)
    print("POST-EXECUTION ANALYSIS COMPLETE")
    print("="*70)
    print("\nAll visualization and analysis files have been generated.")
    print(f"See {results_dir} for complete results.")


if __name__ == "__main__":
    main()
