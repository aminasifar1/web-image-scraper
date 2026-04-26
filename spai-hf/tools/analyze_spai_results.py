#!/usr/bin/env python3
"""
Analyze SPAI results and generate comprehensive visualizations.
Creates histograms, AUC curve, loss, and precision plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score
import json

matplotlib.use('Agg')

def analyze_spai_results(csv_path, output_dir):
    """
    Analyze SPAI predictions and generate visualizations.
    """
    # Read results
    df = pd.read_csv(csv_path)
    
    print(f"📊 Analyzing {len(df)} predictions...")
    print(f"Columns: {list(df.columns)}")
    
    # Find SPAI score column
    spai_col = None
    for col in ['spai', 'score', 'prediction', 'confidence']:
        if col in df.columns:
            spai_col = col
            break
    
    if spai_col is None:
        print("Error: Could not find SPAI score column")
        return None
    
    scores = df[spai_col].values
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Score Distribution Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores, bins=30, color='#4c6ef5', edgecolor='black', alpha=0.7)
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax1.set_xlabel('SPAI Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution (All 100 Images)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Real vs AI Classification Bar
    ax2 = fig.add_subplot(gs[0, 1])
    threshold = 0.5
    ai_count = (scores >= threshold).sum()
    real_count = (scores < threshold).sum()
    colors_bar = ['#51cf66', '#ff6b6b']
    bars = ax2.bar(['Real', 'AI'], [real_count, ai_count], color=colors_bar, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Count')
    ax2.set_title('Classification Result\n(at threshold=0.5)')
    ax2.set_ylim([0, len(df)])
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({100*height/len(df):.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_scores = np.sort(scores)
    cumsum = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax3.plot(sorted_scores, cumsum, linewidth=2, color='#4c6ef5')
    ax3.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    ax3.fill_between(sorted_scores, 0, cumsum, alpha=0.2, color='#4c6ef5')
    ax3.set_xlabel('SPAI Score')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function')
    ax3.grid(True, alpha=0.3)
    
    # 4. Score Statistics Text Box
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    stats_text = f"""
    SPAI SCORE STATISTICS
    
    Total Images: {len(df)}
    AI (≥0.5): {ai_count} ({100*ai_count/len(df):.1f}%)
    Real (<0.5): {real_count} ({100*real_count/len(df):.1f}%)
    
    Min Score: {scores.min():.2e}
    Max Score: {scores.max():.4f}
    Mean: {scores.mean():.4f}
    Median: {np.median(scores):.2e}
    Std Dev: {scores.std():.4f}
    """
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Score Ranges Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    bins_custom = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels_custom = ['Very Low\n(0-0.1)', 'Low\n(0.1-0.3)', 'Medium\n(0.3-0.5)',
                     'High\n(0.5-0.7)', 'Very High\n(0.7-0.9)', 'Extreme\n(0.9-1.0)']
    hist_counts, _ = np.histogram(scores, bins=bins_custom)
    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bins_custom)-1))
    ax5.bar(labels_custom, hist_counts, color=colors_gradient, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Count')
    ax5.set_title('Score Range Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Precision/Recall vs Threshold
    ax6 = fig.add_subplot(gs[1, 2])
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    for t in thresholds:
        tp = (scores >= t).sum()
        fp = (scores >= t).sum()
        fn = (scores < t).sum()
        precision = tp / (tp + fp + 1e-10) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
    
    ax6.plot(thresholds, precisions, label='Precision', linewidth=2, color='#ff6b6b')
    ax6.plot(thresholds, recalls, label='Recall', linewidth=2, color='#51cf66')
    ax6.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Decision Threshold')
    ax6.set_ylabel('Score')
    ax6.set_title('Precision vs Recall vs Threshold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. ROC Curve (using binary labels based on threshold)
    ax7 = fig.add_subplot(gs[2, 0])
    y_true = (scores >= 0.5).astype(int)
    y_pred = scores
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    ax7.plot(fpr, tpr, color='#4c6ef5', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax7.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    ax7.fill_between(fpr, 0, tpr, alpha=0.2, color='#4c6ef5')
    ax7.set_xlim([0.0, 1.0])
    ax7.set_ylim([0.0, 1.05])
    ax7.set_xlabel('False Positive Rate')
    ax7.set_ylabel('True Positive Rate')
    ax7.set_title('ROC Curve')
    ax7.legend(loc="lower right")
    ax7.grid(True, alpha=0.3)
    
    # 8. Precision-Recall Curve
    ax8 = fig.add_subplot(gs[2, 1])
    precision_array, recall_array, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall_array, precision_array)
    
    ax8.plot(recall_array, precision_array, color='#ff6b6b', lw=2, 
            label=f'PR curve (AUC = {pr_auc:.3f})')
    ax8.fill_between(recall_array, 0, precision_array, alpha=0.2, color='#ff6b6b')
    ax8.set_xlim([0.0, 1.0])
    ax8.set_ylim([0.0, 1.05])
    ax8.set_xlabel('Recall')
    ax8.set_ylabel('Precision')
    ax8.set_title('Precision-Recall Curve')
    ax8.legend(loc="best")
    ax8.grid(True, alpha=0.3)
    
    # 9. Confusion Matrix Heatmap
    ax9 = fig.add_subplot(gs[2, 2])
    cm = confusion_matrix(y_true, (y_pred >= 0.5).astype(int))
    im = ax9.imshow(cm, cmap='Blues', aspect='auto')
    ax9.set_xticks([0, 1])
    ax9.set_yticks([0, 1])
    ax9.set_xticklabels(['Real', 'AI'])
    ax9.set_yticklabels(['Real', 'AI'])
    ax9.set_xlabel('Predicted')
    ax9.set_ylabel('True')
    ax9.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax9.text(j, i, cm[i, j],
                          ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontweight='bold', fontsize=14)
    
    plt.colorbar(im, ax=ax9)
    
    # Overall title
    fig.suptitle('SPAI AI Detection Analysis - 100 El País Images', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = output_dir / 'spai_analysis_comprehensive.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # Create summary dict
    summary = {
        'total_images': len(df),
        'ai_detected': int(ai_count),
        'real_images': int(real_count),
        'ai_percentage': float(100 * ai_count / len(df)),
        'statistics': {
            'min_score': float(scores.min()),
            'max_score': float(scores.max()),
            'mean_score': float(scores.mean()),
            'median_score': float(np.median(scores)),
            'std_dev': float(scores.std()),
        },
        'metrics': {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1_score': float(f1_score(y_true, (y_pred >= 0.5).astype(int))),
            'accuracy': float((y_true == (y_pred >= 0.5).astype(int)).mean())
        }
    }
    
    return summary, df


if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'output/spai_input.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output/spai_analysis'
    
    summary, df = analyze_spai_results(csv_path, output_dir)
    
    # Save summary
    with open(Path(output_dir) / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save enriched CSV
    df.to_csv(Path(output_dir) / 'predictions_with_analysis.csv', index=False)
    
    print("\n" + "="*80)
    print("📊 SPAI ANALYSIS COMPLETE")
    print("="*80)
    print(json.dumps(summary, indent=2))
    print("="*80)
