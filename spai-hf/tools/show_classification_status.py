#!/usr/bin/env python3
"""
Quick status report generator - shows what files exist and their contents at a glance
"""

from pathlib import Path
import json
import pandas as pd
import sys

def format_size(bytes_val):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}GB"

def print_results_status(results_dir):
    """Print a quick status of results directory"""
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS STATUS")
    print("="*70)
    
    print(f"\n📁 Location: {results_dir}")
    
    # Check for main files
    summary_csv = results_dir / "summary_by_category.csv"
    agg_json = results_dir / "aggregated_results.json"
    report_md = results_dir / "RESULTS_REPORT.md"
    
    main_files = {
        "summary_by_category.csv": summary_csv,
        "aggregated_results.json": agg_json,
        "RESULTS_REPORT.md": report_md,
    }
    
    print("\n📋 Main Output Files:")
    for name, path in main_files.items():
        if path.exists():
            size = format_size(path.stat().st_size)
            print(f"  ✓ {name:30} ({size})")
        else:
            print(f"  ✗ {name:30} (missing)")
    
    # Check for plots
    plot_files = sorted(results_dir.glob("*.png"))
    if plot_files:
        print(f"\n📊 Comparison Plots ({len(plot_files)} files):")
        for plot in plot_files:
            size = format_size(plot.stat().st_size)
            print(f"  ✓ {plot.name:50} ({size})")
    else:
        print("\n📊 Comparison Plots: (none found)")
    
    # Check for category directories
    categories = ["news", "social_media", "arts_illustration", "education_institution", "corporate"]
    
    print(f"\n📂 Category Results:")
    for category in categories:
        cat_dir = results_dir / category
        if cat_dir.exists():
            pred_csv = cat_dir / f"{category}_predictions.csv"
            summary_json = cat_dir / f"{category}_summary.json"
            pngs = list(cat_dir.glob(f"{category}*.png"))
            
            status = "✓"
            details = []
            
            if pred_csv.exists():
                details.append(f"predictions")
            if summary_json.exists():
                details.append(f"summary")
            details.append(f"{len(pngs)} plots")
            
            detail_str = ", ".join(details)
            print(f"  {status} {category:30} ({detail_str})")
        else:
            print(f"  ✗ {category:30} (not found)")
    
    # Try to load and display summary CSV
    if summary_csv.exists():
        print("\n" + "="*70)
        print("RESULTS SUMMARY TABLE")
        print("="*70)
        
        try:
            df = pd.read_csv(summary_csv)
            print(df.to_string(index=False))
            
            # Calculate totals if available
            if "total_images" in df.columns:
                total = df.iloc[:-1]["total_images"].sum() if len(df) > 1 else 0
                print(f"\nTotal images classified (excl. global): {total}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    # Try to load aggregated results
    if agg_json.exists():
        print("\n" + "="*70)
        print("KEY STATISTICS")
        print("="*70)
        
        try:
            with open(agg_json) as f:
                agg = json.load(f)
            
            if "total_images_all_categories" in agg:
                print(f"Total Images:                  {agg['total_images_all_categories']}")
                print(f"Threshold:                     {agg['threshold']}")
                print(f"Predicted as Real:             {agg['predicted_real_count_global']}")
                print(f"Predicted as AI (FP):          {agg['predicted_ai_count_global']}")
                print(f"Global Accuracy (on real):     {agg['accuracy_on_all_real']*100:.2f}%")
                print(f"Global False Positive Rate:    {agg['false_positive_rate_global']*100:.2f}%")
                print(f"\nScore Statistics (Global):")
                stats = agg.get("score_stats_global", {})
                print(f"  Mean:                        {stats.get('mean', 'N/A'):.4f}")
                print(f"  Median:                      {stats.get('median', 'N/A'):.4f}")
                print(f"  Std Dev:                     {stats.get('std', 'N/A'):.4f}")
                print(f"  Min / Max:                   {stats.get('min', 'N/A'):.4f} / {stats.get('max', 'N/A'):.4f}")
        except Exception as e:
            print(f"Error reading JSON: {e}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Review the summary table above
2. Open plots in a viewer (start with 01_predictions_by_category.png)
3. Read RESULTS_REPORT.md for detailed analysis
4. Check category-specific results in each category/ folder
5. Use results to write TFG section on evaluation
    """)
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval"
    
    print_results_status(results_dir)
