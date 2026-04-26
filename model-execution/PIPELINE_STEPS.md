# STEP-BY-STEP: Complete Classification Pipeline

## 🎯 Goal
Classify all 502 crawled images using the SPAI classifier, generate detailed statistics for accuracy/false positives, and create publication-ready visualizations.

---

## 📋 Prerequisites Check

Before starting, verify you have:

```bash
# Check conda environment
conda env list | grep spai

# Check if images exist
ls -la /fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/live/images/

# Verify SPAI model
ls -la /fhome/aaasidar/spai-hf/spai/weights/spai.pth

# Check scripts exist
ls -la /fhome/aaasidar/spai-hf/tools/classify_crawl_complete.py
ls -la /fhome/aaasidar/spai-hf/tools/analyze_crawl_results.py
```

---

## 🚀 Quick Start (Recommended)

### Step 1: Run the Full Automated Pipeline

Make the script executable:
```bash
chmod +x /fhome/aaasidar/spai-hf/run_full_classification_pipeline.sh
```

Run it:
```bash
/fhome/aaasidar/spai-hf/run_full_classification_pipeline.sh
```

**What happens:**
1. Submits classification job to GPU cluster
2. Shows job ID and waits for completion
3. Automatically runs post-execution analysis
4. Generates all plots and reports
5. Reports completion with next steps

**Expected runtime:** 30-45 minutes

**Monitor progress:**
```bash
# In another terminal
squeue -u aaasidar
# or check logs
tail -f /fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval/classify_*.log
```

---

## 📊 Step 2: Review Results

Once the pipeline completes:

### Quick Status Report
```bash
python /fhome/aaasidar/spai-hf/tools/show_classification_status.py
```

This shows:
- ✓/✗ status of all output files
- Summary table of results
- Key statistics (accuracy, FPR, etc.)
- File locations and sizes

### Read the Summary

```bash
# View the CSV summary
cat /fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval/summary_by_category.csv

# Or open in your editor
code /fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval/summary_by_category.csv
```

Expected format:
```
category,total_images,predicted_ai,predicted_real,false_positives,fpr_percent,accuracy_percent,score_mean,score_median,score_std
news,192,34,158,34,17.69,82.31,0.3012,0.2845,0.1234
...
GLOBAL,502,XX,XX,XX,XX.XX,XX.XX,0.XXXX,0.XXXX,0.XXXX
```

### View Visualizations

Open the plots (images) in any viewer:

```bash
# View all plots
ls -la /fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval/*.png

# Recommended viewing order:
# 1. 01_predictions_by_category.png - See AI vs Real distribution
# 2. 02_false_positive_rate_by_category.png - See FPR comparison
# 3. 03_accuracy_by_category.png - See accuracy comparison
# 4. 06_results_summary_table.png - Visual table
# 5. 07_threshold_analysis_curve.png - Understand threshold sensitivity
```

---

## 📄 Step 3: Understand the Statistics

### Key Metrics Explained

From the CSV columns:

| Column | Meaning | Example | Interpretation |
|--------|---------|---------|-----------------|
| `category` | Website category | "news" | Which category of images |
| `total_images` | Total images in category | 192 | Sample size |
| `predicted_ai` | Count classified as AI | 34 | False positives (assuming GT=real) |
| `predicted_real` | Count classified as real | 158 | Correct classifications |
| `false_positives` | Same as predicted_ai | 34 | FP count |
| `fpr_percent` | False Positive Rate | 17.69 | (FP / total) × 100 |
| `accuracy_percent` | Accuracy on real images | 82.31 | (Correct / total) × 100 |
| `score_mean` | Average SPAI score | 0.3012 | Lower = more real-like |
| `score_median` | Median SPAI score | 0.2845 | Central tendency |
| `score_std` | Score variability | 0.1234 | Confidence spread |

### Interpreting Results

**Good Signs:**
- Accuracy > 80% (few false positives)
- FPR < 20% (low misclassification rate)
- Score mean < 0.4 (classifier thinks images are mostly real)
- Consistent across categories (no huge variations)

**Areas to Investigate:**
- Category with very low accuracy (< 70%)
- Category with high FPR (> 30%)
- Very high score_mean (> 0.6) in a category
- Outlier images with extreme scores

---

## 🔍 Step 4: Per-Category Deep Dive

For each category, examine detailed results:

```bash
# Example: News category
cd /fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval/news

# View all predictions for news images
head -20 news_predictions.csv

# View per-category statistics
cat news_summary.json | python -m json.tool

# View category-specific plots
ls -la news*.png
```

### Finding Problematic Images

To find which images are classified as AI (false positives):

```python
import pandas as pd

# Load news predictions
pred = pd.read_csv("news/news_predictions.csv")

# Find all AI-classified images (score > 0.35)
ai_images = pred[pred['predicted_label'] == 1].sort_values('score', ascending=False)

print("Top 10 images most likely to be AI:")
print(ai_images[['image_path', 'score']].head(10))

# Save to file
ai_images.to_csv("false_positives_news.csv", index=False)
```

---

## 📖 Step 5: Generate TFG Section

Create a results section for your TFG document:

### Template

```markdown
## 6. Evaluation Results

### 6.1 Classifier Performance on Crawled Images

We evaluated the SPAI classifier on 502 real images collected from our 5-website category 
crawler. The classifier was set to use a decision threshold of 0.35, determined through 
prior threshold analysis on synthetic and real benchmark datasets.

#### Classification Results by Category

[INSERT TABLE from summary_by_category.csv]

#### Key Findings

- **Overall Accuracy:** XX% (XX out of 502 images correctly classified as real)
- **Global False Positive Rate:** XX% (XX images misclassified as AI)
- **Mean SPAI Score:** 0.XXXX ± 0.XXXX (indicating predominantly real content)

[INSERT: 01_predictions_by_category.png caption and interpretation]

[INSERT: 02_false_positive_rate_by_category.png caption and interpretation]

[INSERT: 07_threshold_analysis_curve.png caption and interpretation]

### 6.2 Per-Category Analysis

[For each category with notable findings...]

**News Category (192 images):** 
- Accuracy: XX%, FPR: XX%
- [observations about news images]

[... repeat for other categories ...]

### 6.3 Analysis and Discussion

The classifier shows [good/moderate/poor] performance on crawled images with XX% accuracy.
The XX% FPR suggests [implications for use case]. Category-specific analysis reveals 
[notable differences between categories].
```

---

## 🛠️ Step 6: Advanced Analysis (Optional)

### Generate Custom Reports

```bash
# Read the full JSON results
python -c "import json; print(json.dumps(json.load(open('/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval/aggregated_results.json')), indent=2))"
```

### Extract Specific Metrics

```python
import json
import pandas as pd

# Load results
with open("aggregated_results.json") as f:
    results = json.load(f)

# Print summary for TFG
print(f"Total images: {results['total_images_all_categories']}")
print(f"Accuracy: {results['accuracy_on_all_real']*100:.1f}%")
print(f"FPR: {results['false_positive_rate_global']*100:.1f}%")

# Load CSV for specific category
news = pd.read_csv("news/news_predictions.csv")
print(f"News: {len(news)} total, mean score = {news['score'].mean():.4f}")
```

### Create Comparison with Other Models

If you have results from other classifiers, compare:

```python
# Compare accuracy across classifiers
import pandas as pd

results_df = pd.DataFrame({
    'Classifier': ['SPAI', 'OtherModel1', 'OtherModel2'],
    'Accuracy': [82.31, 75.50, 88.20],
    'FPR': [17.69, 24.50, 11.80]
})

print(results_df)
```

---

## ⚠️ Troubleshooting

### Job Doesn't Start
```bash
# Check partition availability
sinfo -p tfg

# Check your account
sacctmgr show user format=User,Account

# Try submitting without GPU
sbatch --gres=gpu:0 ...
```

### Out of Memory Error
```bash
# Increase memory in the sbatch commands
# Edit run_full_classification_pipeline.sh or run_classify_complete.sh
# Change: --mem=32G  →  --mem=64G
```

### No Output Files Generated
```bash
# Check if Python script ran correctly
tail -100 /path/to/classify_*.err

# Try running script directly (not via sbatch)
python /fhome/aaasidar/spai-hf/tools/classify_crawl_complete.py \
  --crawl-dir /path/to/crawl \
  --output-dir /tmp/test \
  --model-dir /fhome/aaasidar/spai-hf
```

### Plots Look Wrong
```bash
# Regenerate plots from existing predictions
python /fhome/aaasidar/spai-hf/tools/analyze_crawl_results.py /path/to/output/dir
```

---

## ✅ Completion Checklist

After running the pipeline:

- [ ] Classification job submitted successfully
- [ ] Job completed without errors
- [ ] `summary_by_category.csv` exists and is readable
- [ ] `aggregated_results.json` exists with complete statistics
- [ ] All 8 comparison plots (01-08) are generated
- [ ] All 5 category directories have predictions CSV and plots
- [ ] Run `show_classification_status.py` to verify completeness
- [ ] Read `RESULTS_REPORT.md` for interpretation
- [ ] Prepare TFG section with results and visualizations
- [ ] Document any notable findings or anomalies

---

## 📞 Support

If something doesn't work:

1. **Check the logs:** `tail -200 classify_*.err`
2. **Verify inputs:** `ls -la /path/to/images/*`
3. **Run quick test:** `bash test_classification_quick.sh`
4. **Read README:** `cat CLASSIFICATION_PIPELINE_README.md`
5. **Ask for help** with the error message and output directory

---

## 📚 Related Files

- Main classifier: `/spai/weights/spai.pth`
- Classification script: `/tools/classify_crawl_complete.py`
- Analysis script: `/tools/analyze_crawl_results.py`
- Status script: `/tools/show_classification_status.py`
- Documentation: `CLASSIFICATION_PIPELINE_README.md`

---

**You're all set! Start with Step 1 above.** 🚀
