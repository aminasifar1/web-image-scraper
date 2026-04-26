#!/bin/bash
# Master orchestration script for complete classification pipeline
# This script:
# 1. Submits the main classification job
# 2. Waits for completion
# 3. Runs post-execution analysis and generates advanced plots

set -e

CRAWL_DIR="/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/live"
OUTPUT_DIR="/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval"
MODEL_DIR="/fhome/aaasidar/spai-hf"
THRESHOLD=0.35

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "SPAI CLASSIFIER - COMPLETE PIPELINE"
echo "==========================================${NC}"
echo ""
echo "Configuration:"
echo "  Crawl directory: ${CRAWL_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Model directory: ${MODEL_DIR}"
echo "  Threshold: ${THRESHOLD}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Step 1: Submit classification job
echo -e "${YELLOW}STEP 1: Submitting classification job...${NC}"

JOB_ID=$(sbatch \
    --job-name="spai_classify" \
    --partition=tfg \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --time=04:00:00 \
    --output="${OUTPUT_DIR}/classify_%j.log" \
    --error="${OUTPUT_DIR}/classify_%j.err" \
    --export=ALL,CRAWL_DIR="${CRAWL_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",MODEL_DIR="${MODEL_DIR}",THRESHOLD="${THRESHOLD}" \
    --wrap="source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh; conda activate spai; python /fhome/aaasidar/spai-hf/tools/classify_crawl_complete.py --crawl-dir \"${CRAWL_DIR}\" --output-dir \"${OUTPUT_DIR}\" --model-dir \"${MODEL_DIR}\" --threshold \"${THRESHOLD}\"" \
    2>&1 | grep -oP '(?<=job )\d+')

echo -e "${GREEN}✓ Job submitted with ID: ${JOB_ID}${NC}"
echo ""

# Step 2: Wait for job completion
echo -e "${YELLOW}STEP 2: Waiting for classification job to complete...${NC}"
echo "  (You can monitor progress with: squeue -u aaasidar -j ${JOB_ID})"
echo ""

# Check job status every 30 seconds
while true; do
    JOB_STATE=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || echo "COMPLETED")
    
    if [ "${JOB_STATE}" = "COMPLETED" ] || [ "${JOB_STATE}" = "FAILED" ] || [ "${JOB_STATE}" = "CANCELLED" ]; then
        echo -e "${GREEN}✓ Classification job completed (state: ${JOB_STATE})${NC}"
        break
    elif [ "${JOB_STATE}" = "RUNNING" ]; then
        echo "  Job is running..."
        sleep 30
    else
        echo "  Job state: ${JOB_STATE}"
        sleep 30
    fi
done

echo ""

# Step 3: Run post-execution analysis
echo -e "${YELLOW}STEP 3: Running post-execution analysis and generating advanced plots...${NC}"

python /fhome/aaasidar/spai-hf/tools/analyze_crawl_results.py "${OUTPUT_DIR}"

echo ""
echo -e "${GREEN}=========================================="
echo "✓ PIPELINE COMPLETED SUCCESSFULLY"
echo "==========================================${NC}"
echo ""
echo "Results location: ${OUTPUT_DIR}"
echo ""
echo "Key output files:"
echo "  - summary_by_category.csv (tabular results)"
echo "  - aggregated_results.json (complete statistics)"
echo "  - RESULTS_REPORT.md (analysis summary)"
echo "  - 01_*.png through 08_*.png (visualizations)"
echo "  - <category>/ directories with per-category results"
echo ""
echo "Next steps:"
echo "  1. Review CSV and JSON files for metrics"
echo "  2. Check PNG plots for visualization"
echo "  3. Read RESULTS_REPORT.md for detailed findings"
echo "  4. Examine specific category results in <category>/ folders"
echo ""
