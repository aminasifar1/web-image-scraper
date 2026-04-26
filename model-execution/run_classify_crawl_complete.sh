#!/bin/bash
# Complete classification pipeline for all crawled images
# Submits category-by-category classification and then global aggregation

CRAWL_DIR="/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/live"
OUTPUT_DIR="/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval"
MODEL_DIR="/fhome/aaasidar/spai-hf"
THRESHOLD=0.35

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "CLASSIFICATION PIPELINE - CATEGORY BY CATEGORY"
echo "=========================================="
echo "Crawl dir: ${CRAWL_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model dir: ${MODEL_DIR}"
echo "Threshold: ${THRESHOLD}"
echo ""

# Submit sbatch for each category
CATEGORIES=("news" "social_media" "arts_illustration" "education_institution" "corporate")

for CATEGORY in "${CATEGORIES[@]}"; do
    JOB_ID=$(sbatch \
        --job-name="classify_${CATEGORY}" \
        --partition=tfg \
        --cpus-per-task=4 \
        --mem=16G \
        --gres=gpu:1 \
        --time=02:00:00 \
        --output="${OUTPUT_DIR}/${CATEGORY}_%j.log" \
        --error="${OUTPUT_DIR}/${CATEGORY}_%j.err" \
        --export=ALL,CRAWL_DIR="${CRAWL_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",MODEL_DIR="${MODEL_DIR}",THRESHOLD="${THRESHOLD}" \
        --wrap="source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh; conda activate spai; python /fhome/aaasidar/spai-hf/tools/classify_crawl_complete.py --crawl-dir \"${CRAWL_DIR}\" --output-dir \"${OUTPUT_DIR}\" --model-dir \"${MODEL_DIR}\" --threshold \"${THRESHOLD}\"" \
        | awk '{print $NF}')
    
    echo "Submitted classification for category: ${CATEGORY} (Job ID: ${JOB_ID})"
done

echo ""
echo "All category jobs submitted!"
echo "Monitor progress with: squeue -u aaasidar"
echo ""
echo "Once all jobs complete, CSV summaries and plots will be in:"
echo "  ${OUTPUT_DIR}/summary_by_category.csv"
echo "  ${OUTPUT_DIR}/aggregated_results.json"
echo "  ${OUTPUT_DIR}/0X_*.png (comparison plots)"
