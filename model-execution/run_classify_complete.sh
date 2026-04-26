#!/bin/bash
# Single sbatch job for complete classification pipeline

CRAWL_DIR="/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/live"
OUTPUT_DIR="/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/classifier_eval"
MODEL_DIR="/fhome/aaasidar/spai-hf"
THRESHOLD=0.35

# Create output directory before submitting job
mkdir -p "${OUTPUT_DIR}"

echo "Submitting complete classification pipeline..."
echo "  Crawl dir: ${CRAWL_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Model dir: ${MODEL_DIR}"
echo "  Threshold: ${THRESHOLD}"
echo ""

sbatch \
    --job-name="classify_complete" \
    --partition=tfg \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --time=04:00:00 \
    --output="${OUTPUT_DIR}/complete_%j.log" \
    --error="${OUTPUT_DIR}/complete_%j.err" \
    --export=ALL,CRAWL_DIR="${CRAWL_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",MODEL_DIR="${MODEL_DIR}",THRESHOLD="${THRESHOLD}" \
    --wrap="source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh; conda activate spai; python /fhome/aaasidar/spai-hf/tools/classify_crawl_complete.py --crawl-dir \"${CRAWL_DIR}\" --output-dir \"${OUTPUT_DIR}\" --model-dir \"${MODEL_DIR}\" --threshold \"${THRESHOLD}\""

echo "Job submitted!"
echo "Monitor with: squeue -u aaasidar"
