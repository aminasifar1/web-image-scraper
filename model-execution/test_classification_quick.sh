#!/bin/bash
# Quick test: classify a small subset to verify setup works

CRAWL_DIR="/fhome/aaasidar/spai-hf/crawl_runs/20260416_5x5_200117/live"
OUTPUT_DIR="/tmp/spai_test"
MODEL_DIR="/fhome/aaasidar/spai-hf"

echo "Running quick test classification on subset..."
echo "Output: ${OUTPUT_DIR}"
echo ""

python /fhome/aaasidar/spai-hf/tools/classify_crawl_complete.py \
  --crawl-dir "${CRAWL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --threshold 0.35

echo ""
echo "Test completed! Check files in ${OUTPUT_DIR}"
echo ""
echo "To see results:"
echo "  cat ${OUTPUT_DIR}/summary_by_category.csv"
echo "  ls -la ${OUTPUT_DIR}/*_summary.json"
echo "  find ${OUTPUT_DIR} -name '*.png'"
