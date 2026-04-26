#!/usr/bin/env bash
#SBATCH --job-name=classify_merged_240
#SBATCH --partition=tfg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/merged_genaibench_siuuu/slurm_%j.out
#SBATCH --error=/tmp/merged_genaibench_siuuu/slurm_%j.err

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate spai 2>/dev/null || true

IMAGES_DIR="/tmp/merged_genaibench_siuuu/images"
OUTPUT_DIR="/tmp/merged_genaibench_siuuu/classifier_eval"
MODEL_DIR="/fhome/aaasidar/spai-hf"
THRESHOLD="0.35"

mkdir -p "$OUTPUT_DIR"

# Extra execution logs (besides SLURM logs)
LOG_DIR="/tmp/merged_genaibench_siuuu/run_logs"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/classify_merged_${RUN_TS}_job_${SLURM_JOB_ID:-local}.log"

START_EPOCH="$(date +%s)"
exec > >(tee -a "$RUN_LOG") 2>&1

finish() {
  EXIT_CODE=$?
  END_EPOCH="$(date +%s)"
  ELAPSED=$((END_EPOCH - START_EPOCH))
  if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[OK] Execution finished | exit_code=${EXIT_CODE} | elapsed_s=${ELAPSED}"
  else
    echo "[ERROR] Execution failed | exit_code=${EXIT_CODE} | elapsed_s=${ELAPSED}"
  fi
  echo "[INFO] Detailed run log: ${RUN_LOG}"
}
trap finish EXIT

echo "[INFO] Start: $(date -Iseconds)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-local}"
echo "[INFO] Images dir: ${IMAGES_DIR}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Threshold: ${THRESHOLD}"
echo "[INFO] Python: $(command -v python)"

python /fhome/aaasidar/spai-hf/tools/classify_news_from_crawl.py \
  --images-dir "$IMAGES_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-dir "$MODEL_DIR" \
  --threshold "$THRESHOLD" \
  --max-images 0

echo "[OK] Done"
echo "[OK] Predictions: $OUTPUT_DIR/news_predictions.csv"
echo "[OK] Summary: $OUTPUT_DIR/news_analysis_summary.json"
echo "[INFO] End: $(date -Iseconds)"
