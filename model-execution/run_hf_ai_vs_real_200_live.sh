#!/usr/bin/env bash
#SBATCH --job-name=hf_ai_vs_real_200
#SBATCH --partition=tfg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fhome/aaasidar/spai-hf/hf_runs/ai_vs_real_200_live/slurm_%j.out
#SBATCH --error=/fhome/aaasidar/spai-hf/hf_runs/ai_vs_real_200_live/slurm_%j.err

set -euo pipefail

# Uso opcional:
#   OUT_BASE=/fhome/aaasidar/spai-hf/hf_runs/ai_vs_real_200_live/run_$(date +%Y%m%d_%H%M%S) ./run_hf_ai_vs_real_200_live.sh
# Si no defines OUT_BASE, se crea automáticamente con timestamp.

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate spai 2>/dev/null || true

OUT_BASE="${OUT_BASE:-/fhome/aaasidar/spai-hf/hf_runs/ai_vs_real_200_live/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_BASE/exported_images"
mkdir -p /fhome/aaasidar/spai-hf/hf_runs/ai_vs_real_200_live

echo "[INFO] Output base: $OUT_BASE"
echo "[INFO] Log en vivo + guardado en: $OUT_BASE/run.log"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"

time python -u /fhome/aaasidar/spai-hf/infer_hf_dataset.py \
	--dataset Parveshiiii/AI-vs-Real \
	--split train \
	--image-column image \
	--max-images 200 \
	--overwrite-outputs \
	--randomize \
	--seed 42 \
	--threshold 0.35 \
	--label-to-model-map 0:1,1:0 \
	--output-csv "$OUT_BASE/results.csv" \
	--output-jsonl "$OUT_BASE/results.jsonl" \
	--export-image-dir "$OUT_BASE/exported_images" \
	--export-image-size 224 \
	2>&1 | tee "$OUT_BASE/run.log"

echo ""
echo "[OK] Ejecución finalizada"
echo "[OK] CSV:          $OUT_BASE/results.csv"
echo "[OK] JSONL:        $OUT_BASE/results.jsonl"
echo "[OK] Log completo: $OUT_BASE/run.log"
