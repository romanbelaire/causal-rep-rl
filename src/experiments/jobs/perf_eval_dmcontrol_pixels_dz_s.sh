#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 04:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=pix-eval
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x_%a.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x_%a.%j.err
#SBATCH --array=0-5%6
#SBATCH --requeue
#
# Performance eval for pixels CTRO ± D_Z (and optional latent controls).
# Array packs exp × seed: methods=(exp_ctro_dz exp_ctro) × seeds 42..44 → 0-5

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm results/perf_eval

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1 2>/dev/null || true
nvidia-smi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export MUJOCO_GL=egl

SUITE=dmcontrol_pixels
METHODS=(exp_ctro_dz exp_ctro)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
METHOD=${METHODS[$((i / 3))]}
SEED=${SEEDS[$((i % 3))]}

CHECKPOINT_ROOT="results/${SUITE}/${METHOD}"
OUTPUT_ROOT="results/perf_eval/${SUITE}/${METHOD}"
CKPT_DIR="${CHECKPOINT_ROOT}/seed_${SEED}"
OUT_DIR="${OUTPUT_ROOT}/seed_${SEED}"
mkdir -p "${OUT_DIR}"

if [ ! -d "${CKPT_DIR}" ]; then
  echo "ERROR: missing checkpoint dir ${CKPT_DIR}"
  exit 1
fi
if [ -f "${OUT_DIR}/performance_eval_metrics.csv" ]; then
  echo "Skipping — eval exists: ${OUT_DIR}/performance_eval_metrics.csv"
  exit 0
fi

echo "Eval suite=${SUITE} method=${METHOD} seed=${SEED}"
python -m src.experiments.run_performance_eval \
  --suite "${SUITE}" \
  --checkpoint "${CKPT_DIR}" \
  --config "${CKPT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --device cuda
