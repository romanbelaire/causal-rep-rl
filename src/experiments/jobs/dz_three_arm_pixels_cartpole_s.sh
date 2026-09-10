#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=dz3-arm
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-2
#SBATCH --requeue
#
# Three-arm short-horizon D_Z comparison (same stack/seed):
#   0: D_Z off (λ=0, log D_Z, no penalty)     → exp_dz3_off
#   1: fixed calibrated η                      → exp_dz3_fixed
#   2: adaptive η_t = max(η_min, c * EMA)      → exp_dz3_adapt
#
# Budget: 1.5M steps (covers formation + early refinement). No early-stop.
# PR is logged every metrics dump; pairwise distance hists at checkpoints;
# adaptive arm logs eta_dz / dz_ema every update.
#
# Submit: sbatch src/experiments/jobs/dz_three_arm_pixels_cartpole_s.sh

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1 2>/dev/null || true
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export MUJOCO_GL=egl

SUITE=dmcontrol_pixels
TASK=cartpole-swingup
SEED=42
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"

ETA_MIN_FILE="results/${SUITE}/exp_ctro_dz_calib/seed_${SEED}/${TASK}/eta_calib.txt"
if [ -z "${ETA_DZ:-}" ]; then
  if [ -f "${ETA_MIN_FILE}" ]; then
    ETA_DZ=$(cat "${ETA_MIN_FILE}")
  else
    echo "ERROR: set ETA_DZ or run calib first (${ETA_MIN_FILE})"
    exit 1
  fi
fi
ETA_DZ_C="${ETA_DZ_C:-1.5}"
DZ_EMA_TAU="${DZ_EMA_TAU:-0.05}"
DZ_EMA_INIT="${DZ_EMA_INIT:-0.5}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXP_NAME=exp_dz3_off
    EXTRA=(--dz-enabled --lambda-dz 0 --no-dz-adapt)
    ;;
  1)
    EXP_NAME=exp_dz3_fixed
    EXTRA=(--dz-enabled --eta-dz "${ETA_DZ}" --lambda-dz 1.0)
    ;;
  2)
    EXP_NAME=exp_dz3_adapt
    EXTRA=(
      --dz-enabled --lambda-dz 1.0 --dz-eta-adapt
      --eta-dz "${ETA_DZ}" --eta-dz-min "${ETA_DZ}"
      --eta-dz-c "${ETA_DZ_C}" --dz-ema-tau "${DZ_EMA_TAU}"
      --dz-ema-init "${DZ_EMA_INIT}"
    )
    ;;
  *)
    echo "Unknown array id ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "Three-arm id=${SLURM_ARRAY_TASK_ID} exp=${EXP_NAME} eta=${ETA_DZ} steps=${TOTAL_STEPS}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --device cuda \
  --results-root results \
  "${EXTRA[@]}"
