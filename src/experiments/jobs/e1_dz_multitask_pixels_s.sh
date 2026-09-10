#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=e1-dz-multitask
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-39
#SBATCH --requeue
#
# E1 — multi-task Run 2 replication (D_Z off vs fixed).
# Layout: 4 tasks × 2 arms × 5 seeds = 40 cells.
#   task_idx = IDX % 4
#   arm_idx  = (IDX / 4) % 2     0=off 1=fixed
#   seed_idx = IDX / 8
# Fresh exp names with T1 diagnostics (tilde mu_PL, pairwise series, f hist).
# Protocol: docs/run2_analysis_protocol.json
#
# Submit: sbatch src/experiments/jobs/e1_dz_multitask_pixels_s.sh

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
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"
TASKS=(cartpole-swingup cheetah-run walker-walk hopper-hop)
SEEDS=(42 43 44 45 46)

ETA_MIN_FILE="results/${SUITE}/exp_ctro_dz_calib/seed_42/cartpole-swingup/eta_calib.txt"
if [ -z "${ETA_DZ:-}" ]; then
  if [ -f "${ETA_MIN_FILE}" ]; then
    ETA_DZ=$(cat "${ETA_MIN_FILE}")
  else
    echo "ERROR: set ETA_DZ or run calib first (${ETA_MIN_FILE})"
    exit 1
  fi
fi

IDX="${SLURM_ARRAY_TASK_ID}"
TASK_IDX=$((IDX % 4))
ARM_IDX=$(( (IDX / 4) % 2 ))
SEED_IDX=$((IDX / 8))
TASK="${TASKS[$TASK_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

if [ "${ARM_IDX}" -eq 0 ]; then
  ARM=off
  EXP_NAME=exp_e1_off
  EXTRA=(--dz-enabled --lambda-dz 0 --no-dz-adapt)
else
  ARM=fixed
  EXP_NAME=exp_e1_fixed
  EXTRA=(--dz-enabled --eta-dz "${ETA_DZ}" --lambda-dz 1.0)
fi

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "E1 arm=${ARM} task=${TASK} seed=${SEED} exp=${EXP_NAME} eta=${ETA_DZ} steps=${TOTAL_STEPS}"
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
