#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=dz-confirm
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-14
#SBATCH --requeue
#
# Stage B confirmation (reference-scaled D_Z^log), 3 arms × 5 seeds:
#   0-4  : off          (λ=0, log D_Z)     exp_dzb_off
#   5-9  : fixed dual   (adapt λ, fixed η) exp_dzb_fixed
#   10-14: fixed penalty (frozen λ, no adapt) exp_dzb_penalty
#
# Requires: ETA_DZ from calib; LAMBDA_FIXED from a short dual pilot median λ
# (defaults to 1.0 until pilot median is known).
#
# Submit after calib:
#   ETA_DZ=$(cat results/dmcontrol_pixels/exp_ctro_dz_calib_sref/seed_42/cartpole-swingup/eta_calib.txt)
#   ETA_DZ=$ETA_DZ sbatch src/experiments/jobs/dz_confirm_pixels_cartpole_s.sh

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
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"
SEEDS=(42 43 44 45 46)
LAMBDA_FIXED="${LAMBDA_FIXED:-1.0}"

CALIB_ETA="results/${SUITE}/exp_ctro_dz_calib_sref/seed_42/${TASK}/eta_calib.txt"
if [ -z "${ETA_DZ:-}" ]; then
  if [ -f "${CALIB_ETA}" ]; then
    ETA_DZ=$(cat "${CALIB_ETA}")
  else
    echo "ERROR: set ETA_DZ or run dz_eta_calib_pixels_cartpole_s.sh first"
    exit 1
  fi
fi

IDX="${SLURM_ARRAY_TASK_ID}"
ARM_I=$((IDX / 5))
SEED_I=$((IDX % 5))
SEED="${SEEDS[$SEED_I]}"

case "${ARM_I}" in
  0)
    ARM=off
    EXP_NAME=exp_dzb_off
    EXTRA=(--dz-enabled --lambda-dz 0 --no-dz-adapt)
    ;;
  1)
    ARM=fixed
    EXP_NAME=exp_dzb_fixed
    EXTRA=(--dz-enabled --eta-dz "${ETA_DZ}" --lambda-dz 1.0)
    ;;
  2)
    ARM=penalty
    EXP_NAME=exp_dzb_penalty
    EXTRA=(--dz-enabled --eta-dz "${ETA_DZ}" --lambda-dz "${LAMBDA_FIXED}" --no-dz-adapt)
    ;;
  *)
    echo "Unknown arm index ${ARM_I}"; exit 1
    ;;
esac

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "confirm arm=${ARM} seed=${SEED} exp=${EXP_NAME} eta=${ETA_DZ} lambda_fixed=${LAMBDA_FIXED} steps=${TOTAL_STEPS}"
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
