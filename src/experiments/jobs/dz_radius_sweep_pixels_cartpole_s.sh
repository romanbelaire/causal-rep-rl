#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=dz-radius
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-9
#SBATCH --requeue
#
# Phase 0 P0.2 dual-radius sweep (adaptive λ, fixed η):
#   0-4  : dual-tight  η=0.05  → exp_dzb_dual_tight
#   5-9  : dual-medium η=0.07  → exp_dzb_dual_medium
#
# Reuses Stage B confirm recipe; does NOT write into exp_dzb_fixed (loose η=0.14).
#
# Submit:
#   sbatch src/experiments/jobs/dz_radius_sweep_pixels_cartpole_s.sh

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

export FREEZE_ID="${FREEZE_ID:-ltro_phase0_v1}"
SUITE=dmcontrol_pixels
TASK=cartpole-swingup
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"
SEEDS=(42 43 44 45 46)
ETAS=(0.05 0.07)
EXP_NAMES=(exp_dzb_dual_tight exp_dzb_dual_medium)
ARM_LABELS=(dual_tight dual_medium)

IDX="${SLURM_ARRAY_TASK_ID}"
ARM_I=$((IDX / 5))
SEED_I=$((IDX % 5))
SEED="${SEEDS[$SEED_I]}"
ETA_DZ="${ETAS[$ARM_I]}"
EXP_NAME="${EXP_NAMES[$ARM_I]}"
ARM="${ARM_LABELS[$ARM_I]}"

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "radius arm=${ARM} seed=${SEED} exp=${EXP_NAME} eta=${ETA_DZ} steps=${TOTAL_STEPS}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --dz-enabled \
  --eta-dz "${ETA_DZ}" \
  --lambda-dz 1.0 \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --device cuda \
  --results-root results
