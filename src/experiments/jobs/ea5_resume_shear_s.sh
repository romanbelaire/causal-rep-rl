#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 0-12:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=ea5-shear
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-24
#SBATCH --requeue
#
# E-A.5: resume training from compensated shear checkpoints.
# Array: 5 seeds × 5 c-values (1,2,5,10,30).
#
# Prerequisites (CPU, once per seed):
#   PYTHONPATH=. python -m src.experiments.shear_gauge_experiment \
#     --run-dir results/dmcontrol_state/exp_anti_aliased_ppo/seed_${SEED}/cartpole-swingup \
#     --c-values 1,2,5,10,30 --write-sheared-ckpts --device cpu \
#     --output-dir results/dmcontrol_state/shear_ckpts/seed_${SEED}
#
# Do NOT submit until E-A.1 passes on those checkpoints.
# Submit: sbatch src/experiments/jobs/ea5_resume_shear_s.sh

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
export PYTHONPATH="$REPO"

SUITE=dmcontrol_state
TASK=cartpole-swingup
RESUME_STEPS="${RESUME_STEPS:-500000}"
SEEDS=(42 43 44 45 46)
CS=(1 2 5 10 30)

IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$((IDX / 5))]}"
C="${CS[$((IDX % 5))]}"
EXP_NAME="exp_ea5_resume_c${C}"
INIT="results/${SUITE}/shear_ckpts/seed_${SEED}/c_${C}/weights_final.pt"

if [ ! -f "${INIT}" ]; then
  echo "ERROR: missing sheared checkpoint ${INIT}"
  echo "Run shear_gauge_experiment --write-sheared-ckpts first."
  exit 1
fi

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "E-A.5 seed=${SEED} c=${C} steps=${RESUME_STEPS} init=${INIT}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --algo-preset anti_aliased_ppo \
  --init-weights "${INIT}" \
  --total-steps "${RESUME_STEPS}" \
  --no-early-stop \
  --device cuda \
  --results-root results
