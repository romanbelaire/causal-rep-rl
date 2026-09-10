#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=p1-ppo-ep
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-19
#SBATCH --requeue
#
# Phase 1 scaffold: PPO-only epoch stress dial on cartpole-swingup pixels.
#   epochs ∈ {4, 8, 16, 32} × seeds {42..46}
#   EXP_NAME=exp_p1v2_ppo_ep${N}  (v2: eval forced on eval_frequency schedule)
#
# Keep lr/clip/rollout/architecture at suite defaults; vary only --num-epochs.
# After healthy ep4 finishes: fit thresholds, then inspect collapse vs stress.
#
# Later matrix (not this job): PPO / PFO / CTRO / LTRO-fixed with frozen η.
#
# Submit:
#   sbatch src/experiments/jobs/p1_ppo_epoch_sweep_cartpole_s.sh

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
EPOCHS=(4 8 16 32)
FREEZE_ID="${FREEZE_ID:-ltro_phase0_v1}"
export FREEZE_ID

IDX="${SLURM_ARRAY_TASK_ID}"
EP_I=$((IDX / 5))
SEED_I=$((IDX % 5))
SEED="${SEEDS[$SEED_I]}"
NUM_EPOCHS="${EPOCHS[$EP_I]}"
EXP_NAME="exp_p1v2_ppo_ep${NUM_EPOCHS}"

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "p1 ppo epoch_sweep freeze=${FREEZE_ID} epochs=${NUM_EPOCHS} seed=${SEED} exp=${EXP_NAME} steps=${TOTAL_STEPS}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ppo \
  --num-epochs "${NUM_EPOCHS}" \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --device cuda \
  --results-root results
