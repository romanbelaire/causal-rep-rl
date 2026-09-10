#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=p1a-ale-ppo
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-11
#SBATCH --requeue
#
# Stage A: PPO-only Moalla ALE collapse screen.
#   games × epochs{4,16} × seeds{42,43,44} = 12 cells
#   total_steps = 100M (Moalla released gym-atari default)
#
# Submit:
#   sbatch src/experiments/jobs/p1_ale_stage_a_ppo_s.sh

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1 2>/dev/null || true
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

SUITE=ale
TOTAL_STEPS="${TOTAL_STEPS:-100000000}"
SEEDS=(42 43 44)
EPOCHS=(4 16)
GAMES=("ALE/Phoenix-v5" "ALE/NameThisGame-v5")
FREEZE_ID="${FREEZE_ID:-ltro_phase0_v1}"
export FREEZE_ID

IDX="${SLURM_ARRAY_TASK_ID}"
# layout: game(2) × epoch(2) × seed(3)
GAME_I=$((IDX / 6))
REM=$((IDX % 6))
EP_I=$((REM / 3))
SEED_I=$((REM % 3))
TASK="${GAMES[$GAME_I]}"
SEED="${SEEDS[$SEED_I]}"
NUM_EPOCHS="${EPOCHS[$EP_I]}"
EXP_NAME="exp_p1a_ppo_ep${NUM_EPOCHS}"

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "p1a ale freeze=${FREEZE_ID} game=${TASK} epochs=${NUM_EPOCHS} seed=${SEED} exp=${EXP_NAME} steps=${TOTAL_STEPS}"
RESUME_ARGS=()
LATEST="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_latest.pt"
if [ -f "${LATEST}" ]; then
  RESUME_ARGS=(--resume)
fi
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ppo \
  --num-epochs "${NUM_EPOCHS}" \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --no-collapse \
  --device cuda \
  --results-root results \
  "${RESUME_ARGS[@]}"
