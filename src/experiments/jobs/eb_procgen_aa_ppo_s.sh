#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=eb-procgen
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-79
#SBATCH --requeue
#
# E-B Procgen subset (4 games × 10 seeds × 2 arms).
# Games: coinrun, starpilot, caveflyer, fruitbot (four-game subset).
# GATE: after E-A.5. Matched --algo-preset anti_aliased_ppo vs PPO.
#
# Submit: sbatch src/experiments/jobs/eb_procgen_aa_ppo_s.sh

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1 2>/dev/null || true
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export PYTHONPATH="$REPO"

SUITE=procgen_easy
TOTAL_STEPS="${TOTAL_STEPS:-25000000}"
TASKS=(coinrun starpilot caveflyer fruitbot)
SEEDS=(42 43 44 45 46 47 48 49 50 51)

IDX="${SLURM_ARRAY_TASK_ID}"
# 2 arms × 4 tasks × 10 seeds = 80
ARM=$((IDX / 40))
REST=$((IDX % 40))
TASK_I=$((REST / 10))
SEED="${SEEDS[$((REST % 10))]}"
TASK="${TASKS[$TASK_I]}"

case "${ARM}" in
  0)
    EXP_NAME=exp_eb_ppo
    AGENT=ppo
    EXTRA=()
    ;;
  1)
    EXP_NAME=exp_eb_aa_ppo
    AGENT=ctro
    EXTRA=(--algo-preset anti_aliased_ppo)
    ;;
  *)
    echo "Unknown arm ${ARM}"
    exit 1
    ;;
esac

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "E-B Procgen arm=${ARM} exp=${EXP_NAME} task=${TASK} seed=${SEED}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent "${AGENT}" \
  --total-steps "${TOTAL_STEPS}" \
  --device cuda \
  --results-root results \
  "${EXTRA[@]}"
