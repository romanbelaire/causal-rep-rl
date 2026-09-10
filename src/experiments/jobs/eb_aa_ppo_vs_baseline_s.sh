#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=eb-aa-ppo
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-59
#SBATCH --requeue
#
# E-B / E-C: anti-aliased PPO vs unmodified PPO (state DMC, 3 tasks × 10 seeds × 2 arms).
# GATE: submit only after E-A.5 supports H1 (see docs/next_round_experiments.md).
# B4 walker/cheetah baselines must pass before quoting those tasks.
#
# Array layout:
#   arm 0 = PPO baseline          exp_eb_ppo
#   arm 1 = anti_aliased_ppo      exp_eb_aa_ppo
#   tasks: cartpole-swingup, walker-walk, cheetah-run
#   seeds: 42..51
#
# Submit: sbatch src/experiments/jobs/eb_aa_ppo_vs_baseline_s.sh

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
TOTAL_STEPS="${TOTAL_STEPS:-8000000}"
TASKS=(cartpole-swingup walker-walk cheetah-run)
SEEDS=(42 43 44 45 46 47 48 49 50 51)

IDX="${SLURM_ARRAY_TASK_ID}"
# 2 arms × 3 tasks × 10 seeds = 60
ARM=$((IDX / 30))
REST=$((IDX % 30))
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

echo "E-B arm=${ARM} exp=${EXP_NAME} task=${TASK} seed=${SEED} steps=${TOTAL_STEPS}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent "${AGENT}" \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --device cuda \
  --results-root results \
  "${EXTRA[@]}"
