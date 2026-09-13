#!/bin/bash
# A0/A1/A2 screen: starpilot + maze, seeds 42-46 (30 cells).
# Submit AFTER predictiveness re-analysis and AFTER α/λ are frozen.
# ALPHA_SEP and LAMBDA_SEP must be exported.
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=procgen-sep-screen
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --array=0-29%8
#SBATCH --requeue

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

ARMS=(exp_sep_a0 exp_sep_a1 exp_sep_a2)
GAMES=(starpilot maze)
SEEDS=(42 43 44 45 46)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$((i / 10))]}
rest=$((i % 10))
GAME=${GAMES[$((rest / 5))]}
SEED=${SEEDS[$((rest % 5))]}

ALPHA_SEP="${ALPHA_SEP:?set ALPHA_SEP from fit_sep_alpha}"
LAMBDA_SEP="${LAMBDA_SEP:?set LAMBDA_SEP from the one-game sweep}"

BASE="--agent ppo --policy-on-latent --num-envs 64"
case "$ARM" in
  exp_sep_a0) EXTRA_ARGS="$BASE --sep-coef 0" ;;
  exp_sep_a1) EXTRA_ARGS="$BASE --sep-coef ${LAMBDA_SEP} --alpha-sep ${ALPHA_SEP}" ;;
  exp_sep_a2) EXTRA_ARGS="$BASE --sep-coef ${LAMBDA_SEP} --alpha-sep ${ALPHA_SEP} --sep-shuffle-returns" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

CKPT="results/procgen_easy/${ARM}/seed_${SEED}/${GAME}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "arm=${ARM} game=${GAME} seed=${SEED} extras=${EXTRA_ARGS}"
# shellcheck disable=SC2086
python -m src.experiments.run_performance_train \
  --suite procgen_easy \
  --task "${GAME}" \
  --seed "${SEED}" \
  --exp-name "${ARM}" \
  --device cuda \
  ${EXTRA_ARGS}
