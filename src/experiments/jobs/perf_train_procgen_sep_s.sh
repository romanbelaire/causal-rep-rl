#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=procgen-sep
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --requeue
# TASKS, SEEDS, EXP_NAME, EXTRA_ARGS via --export. Array size = #TASKS * #SEEDS.

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda_device:', torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

IFS=' ' read -r -a TASKS <<< "${TASKS:-starpilot maze}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46}"
n_seeds=${#SEEDS[@]}
n_tasks=${#TASKS[@]}
i=$SLURM_ARRAY_TASK_ID
n=$((n_tasks * n_seeds))
if [ "$i" -ge "$n" ]; then echo "i=$i out of range n=$n"; exit 1; fi
TASK=${TASKS[$((i / n_seeds))]}
SEED=${SEEDS[$((i % n_seeds))]}

EXP_NAME="${EXP_NAME:-exp_sep_a0}"
EXTRA_ARGS="${EXTRA_ARGS:---agent ppo --policy-on-latent --sep-coef 0 --num-envs 64}"

CKPT="results/procgen_easy/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "Training task=${TASK} seed=${SEED} exp=${EXP_NAME} extras=${EXTRA_ARGS} array=${i}"
# shellcheck disable=SC2086
python -m src.experiments.run_performance_train \
  --suite procgen_easy \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --device cuda \
  ${EXTRA_ARGS}
