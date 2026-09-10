#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=ctro-train-dmcontrol
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --array=0-11%8
# 12 tasks (4 tasks x 3 seeds). EXP_NAME via --export.
#SBATCH --requeue

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH

module load cuda/12.6.1
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda_device:', torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export MUJOCO_GL=egl

TASKS=(cheetah-run walker-walk hopper-hop cartpole-swingup)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -ge 12 ]; then echo "i=$i out of range"; exit 1; fi
TASK=${TASKS[$((i / 3))]}
SEED=${SEEDS[$((i % 3))]}
EXP_NAME="${EXP_NAME:-exp_ctro_mlp}"
EXTRA_ARGS="${EXTRA_ARGS:---agent ctro}"

CKPT="results/dmcontrol_state/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

# shellcheck disable=SC2086
echo "Training task=${TASK} seed=${SEED} exp=${EXP_NAME} extras=${EXTRA_ARGS} array=${i}"
python -m src.experiments.run_performance_train \
  --suite dmcontrol_state \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --device cuda \
  ${EXTRA_ARGS}
