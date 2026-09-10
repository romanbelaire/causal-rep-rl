#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=pix-requeue
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x_%a.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x_%a.%j.err
#SBATCH --array=0-35%6
#SBATCH --requeue
#
# Requeue unfinished dmcontrol_pixels exp_{baseline,latent_nolink,ctro} cells.
# One process per GPU (MAX_PARALLEL=1 ethos). Skips cells with weights_final.pt.
# Methods: 0=baseline PPO  1=latent_nolink  2=ctro

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
TASKS=(cheetah-run walker-walk hopper-hop cartpole-swingup)
SEEDS=(42 43 44)
# packing: method * 12 + task * 3 + seed
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -ge 36 ]; then echo "out of range"; exit 1; fi
method_i=$((i / 12))
rest=$((i % 12))
task_i=$((rest / 3))
seed_i=$((rest % 3))
TASK=${TASKS[$task_i]}
SEED=${SEEDS[$seed_i]}

case $method_i in
  0)
    EXP_NAME=exp_baseline
    EXTRA=(--agent ppo)
    ;;
  1)
    EXP_NAME=exp_latent_nolink
    EXTRA=(--agent ctro --alpha 0 --beta 0)
    ;;
  2)
    EXP_NAME=exp_ctro
    EXTRA=(--agent ctro)
    ;;
  *)
    echo "bad method_i=$method_i"; exit 1
    ;;
esac

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "Requeue pixels exp=${EXP_NAME} task=${TASK} seed=${SEED} array=${i}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --device cuda \
  --results-root results \
  "${EXTRA[@]}"
