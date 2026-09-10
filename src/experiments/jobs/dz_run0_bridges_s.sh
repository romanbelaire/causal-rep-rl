#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=dz-run0
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x_%a.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x_%a.%j.err
#SBATCH --array=0-3
#SBATCH --requeue
#
# Run 0: eta sweep on cartpole-swingup, seed 42 (constraint sanity first wave).
# Expand seeds after lambda/tracking looks healthy.

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

ETAS=(0.01 0.03 0.1 0.3)
ETA=${ETAS[$SLURM_ARRAY_TASK_ID]}
SEED=42
TASK=cartpole-swingup
EXP_NAME="exp_dz_run0_eta${ETA}"

CKPT="results/dmcontrol_state/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "Run0 eta=${ETA} task=${TASK} seed=${SEED}"
python -m src.experiments.run_performance_train \
  --suite dmcontrol_state \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --dz-enabled \
  --head-phasing \
  --eta-dz "${ETA}" \
  --lambda-dz 1.0 \
  --alpha 0.01 \
  --beta 0.1 \
  --no-collapse \
  --device cuda \
  --num-envs 4 \
  --results-root results
