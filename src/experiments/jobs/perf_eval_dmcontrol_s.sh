#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-small
#SBATCH -t 08:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=ctro-perf-dmcontrol
#SBATCH -o logs/%x_%a.%j.out

#SBATCH --array=0-2
#SBATCH --requeue

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm

# PyTorch + deps: Ocean conda env (Python 3.10)
export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH

module load cuda/12.6.1
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda_device:', torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EXP_NAME="${EXP_NAME:-exp_full}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-results/dmcontrol_state/${EXP_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/perf_eval/dmcontrol_state/${EXP_NAME}}"
CKPT_DIR="${CHECKPOINT_ROOT}/seed_${SEED}"
OUT_DIR="${OUTPUT_ROOT}/seed_${SEED}"
mkdir -p "${OUT_DIR}"
if [ ! -d "${CKPT_DIR}" ]; then
  echo "ERROR: missing checkpoint dir ${CKPT_DIR}"
  exit 1
fi
python -m src.experiments.run_performance_eval \
  --suite dmcontrol_state \
  --checkpoint "${CKPT_DIR}" \
  --config "${CKPT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --device cuda
