#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-small
#SBATCH -t 08:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=rstr-impala-vae-strict
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

CONFIGS=("configs/exp4_rstr_impala_vae_strict.json")
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
CFG=${CONFIGS[$((i / 3))]}
SEED=${SEEDS[$((i % 3))]}
if [ -f src/main.py ]; then
  python -m src.main --config "$CFG" --seed "$SEED"
else
  echo "src.main missing; use src/experiments/jobs/*.sh for live CTRO experiments" >&2
  exit 1
fi
