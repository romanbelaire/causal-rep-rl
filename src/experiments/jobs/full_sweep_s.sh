#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-small
#SBATCH -t 08:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=ctro-full-sweep
#SBATCH -o logs/%x_%a.%j.out

#SBATCH --array=0-9%2
# GPU-small MaxSubmit=10 MaxJobs=2. Resubmit with OFFSET=10 and OFFSET=20 for remaining cells.
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
ALPHAS=(0.01 0.1 0.5)
BETAS=(0.01 0.1 0.5)
OFFSET=${OFFSET:-0}
i=$((SLURM_ARRAY_TASK_ID + OFFSET))
if [ "$i" -ge 27 ]; then echo "i=$i out of range"; exit 0; fi
SEED=${SEEDS[$((i % 3))]}
cell=$((i / 3))
ALPHA=${ALPHAS[$((cell / 3))]}
BETA=${BETAS[$((cell % 3))]}
python -m src.experiments.exp_full --seed "$SEED" --alpha "$ALPHA" --beta "$BETA" --device cuda
