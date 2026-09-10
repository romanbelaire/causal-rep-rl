#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=actro-mg
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-8
#SBATCH --requeue
#
# MiniGrid Unlock E0 + E6: 3 arms × 3 seeds.
#   arm 0 = vanilla PPO          exp_e0_vanilla
#   arm 1 = anti-aliased PPO     exp_e0_aa_ppo
#   arm 2 = active CTRO          exp_active_ctro
# Seeds: 42, 43, 44. Full BASE_TRAINING_CONFIG budget (1500 epochs).
#
# Submit (after toys): sbatch src/experiments/jobs/active_ctro_minigrid_s.sh
# Or via: bash src/experiments/jobs/submit_active_ctro.sh

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/active_ctro/minigrid results/slurm

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export PYTHONPATH="$REPO"

ARMS=(exp_e0_vanilla exp_e0_aa_ppo exp_active_ctro)
MODULES=(src.experiments.exp_e0_vanilla src.experiments.exp_e0_aa_ppo src.experiments.exp_active_ctro)
SEEDS=(42 43 44)

IDX="${SLURM_ARRAY_TASK_ID}"
ARM=$((IDX / 3))
SEED="${SEEDS[$((IDX % 3))]}"
EXP_NAME="${ARMS[$ARM]}"
MODULE="${MODULES[$ARM]}"
ROOT=results/active_ctro/minigrid
CKPT="${ROOT}/${EXP_NAME}/seed_${SEED}/weights_final.pt"

if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "active-CTRO MiniGrid arm=${ARM} exp=${EXP_NAME} seed=${SEED}"
python -m "${MODULE}" \
  --seed "${SEED}" \
  --device cuda \
  --results-root "${ROOT}"
