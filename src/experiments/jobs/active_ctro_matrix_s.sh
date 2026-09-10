#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=actro-mx
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-20
#SBATCH --requeue
#
# Stage 5 MiniGrid matrix. Array layout: row r=1..7, seed in {42,43,44}.
#   idx = (row-1)*3 + seed_index
# Submit ONLY after G0–G5 pass. Stop advancing when a row fails.
#
# Submit: sbatch src/experiments/jobs/active_ctro_matrix_s.sh
# Or a single row: MATRIX_ROW=4 sbatch --array=9-11 src/experiments/jobs/active_ctro_matrix_s.sh

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/active_ctro/matrix results/slurm

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export PYTHONPATH="$REPO"

SEEDS=(42 43 44)
IDX="${SLURM_ARRAY_TASK_ID}"
ROW=$((IDX / 3 + 1))
SEED="${SEEDS[$((IDX % 3))]}"
ROOT=results/active_ctro/matrix
NAMES=(
  matrix_r1_aa_ppo
  matrix_r2_aa_ppo_target_pl
  matrix_r3_replay_q
  matrix_r4_ac_mico
  matrix_r5_lsep
  matrix_r6_query_u
  matrix_r7_relational_observe
)
EXP_NAME="${NAMES[$((ROW - 1))]}"
CKPT="${ROOT}/${EXP_NAME}/seed_${SEED}/weights_final.pt"

if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "matrix row=${ROW} exp=${EXP_NAME} seed=${SEED}"
python -m src.experiments.exp_matrix_row \
  --row "${ROW}" \
  --seed "${SEED}" \
  --device cuda \
  --results-root "${ROOT}"
