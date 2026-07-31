#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=ctro-train-procgen-finish
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --array=0-15%8
# Remaining CTRO procgen_easy runs only (skip completed finals).
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

# Explicit unfinished (task, seed) pairs as of 2026-07-25:
# partial: coinrun/caveflyer/fruitbot @43
# missing: coinrun/starpilot/caveflyer/fruitbot @44; leaper/maze/miner @42,43,44
PAIRS=(
  "coinrun 43"
  "coinrun 44"
  "starpilot 44"
  "caveflyer 43"
  "caveflyer 44"
  "fruitbot 43"
  "fruitbot 44"
  "leaper 42"
  "leaper 43"
  "leaper 44"
  "maze 42"
  "maze 43"
  "maze 44"
  "miner 42"
  "miner 43"
  "miner 44"
)

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -ge "${#PAIRS[@]}" ]; then
  echo "i=$i out of range (n=${#PAIRS[@]})"
  exit 1
fi
read -r TASK SEED <<< "${PAIRS[$i]}"
EXP_NAME="${EXP_NAME:-exp_full}"
CKPT="results/procgen_easy/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi
LATEST="results/procgen_easy/${EXP_NAME}/seed_${SEED}/${TASK}/weights_latest.pt"
if [ -f "${LATEST}" ]; then
  python - <<PY
import torch, sys
ck = torch.load("${LATEST}", map_location="cpu", weights_only=False)
need = ("epoch", "total_steps", "normalization", "step")
missing = [k for k in need if k not in ck]
if missing:
    print(f"Removing incompatible checkpoint (missing {missing}): ${LATEST}")
    sys.exit(2)
print(f"Compatible checkpoint found: ${LATEST}")
PY
  if [ $? -eq 2 ]; then
    rm -f "${LATEST}"
  fi
fi
echo "Training task=${TASK} seed=${SEED} exp=${EXP_NAME} array=${i}"
python -m src.experiments.run_performance_train \
  --suite procgen_easy \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --device cuda
