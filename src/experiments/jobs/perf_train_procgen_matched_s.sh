#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=procgen-matched-train
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --array=0-23%8
# 24 tasks (8 games x 3 seeds). EXP_NAME / EXTRA_ARGS via --export.
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

TASKS=(coinrun starpilot caveflyer fruitbot chaser leaper maze miner)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -ge 24 ]; then echo "i=$i out of range"; exit 1; fi
TASK=${TASKS[$((i / 3))]}
SEED=${SEEDS[$((i % 3))]}

EXP_NAME="${EXP_NAME:-exp_ctro_cnn}"
# Space-separated extra args for run_performance_train, e.g. "--alpha 0 --beta 0"
EXTRA_ARGS="${EXTRA_ARGS:---agent ctro}"

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

# shellcheck disable=SC2086
echo "Training task=${TASK} seed=${SEED} exp=${EXP_NAME} extras=${EXTRA_ARGS} array=${i}"
python -m src.experiments.run_performance_train \
  --suite procgen_easy \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --device cuda \
  ${EXTRA_ARGS}
