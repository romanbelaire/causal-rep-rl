#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=nolink-train-dmcontrol-finish
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --array=0-9%8
# Retrain exp_latent_nolink pairs with Linear (pre-fix) action_log_std so
# performance eval can load them (state-independent Parameter log_std).
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

# Pairs with Linear action_log_std (cannot eval) as of 2026-08-05.
# Kept good: hopper-hop@42, cartpole-swingup@43 (Parameter log_std).
PAIRS=(
  "cartpole-swingup 42"
  "cheetah-run 42"
  "walker-walk 42"
  "cheetah-run 43"
  "hopper-hop 43"
  "walker-walk 43"
  "cartpole-swingup 44"
  "cheetah-run 44"
  "hopper-hop 44"
  "walker-walk 44"
)

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -ge "${#PAIRS[@]}" ]; then
  echo "i=$i out of range (n=${#PAIRS[@]})"
  exit 1
fi
read -r TASK SEED <<< "${PAIRS[$i]}"
EXP_NAME="${EXP_NAME:-exp_latent_nolink}"
RUN_DIR="results/dmcontrol_state/${EXP_NAME}/seed_${SEED}/${TASK}"
CKPT="${RUN_DIR}/weights_final.pt"

# Only treat Parameter log_std finals as done; wipe Linear pre-fix ckpts.
if [ -f "${CKPT}" ]; then
  python - <<PY
import torch, sys
from pathlib import Path
ck = torch.load("${CKPT}", map_location="cpu", weights_only=False)
pol = ck["policy"]
if "action_log_std.weight" in pol or "action_log_std.bias" in pol:
    print("Removing Linear log_std checkpoint: ${CKPT}")
    Path("${CKPT}").unlink(missing_ok=True)
    Path("${RUN_DIR}/weights_latest.pt").unlink(missing_ok=True)
    sys.exit(0)
if "action_log_std" in pol:
    print("Valid Parameter log_std — skip: ${CKPT}")
    sys.exit(2)
print("Unknown log_std layout — retrain: ${CKPT}")
Path("${CKPT}").unlink(missing_ok=True)
sys.exit(0)
PY
  rc=$?
  if [ "$rc" -eq 2 ]; then
    exit 0
  fi
fi

echo "Training task=${TASK} seed=${SEED} exp=${EXP_NAME} array=${i}"
python -m src.experiments.run_performance_train \
  --suite dmcontrol_state \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --alpha 0 \
  --beta 0 \
  --device cuda
