#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=p1b-ale-matrix
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-23
#SBATCH --requeue
#
# Stage B matrix (per qualifying game): methods × epochs × seeds.
# Default TASK=ALE/Phoenix-v5 — re-submit with TASK=... for second game.
# Methods: PPO, PFO, CTRO, LTRO-fixed × {4,16} × {42,43,44} = 24 cells / game
#
# Submit after Stage A gate:
#   TASK=ALE/Phoenix-v5 sbatch src/experiments/jobs/p1_ale_stage_b_matrix_s.sh

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1 2>/dev/null || true
nvidia-smi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export FREEZE_ID="${FREEZE_ID:-ltro_phase0_v1}"

SUITE=ale
TASK="${TASK:-ALE/Phoenix-v5}"
TOTAL_STEPS="${TOTAL_STEPS:-100000000}"
SEEDS=(42 43 44)
EPOCHS=(4 16)
# method index: 0=ppo 1=pfo 2=ctro 3=ltro
METHODS=(ppo pfo ctro ltro)

IDX="${SLURM_ARRAY_TASK_ID}"
M_I=$((IDX / 6))
REM=$((IDX % 6))
EP_I=$((REM / 3))
SEED_I=$((REM % 3))
METHOD="${METHODS[$M_I]}"
SEED="${SEEDS[$SEED_I]}"
NUM_EPOCHS="${EPOCHS[$EP_I]}"
EXP_NAME="exp_p1b_${METHOD}_ep${NUM_EPOCHS}"

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

EXTRA=()
AGENT=ppo
case "${METHOD}" in
  ppo)
    AGENT=ppo
    ;;
  pfo)
    AGENT=pfo
    ;;
  ctro)
    AGENT=ctro
    ;;
  ltro)
    AGENT=ctro
    EXTRA+=(--dz-enabled --no-dz-adapt --lambda-dz 1.0 --eta-dz 0.14)
    EXTRA+=(--load-freeze configs/frozen/ltro_phase0_v1.json)
    ;;
  *)
    echo "unknown method ${METHOD}"; exit 1
    ;;
esac

echo "p1b ale method=${METHOD} task=${TASK} epochs=${NUM_EPOCHS} seed=${SEED}"
RESUME_ARGS=()
LATEST="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_latest.pt"
if [ -f "${LATEST}" ]; then
  RESUME_ARGS=(--resume)
fi
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent "${AGENT}" \
  --num-epochs "${NUM_EPOCHS}" \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --no-collapse \
  --device cuda \
  --results-root results \
  "${EXTRA[@]}" \
  "${RESUME_ARGS[@]}"
