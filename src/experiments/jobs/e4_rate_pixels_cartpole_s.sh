#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=e4-rate
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-14
#SBATCH --requeue
#
# E4 — rate test: eta sweep vs tilde mu_PL degradation.
#   etas = 0.03 0.06 0.101 0.2 0.4
#   seeds 42 43 44
#   task cartpole-swingup (pixel)
# Layout: IDX = eta_idx * 3 + seed_idx
# After all finish (CPU):
#   python -m src.experiments.fit_dz_rate \
#     --results-root results/dmcontrol_pixels \
#     --etas 0.03 0.06 0.101 0.2 0.4 \
#     --task cartpole-swingup
#
# Submit: sbatch src/experiments/jobs/e4_rate_pixels_cartpole_s.sh

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

SUITE=dmcontrol_pixels
TASK=cartpole-swingup
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"
ETAS=(0.03 0.06 0.101 0.2 0.4)
SEEDS=(42 43 44)

IDX="${SLURM_ARRAY_TASK_ID}"
ETA_IDX=$((IDX / 3))
SEED_IDX=$((IDX % 3))
ETA="${ETAS[$ETA_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
EXP_NAME="exp_dz_run4_eta${ETA}"

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "E4 eta=${ETA} seed=${SEED} exp=${EXP_NAME} steps=${TOTAL_STEPS}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --dz-enabled \
  --eta-dz "${ETA}" \
  --lambda-dz 1.0 \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --no-collapse \
  --device cuda \
  --results-root results
