#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=e2-run3
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-11
#SBATCH --requeue
#
# E2 / Run 3: does the bisimulation target earn its cost?
#   0-2   A  PL only + tilde mu_PL          exp_e2_pl_tilde
#   3-5   B  PL + MICo (full)               exp_e2_full
#   6-8   C  PL + spectral value head       exp_e2_pl_spectral
#   9-11  D  MICo only                      exp_e2_mico
# Seeds 42,43,44 on dmcontrol_pixels / cartpole-swingup / 1.5M steps.
# Primary metric: latent_pair_p05. See docs/next_round_experiments.md.
#
# Submit: sbatch src/experiments/jobs/e2_run3_pixels_cartpole_s.sh

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
SEEDS=(42 43 44)

IDX="${SLURM_ARRAY_TASK_ID}"
ARM=$((IDX / 3))
SEED="${SEEDS[$((IDX % 3))]}"

case "${ARM}" in
  0)
    EXP_NAME=exp_e2_pl_tilde
    EXTRA=(--alpha 0 --beta 0.532 --pl-scale-invariant --pl-value-normalize --alpha-warmup-epochs 0 --beta-warmup-epochs 500)
    ;;
  1)
    EXP_NAME=exp_e2_full
    EXTRA=(--alpha 0.00205 --beta 0.532 --pl-scale-invariant --pl-value-normalize --alpha-warmup-epochs 500 --beta-warmup-epochs 500)
    ;;
  2)
    EXP_NAME=exp_e2_pl_spectral
    EXTRA=(--alpha 0 --beta 0.532 --pl-scale-invariant --pl-value-normalize --value-spectral-norm --alpha-warmup-epochs 0 --beta-warmup-epochs 500)
    ;;
  3)
    EXP_NAME=exp_e2_mico
    EXTRA=(--alpha 0.00205 --beta 0 --alpha-warmup-epochs 500 --beta-warmup-epochs 0)
    ;;
  *)
    echo "Unknown arm ${ARM}"
    exit 1
    ;;
esac

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "E2 arm=${ARM} exp=${EXP_NAME} seed=${SEED} steps=${TOTAL_STEPS}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --device cuda \
  --results-root results \
  "${EXTRA[@]}"
