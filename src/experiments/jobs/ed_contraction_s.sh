#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=ed-contract
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-9
#SBATCH --requeue
#
# E-D: contraction account — scale-corrected hinge vs uncorrected hinge.
# One task (cartpole-swingup state), 5 seeds × 2 arms.
# GATE: after E-A.5; can run in parallel with E-B.
#
#   arm 0 = corrected (anti_aliased_ppo)     exp_ed_tilde
#   arm 1 = uncorrected mu_PL hinge          exp_ed_raw
#
# Submit: sbatch src/experiments/jobs/ed_contraction_s.sh

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
export PYTHONPATH="$REPO"

SUITE=dmcontrol_state
TASK=cartpole-swingup
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"
SEEDS=(42 43 44 45 46)

IDX="${SLURM_ARRAY_TASK_ID}"
ARM=$((IDX / 5))
SEED="${SEEDS[$((IDX % 5))]}"

case "${ARM}" in
  0)
    EXP_NAME=exp_ed_tilde
    EXTRA=(--algo-preset anti_aliased_ppo)
    ;;
  1)
    EXP_NAME=exp_ed_raw
    # Same frozen-ref / phasing / clamp recipe, but hinge on uncorrected mu_PL.
    EXTRA=(
      --alpha 0 --beta 0.532
      --pl-on-ref-buffer --pl-f-mode clamp --head-phasing
      --alpha-warmup-epochs 0 --beta-warmup-epochs 500
    )
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

echo "E-D arm=${ARM} exp=${EXP_NAME} seed=${SEED} steps=${TOTAL_STEPS}"
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
