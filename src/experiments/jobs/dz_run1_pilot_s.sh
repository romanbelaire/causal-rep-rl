#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=dz-run1-pilot
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%A_%a.err
#SBATCH --array=0-3
#SBATCH --requeue
#
# Run 1 pilot (seed 42, 1.5M steps): D_Z as the only policy trust region
# (--no-policy-clip) plus required stress control (no clip, no D_Z).
#
# Array:
#   0  stress   — no clip, D_Z absent          → exp_dz_run1_stress
#   1  run1     — no clip, η=0.101 (calibrated) → exp_dz_run1_eta0.101
#   2  run1     — no clip, η=0.32  (mid)        → exp_dz_run1_eta0.32
#   3  run1     — no clip, η=0.71  (loose≈√0.5) → exp_dz_run1_eta0.71
#
# Success criteria (pre-declared): docs/dz_run1_pilot_criteria.md
# Submit: sbatch src/experiments/jobs/dz_run1_pilot_s.sh

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
SEED=42
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"

ETA_CALIB_FILE="results/${SUITE}/exp_ctro_dz_calib/seed_42/${TASK}/eta_calib.txt"
if [ -f "${ETA_CALIB_FILE}" ]; then
  ETA_CALIB=$(cat "${ETA_CALIB_FILE}")
else
  ETA_CALIB=0.10123514
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    EXP_NAME=exp_dz_run1_stress
    # Same CTRO stack as Run 2 (suite α/β), policy clip off, D_Z absent.
    EXTRA=(--agent ctro --no-policy-clip)
    ;;
  1)
    EXP_NAME=exp_dz_run1_eta0.101
    EXTRA=(--dz-enabled --eta-dz "${ETA_CALIB}" --lambda-dz 1.0 --no-policy-clip)
    ;;
  2)
    EXP_NAME=exp_dz_run1_eta0.32
    EXTRA=(--dz-enabled --eta-dz 0.32 --lambda-dz 1.0 --no-policy-clip)
    ;;
  3)
    EXP_NAME=exp_dz_run1_eta0.71
    EXTRA=(--dz-enabled --eta-dz 0.71 --lambda-dz 1.0 --no-policy-clip)
    ;;
  *)
    echo "Unknown array id ${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "Run1 pilot id=${SLURM_ARRAY_TASK_ID} exp=${EXP_NAME} seed=${SEED} steps=${TOTAL_STEPS}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --device cuda \
  --results-root results \
  "${EXTRA[@]}"
