#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=dz-pilot-pix
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.err
#SBATCH --requeue
#
# Phase A pilot (log-ratio D_Z): dmcontrol_pixels + CTRO + D_Z, cartpole-swingup seed 42.
# Suite HPs (t8 + α/β warmup). One process, one GPU. Walltime 2 days.
#
# Prerequisites:
#   1) CUDA_VISIBLE_DEVICES= python -m src.experiments.validate_dz_geometry --device cpu
#   2) sbatch dz_eta_calib_pixels_cartpole_s.sh → set ETA_DZ=sqrt(p25)
#      (ratio-form η=0.03 is obsolete under log-ratio)
#
# Pass gates: λ interior (not pinned at 1e4), collapse diagnostics logged
# (dz_collapse_frac / unjustified), return competitive with pixel CTRO.

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
EXP_NAME="${EXP_NAME:-exp_ctro_dz}"
# Must come from λ=0 calib (fit_dz_eta_from_calib). No silent default to old 0.03.
if [ -z "${ETA_DZ:-}" ]; then
  CALIB_ETA="results/${SUITE}/exp_ctro_dz_calib/seed_${SEED}/${TASK}/eta_calib.txt"
  if [ -f "${CALIB_ETA}" ]; then
    ETA_DZ=$(cat "${CALIB_ETA}")
    echo "Loaded ETA_DZ=${ETA_DZ} from ${CALIB_ETA}"
  else
    echo "ERROR: set ETA_DZ or run dz_eta_calib_pixels_cartpole_s.sh first (${CALIB_ETA})"
    exit 1
  fi
fi

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "Pilot D_Z (log-ratio): suite=${SUITE} task=${TASK} seed=${SEED} eta=${ETA_DZ} exp=${EXP_NAME}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --dz-enabled \
  --eta-dz "${ETA_DZ}" \
  --lambda-dz 1.0 \
  --device cuda \
  --results-root results
