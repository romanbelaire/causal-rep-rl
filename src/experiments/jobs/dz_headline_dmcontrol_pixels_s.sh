#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=dz-headline-pix
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x_%a.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x_%a.%j.err
#SBATCH --array=0-23%6
#SBATCH --requeue
#
# Phase B: dmcontrol_pixels CTRO ± D_Z (4 tasks × 3 seeds × 2 methods = 24 cells).
# One array task = one train process (avoids multi-seed-per-GPU OOM).
# Method 0: exp_ctro_dz  (+ --dz-enabled --eta-dz from λ=0 calib)
# Method 1: exp_ctro     (matched control, suite t8 HPs, no D_Z)
#
# Submit only after pilot passes. ETA_DZ must be set from calib
# (results/.../exp_ctro_dz_calib/.../eta_calib.txt); ratio-form 0.03 is obsolete.

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
TASKS=(cheetah-run walker-walk hopper-hop cartpole-swingup)
SEEDS=(42 43 44)
METHODS=(dz ctro)
if [ -z "${ETA_DZ:-}" ]; then
  CALIB_ETA="results/${SUITE}/exp_ctro_dz_calib/seed_42/cartpole-swingup/eta_calib.txt"
  if [ -f "${CALIB_ETA}" ]; then
    ETA_DZ=$(cat "${CALIB_ETA}")
    echo "Loaded ETA_DZ=${ETA_DZ} from ${CALIB_ETA}"
  else
    echo "ERROR: set ETA_DZ or run dz_eta_calib_pixels_cartpole_s.sh first"
    exit 1
  fi
fi

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -ge 24 ]; then
  echo "i=$i out of range"; exit 1
fi
method_i=$((i / 12))
rest=$((i % 12))
task_i=$((rest / 3))
seed_i=$((rest % 3))
METHOD=${METHODS[$method_i]}
TASK=${TASKS[$task_i]}
SEED=${SEEDS[$seed_i]}

if [ "${METHOD}" = "dz" ]; then
  EXP_NAME=exp_ctro_dz
  EXTRA=(--dz-enabled --eta-dz "${ETA_DZ}" --lambda-dz 1.0)
else
  EXP_NAME=exp_ctro
  EXTRA=()
fi

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  exit 0
fi

echo "Headline method=${METHOD} exp=${EXP_NAME} task=${TASK} seed=${SEED} array=${i}"
# shellcheck disable=SC2086
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --device cuda \
  --results-root results \
  "${EXTRA[@]}"
