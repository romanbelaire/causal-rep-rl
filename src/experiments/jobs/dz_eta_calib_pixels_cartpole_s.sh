#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1-00:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=dz-eta-calib
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.err
#SBATCH --requeue
#
# λ=0 calibration for reference-scaled D_Z^log.
# Fit: η² = p25(D_Z) over the first 20% of logged updates; η to 2 sig digits.
#
# Then: ETA_DZ=<eta> sbatch src/experiments/jobs/dz_confirm_pixels_cartpole_s.sh

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
EXP_NAME="${EXP_NAME:-exp_ctro_dz_calib_sref}"
TOTAL_STEPS="${TOTAL_STEPS:-1500000}"

# Build shared diagnostic batch once (random policy obs + fixed pairs).
DIAG="results/${SUITE}/diag/${TASK}/dz_diag.pt"
if [ ! -f "${DIAG}" ]; then
  echo "Building frozen diag batch -> ${DIAG}"
  CUDA_VISIBLE_DEVICES= python -m src.experiments.build_dz_diag_batch \
    --suite "${SUITE}" --task "${TASK}" --out "${DIAG}"
fi

CKPT="results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
if [ -f "${CKPT}" ]; then
  echo "Skipping — already finished: ${CKPT}"
  python -m src.experiments.fit_dz_eta_from_calib \
    --run-dir "results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}" \
    --early-frac 0.2
  exit 0
fi

echo "η calib (s_ref): suite=${SUITE} task=${TASK} seed=${SEED} exp=${EXP_NAME} steps=${TOTAL_STEPS}"
python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ctro \
  --dz-enabled \
  --lambda-dz 0 \
  --no-dz-adapt \
  --total-steps "${TOTAL_STEPS}" \
  --no-early-stop \
  --device cuda \
  --results-root results

echo "Computing η from first 20% of D_Z..."
python -m src.experiments.fit_dz_eta_from_calib \
  --run-dir "results/${SUITE}/${EXP_NAME}/seed_${SEED}/${TASK}" \
  --early-frac 0.2
