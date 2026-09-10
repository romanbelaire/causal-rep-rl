#!/bin/bash
# D_Z orchestration helpers for Bridges-2 / jet.
# Usage: bash src/experiments/jobs/dz_phase3.sh {validate|calib|pilot|headline|requeue_pixels}
#
# Pixel-first contribution path (log-ratio D_Z):
#   validate → sbatch dz_eta_calib_pixels_cartpole_s.sh →
#   ETA_DZ=... sbatch dz_pilot_pixels_cartpole_s.sh → (gate) →
#   sbatch dz_headline_dmcontrol_pixels_s.sh → sbatch perf_eval_dmcontrol_pixels_dz_s.sh

set -euo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-/ocean/projects/cis260223p/rbelaire/causal-rep-rl/results}"
SUITE=dmcontrol_pixels
TASK_PILOT="${TASK_PILOT:-cartpole-swingup}"
ETA_DZ="${ETA_DZ:-}"
DEVICE="${DEVICE:-cuda}"
SEEDS=(42 43 44)
TASKS=(cheetah-run walker-walk hopper-hop cartpole-swingup)

run_one() {
  local exp="$1" task="$2" seed="$3"
  shift 3
  python -m src.experiments.run_performance_train \
    --suite "${SUITE}" \
    --task "${task}" \
    --seed "${seed}" \
    --exp-name "${exp}" \
    --results-root "${RESULTS_ROOT}" \
    --device "${DEVICE}" \
    "$@"
}

validate() {
  CUDA_VISIBLE_DEVICES= python -m src.experiments.validate_dz_geometry --device cpu
}

calib() {
  run_one "exp_ctro_dz_calib_sref" "${TASK_PILOT}" 42 \
    --agent ctro --dz-enabled --lambda-dz 0 --no-dz-adapt \
    --total-steps 1500000 --no-early-stop
  python -m src.experiments.fit_dz_eta_from_calib \
    --run-dir "${RESULTS_ROOT}/${SUITE}/exp_ctro_dz_calib_sref/seed_42/${TASK_PILOT}" \
    --early-frac 0.2
}

pilot() {
  if [ -z "${ETA_DZ}" ]; then
    local calib_eta="${RESULTS_ROOT}/${SUITE}/exp_ctro_dz_calib/seed_42/${TASK_PILOT}/eta_calib.txt"
    if [ -f "${calib_eta}" ]; then
      ETA_DZ=$(cat "${calib_eta}")
    else
      echo "ERROR: set ETA_DZ or run calib first"
      exit 1
    fi
  fi
  run_one "exp_ctro_dz" "${TASK_PILOT}" 42 \
    --agent ctro --dz-enabled --eta-dz "${ETA_DZ}" --lambda-dz 1.0
}

headline() {
  if [ -z "${ETA_DZ}" ]; then
    echo "ERROR: set ETA_DZ from calib before headline"
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
      run_one "exp_ctro_dz" "${task}" "${seed}" \
        --agent ctro --dz-enabled --eta-dz "${ETA_DZ}" --lambda-dz 1.0
      run_one "exp_ctro" "${task}" "${seed}" \
        --agent ctro
    done
  done
}

requeue_pixels() {
  echo "Prefer sbatch src/experiments/jobs/perf_requeue_dmcontrol_pixels_s.sh (1 GPU / cell)"
  exit 1
}

usage() {
  echo "Usage: $0 {validate|calib|pilot|headline|requeue_pixels}"
}

cmd="${1:-}"
case "${cmd}" in
  validate) validate ;;
  calib) calib ;;
  pilot) pilot ;;
  headline) headline ;;
  requeue_pixels) requeue_pixels ;;
  *) usage; exit 1 ;;
esac
