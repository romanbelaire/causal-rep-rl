#!/bin/bash
# Bridges-2: finish BASELINE / NOLINK / CTRO comparison (eval → tables → panels).
#
# Three-way:
#   Procgen:   exp_baseline | exp_latent_nolink | exp_ctro_cnn
#   DMControl: exp_baseline | exp_latent_nolink | exp_ctro_mlp_v2
#
# Deps are split by suite so a DMControl failure does not block Procgen aggs.
# SKIP_TRAIN=1  — no train jobs (assume checkpoints ready)
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs
JOBS=src/experiments/jobs
SKIP_TRAIN="${SKIP_TRAIN:-0}"

TRAIN_DEPS_PPO=""
TRAIN_DEPS_NL=""

if [ "${SKIP_TRAIN}" != "1" ]; then
  TRAIN_PPO=$(sbatch --parsable "${JOBS}/perf_train_procgen_baseline_finish_s.sh")
  echo "Procgen baseline train finish: ${TRAIN_PPO}"
  TRAIN_DEPS_PPO="--dependency=afterok:${TRAIN_PPO}"

  TRAIN_NL=$(sbatch --parsable "${JOBS}/perf_train_dmcontrol_latent_nolink_finish_s.sh")
  echo "DMControl latent_nolink train finish: ${TRAIN_NL}"
  TRAIN_DEPS_NL="--dependency=afterok:${TRAIN_NL}"
else
  echo "SKIP_TRAIN=1 — not submitting train jobs"
fi

# Separate wipe jobs so suites stay independent.
WIPE_PPO=$(sbatch --parsable ${TRAIN_DEPS_PPO} \
  --export=WIPE_TARGETS=procgen_baseline \
  "${JOBS}/perf_wipe_neg_control_evals_s.sh")
echo "Wipe procgen baseline eval: ${WIPE_PPO}"

WIPE_NL=$(sbatch --parsable ${TRAIN_DEPS_NL} \
  --export=WIPE_TARGETS=dmcontrol_nolink \
  "${JOBS}/perf_wipe_neg_control_evals_s.sh")
echo "Wipe dmcontrol nolink eval: ${WIPE_NL}"

EVAL_PPO=$(sbatch --parsable --dependency=afterok:"${WIPE_PPO}" \
  --export=EXP_NAME=exp_baseline \
  --job-name=ppo-perf-procgen \
  "${JOBS}/perf_eval_procgen_baseline_s.sh")
echo "Procgen baseline eval: ${EVAL_PPO}"

EVAL_NL=$(sbatch --parsable --dependency=afterok:"${WIPE_NL}" \
  --export=EXP_NAME=exp_latent_nolink \
  --job-name=nolink-perf-dmcontrol \
  "${JOBS}/perf_eval_dmcontrol_s.sh")
echo "DMControl latent_nolink eval: ${EVAL_NL}"

AGG_JOBS=()
for EXP in exp_baseline exp_latent_nolink exp_ctro_cnn; do
  AGG=$(sbatch --parsable --dependency=afterok:"${EVAL_PPO}" \
    --export=EXP_NAME="${EXP}" \
    "${JOBS}/perf_eval_agg_procgen_s.sh")
  echo "Agg procgen ${EXP}: ${AGG}"
  AGG_JOBS+=("${AGG}")
done

for EXP in exp_baseline exp_latent_nolink exp_ctro_mlp_v2; do
  AGG=$(sbatch --parsable --dependency=afterok:"${EVAL_NL}" \
    --export=EXP_NAME="${EXP}" \
    "${JOBS}/perf_eval_agg_dmcontrol_s.sh")
  echo "Agg dmcontrol ${EXP}: ${AGG}"
  AGG_JOBS+=("${AGG}")
done

PLOT_DEPS=$(IFS=:; echo "${AGG_JOBS[*]}")
PLOT=$(sbatch --parsable --dependency=afterok:"${PLOT_DEPS}" \
  "${JOBS}/perf_plot_neg_controls_s.sh")
echo "Neg-control panels: ${PLOT}"

echo
echo "Submitted neg-control pipeline."
echo "  evals: ${EVAL_PPO} ${EVAL_NL}"
echo "  plot:  ${PLOT}"
