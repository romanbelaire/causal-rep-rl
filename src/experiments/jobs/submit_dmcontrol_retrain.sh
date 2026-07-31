#!/bin/bash
# Fresh DMControl CTRO + PPO after continuous-PPO fixes.
# Uses new exp names so pre-fix finals/latests are not skipped/resumed.
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs

CTRO_EXP=exp_ctro_mlp
PPO_EXP=exp_ppo_mlp

TRAIN_CTRO=$(sbatch --parsable --export=EXP_NAME="${CTRO_EXP}" \
  src/experiments/jobs/perf_train_dmcontrol_s.sh)
echo "CTRO train (${CTRO_EXP}): ${TRAIN_CTRO}"

TRAIN_PPO=$(sbatch --parsable --export=EXP_NAME="${PPO_EXP}" \
  src/experiments/jobs/perf_train_dmcontrol_baseline_s.sh)
echo "PPO train (${PPO_EXP}): ${TRAIN_PPO}"

EVAL_CTRO=$(sbatch --parsable --dependency=afterok:"${TRAIN_CTRO}" \
  --export=EXP_NAME="${CTRO_EXP}" \
  src/experiments/jobs/perf_eval_dmcontrol_s.sh)
echo "CTRO eval: ${EVAL_CTRO}"

EVAL_PPO=$(sbatch --parsable --dependency=afterok:"${TRAIN_PPO}" \
  --export=EXP_NAME="${PPO_EXP}" \
  src/experiments/jobs/perf_eval_dmcontrol_baseline_s.sh)
echo "PPO eval: ${EVAL_PPO}"

# Aggregate DMControl only (procgen roots do not exist under these exp names).
AGG_CTRO=$(sbatch --parsable --dependency=afterok:"${EVAL_CTRO}" \
  --export=EXP_NAME="${CTRO_EXP}" \
  src/experiments/jobs/perf_eval_agg_dmcontrol_s.sh)
echo "Aggregate ${CTRO_EXP}: ${AGG_CTRO}"

AGG_PPO=$(sbatch --parsable --dependency=afterok:"${EVAL_PPO}" \
  --export=EXP_NAME="${PPO_EXP}" \
  src/experiments/jobs/perf_eval_agg_dmcontrol_s.sh)
echo "Aggregate ${PPO_EXP}: ${AGG_PPO}"
