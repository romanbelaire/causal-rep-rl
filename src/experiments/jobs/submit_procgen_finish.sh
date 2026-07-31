#!/bin/bash
# Submit remaining CTRO Procgen training, then eval + tables for the comparison.
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs

TRAIN=$(sbatch --parsable src/experiments/jobs/perf_train_procgen_finish_s.sh)
echo "CTRO train finish: ${TRAIN}"

EVAL_CTRO=$(sbatch --parsable --dependency=afterok:"${TRAIN}" \
  src/experiments/jobs/perf_eval_procgen_s.sh)
echo "CTRO eval (after train): ${EVAL_CTRO}"

EVAL_PPO=$(sbatch --parsable src/experiments/jobs/perf_eval_procgen_baseline_s.sh)
echo "PPO baseline eval: ${EVAL_PPO}"

AGG_CTRO=$(sbatch --parsable --dependency=afterok:"${EVAL_CTRO}" \
  --export=EXP_NAME=exp_full \
  src/experiments/jobs/perf_eval_agg_s.sh)
echo "Aggregate exp_full: ${AGG_CTRO}"

AGG_PPO=$(sbatch --parsable --dependency=afterok:"${EVAL_PPO}" \
  --export=EXP_NAME=exp_baseline \
  src/experiments/jobs/perf_eval_agg_s.sh)
echo "Aggregate exp_baseline: ${AGG_PPO}"
