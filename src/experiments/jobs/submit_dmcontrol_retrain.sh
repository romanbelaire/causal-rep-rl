#!/bin/bash
# Clear stuck DMControl deps and resubmit post-fix continuous PPO chain.
# CTRO uses exp_ctro_mlp_v2 (old exp_ctro_mlp finals are Linear log_std / invalid).
# PPO reuses exp_ppo_mlp finals (eval load now accepts [A] vs [1,A] log_std).
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs

# Cancel stuck agg jobs from the broken chain
scancel 42848296 42848297 2>/dev/null || true

CTRO_EXP=exp_ctro_mlp_v2
PPO_EXP=exp_ppo_mlp
TRAIN_SCRIPT=src/experiments/jobs/perf_train_dmcontrol_bridges_s.sh

TRAIN_CTRO=$(sbatch --parsable \
  --export=EXP_NAME="${CTRO_EXP}",EXTRA_ARGS="--agent ctro" \
  --job-name=ctro-train-dmcontrol \
  "${TRAIN_SCRIPT}")
echo "CTRO train (${CTRO_EXP}): ${TRAIN_CTRO}"

EVAL_CTRO=$(sbatch --parsable --dependency=afterok:"${TRAIN_CTRO}" \
  --export=EXP_NAME="${CTRO_EXP}" \
  src/experiments/jobs/perf_eval_dmcontrol_s.sh)
echo "CTRO eval: ${EVAL_CTRO}"

# PPO training already completed; only re-run eval + agg
EVAL_PPO=$(sbatch --parsable \
  --export=EXP_NAME="${PPO_EXP}" \
  src/experiments/jobs/perf_eval_dmcontrol_baseline_s.sh)
echo "PPO eval (${PPO_EXP}): ${EVAL_PPO}"

AGG_CTRO=$(sbatch --parsable --dependency=afterok:"${EVAL_CTRO}" \
  --export=EXP_NAME="${CTRO_EXP}" \
  src/experiments/jobs/perf_eval_agg_dmcontrol_s.sh)
echo "Aggregate ${CTRO_EXP}: ${AGG_CTRO}"

AGG_PPO=$(sbatch --parsable --dependency=afterok:"${EVAL_PPO}" \
  --export=EXP_NAME="${PPO_EXP}" \
  src/experiments/jobs/perf_eval_agg_dmcontrol_s.sh)
echo "Aggregate ${PPO_EXP}: ${AGG_PPO}"
