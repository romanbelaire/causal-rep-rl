#!/bin/bash
# Do not submit until contribution-1 predictiveness exists and α, λ are frozen.
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs
: "${ALPHA_SEP:?export ALPHA_SEP from python -m src.experiments.fit_sep_alpha}"
: "${LAMBDA_SEP:?export LAMBDA_SEP from the starpilot one-seed sweep}"
JOB=$(sbatch --parsable \
  --export=ALL,ALPHA_SEP="${ALPHA_SEP}",LAMBDA_SEP="${LAMBDA_SEP}" \
  src/experiments/jobs/perf_train_procgen_sep_screen_s.sh)
echo "sep screen job ${JOB} ALPHA_SEP=${ALPHA_SEP} LAMBDA_SEP=${LAMBDA_SEP}"
