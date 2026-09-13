#!/bin/bash
# λ_sep ∈ {0.01, 0.1, 1.0} on starpilot seed 42 after α is frozen.
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs
: "${ALPHA_SEP:?export ALPHA_SEP}"
for LAM in 0.01 0.1 1.0; do
  NAME="exp_sep_lambda_${LAM}"
  JOB=$(sbatch --parsable --array=0-0 \
    --export=ALL,EXP_NAME="${NAME}",TASKS="starpilot",SEEDS="42",EXTRA_ARGS="--agent ppo --policy-on-latent --sep-coef ${LAM} --alpha-sep ${ALPHA_SEP} --num-envs 64" \
    src/experiments/jobs/perf_train_procgen_sep_s.sh)
  echo "lambda ${LAM} job ${JOB} name ${NAME}"
done
