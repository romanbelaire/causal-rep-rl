#!/bin/bash
# Post-gate expand: remaining six games × A0/A1/A2, then A3 on the screen games.
# Do not submit unless screen P1, P2, and P3 all hold.
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs
: "${ALPHA_SEP:?}"
: "${LAMBDA_SEP:?}"
: "${Z_L2_COEF:?match A1 own_sep_encoder_grad before submitting A3}"

REST="coinrun fruitbot caveflyer chaser miner leaper"
SEEDS="42 43 44 45 46"

A0=$(sbatch --parsable --array=0-29 \
  --export=ALL,EXP_NAME=exp_sep_a0,TASKS="${REST}",SEEDS="${SEEDS}",EXTRA_ARGS="--agent ppo --policy-on-latent --sep-coef 0 --num-envs 64" \
  src/experiments/jobs/perf_train_procgen_sep_s.sh)
A1=$(sbatch --parsable --array=0-29 \
  --export=ALL,EXP_NAME=exp_sep_a1,TASKS="${REST}",SEEDS="${SEEDS}",EXTRA_ARGS="--agent ppo --policy-on-latent --sep-coef ${LAMBDA_SEP} --alpha-sep ${ALPHA_SEP} --num-envs 64" \
  src/experiments/jobs/perf_train_procgen_sep_s.sh)
A2=$(sbatch --parsable --array=0-29 \
  --export=ALL,EXP_NAME=exp_sep_a2,TASKS="${REST}",SEEDS="${SEEDS}",EXTRA_ARGS="--agent ppo --policy-on-latent --sep-coef ${LAMBDA_SEP} --alpha-sep ${ALPHA_SEP} --sep-shuffle-returns --num-envs 64" \
  src/experiments/jobs/perf_train_procgen_sep_s.sh)
A3=$(sbatch --parsable --array=0-9 \
  --export=ALL,EXP_NAME=exp_sep_a3,TASKS="starpilot maze",SEEDS="${SEEDS}",EXTRA_ARGS="--agent ppo --policy-on-latent --z-l2-coef ${Z_L2_COEF} --num-envs 64" \
  src/experiments/jobs/perf_train_procgen_sep_s.sh)
echo "expand A0=${A0} A1=${A1} A2=${A2} A3=${A3}"
