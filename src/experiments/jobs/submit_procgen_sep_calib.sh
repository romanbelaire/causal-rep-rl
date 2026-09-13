#!/bin/bash
# λ=0 α-calib on starpilot seed 42, 5M steps (early window of a 25M run).
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs
JOB=$(sbatch --parsable --array=0-0 \
  --export=ALL,EXP_NAME=exp_sep_alpha_calib,TASKS="starpilot",SEEDS="42",EXTRA_ARGS="--agent ppo --policy-on-latent --sep-coef 0 --num-envs 64 --total-steps 5000000 --eval-frequency 50 --eval-episodes 5" \
  src/experiments/jobs/perf_train_procgen_sep_s.sh)
echo "sep alpha calib job ${JOB}"
echo "When done: python -m src.experiments.fit_sep_alpha --run-dir results/procgen_easy/exp_sep_alpha_calib/seed_42/starpilot"
