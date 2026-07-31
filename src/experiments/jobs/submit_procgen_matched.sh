#!/bin/bash
# Submit matched Procgen theory comparison:
#   exp_ctro_cnn       — CNNEncoderCritic, policy-on-Z, α/β > 0, vae_coef=0
#   exp_latent_nolink  — same stack, α=β=0
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs

TRAIN_SCRIPT=src/experiments/jobs/perf_train_procgen_matched_s.sh
EVAL_SCRIPT=src/experiments/jobs/perf_eval_procgen_s.sh
AGG_SCRIPT=src/experiments/jobs/perf_eval_agg_procgen_s.sh

TRAIN_CTRO=$(sbatch --parsable \
  --export=EXP_NAME=exp_ctro_cnn,EXTRA_ARGS="--agent ctro" \
  --job-name=ctro-cnn-train-procgen \
  "${TRAIN_SCRIPT}")
echo "CTRO cnn train (exp_ctro_cnn): ${TRAIN_CTRO}"

TRAIN_NOLINK=$(sbatch --parsable \
  --export=EXP_NAME=exp_latent_nolink,EXTRA_ARGS="--agent ctro --alpha 0 --beta 0" \
  --job-name=nolink-train-procgen \
  "${TRAIN_SCRIPT}")
echo "Latent nolink train (exp_latent_nolink): ${TRAIN_NOLINK}"

EVAL_CTRO=$(sbatch --parsable --dependency=afterok:"${TRAIN_CTRO}" \
  --export=EXP_NAME=exp_ctro_cnn \
  --job-name=ctro-cnn-perf-procgen \
  "${EVAL_SCRIPT}")
echo "CTRO cnn eval: ${EVAL_CTRO}"

EVAL_NOLINK=$(sbatch --parsable --dependency=afterok:"${TRAIN_NOLINK}" \
  --export=EXP_NAME=exp_latent_nolink \
  --job-name=nolink-perf-procgen \
  "${EVAL_SCRIPT}")
echo "Latent nolink eval: ${EVAL_NOLINK}"

AGG_CTRO=$(sbatch --parsable --dependency=afterok:"${EVAL_CTRO}" \
  --export=EXP_NAME=exp_ctro_cnn \
  "${AGG_SCRIPT}")
echo "Aggregate exp_ctro_cnn: ${AGG_CTRO}"

AGG_NOLINK=$(sbatch --parsable --dependency=afterok:"${EVAL_NOLINK}" \
  --export=EXP_NAME=exp_latent_nolink \
  "${AGG_SCRIPT}")
echo "Aggregate exp_latent_nolink: ${AGG_NOLINK}"
