#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=neg-control-plots
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --array=0-0
#SBATCH --requeue

# Panel A/B/C for BASELINE / LATENT_NOLINK / CTRO on both performance suites.

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs plots

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
export PYTHONUNBUFFERED=1

PROCGEN_TASKS=(coinrun starpilot caveflyer fruitbot chaser leaper maze miner)
DMCONTROL_TASKS=(cartpole-swingup cheetah-run hopper-hop walker-walk)

rc=0
for TASK in "${PROCGEN_TASKS[@]}"; do
  echo "=== procgen_easy / ${TASK} ==="
  python -m src.experiments.plot_performance_panels \
    --suite procgen_easy \
    --task "${TASK}" || rc=1
done

for TASK in "${DMCONTROL_TASKS[@]}"; do
  echo "=== dmcontrol_state / ${TASK} ==="
  python -m src.experiments.plot_performance_panels \
    --suite dmcontrol_state \
    --task "${TASK}" || rc=1
done

exit "${rc}"
