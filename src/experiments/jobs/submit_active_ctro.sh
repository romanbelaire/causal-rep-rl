#!/bin/bash
# Submit the active-CTRO pipeline on Bridges-2.
# CPU toys first; MiniGrid GPU array waits on toys; aggregation waits on MiniGrid.
# Does not submit Procgen (gated until MiniGrid + toys succeed).
#
# Usage:
#   bash src/experiments/jobs/submit_active_ctro.sh
#   TOYS_ONLY=1 bash src/experiments/jobs/submit_active_ctro.sh
#   MINIGRID_ONLY=1 bash src/experiments/jobs/submit_active_ctro.sh

set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs

TOYS_ONLY="${TOYS_ONLY:-0}"
MINIGRID_ONLY="${MINIGRID_ONLY:-0}"

if [ "${MINIGRID_ONLY}" = "1" ]; then
  MINI=$(sbatch --parsable src/experiments/jobs/active_ctro_minigrid_s.sh)
  AGG=$(sbatch --parsable --dependency=afterok:"${MINI}" src/experiments/jobs/active_ctro_agg_s.sh)
  echo "minigrid=${MINI} agg=${AGG}"
  exit 0
fi

TOYS=$(sbatch --parsable src/experiments/jobs/active_ctro_toys_s.sh)
echo "toys=${TOYS}"
if [ "${TOYS_ONLY}" = "1" ]; then
  exit 0
fi

MINI=$(sbatch --parsable --dependency=afterok:"${TOYS}" src/experiments/jobs/active_ctro_minigrid_s.sh)
AGG=$(sbatch --parsable --dependency=afterok:"${MINI}" src/experiments/jobs/active_ctro_agg_s.sh)
echo "minigrid=${MINI} agg=${AGG}"
echo "If agg shows DependencyNeverSatisfied, the MiniGrid array failed — check logs/actro-mg.*.err and resubmit with MINIGRID_ONLY=1"
