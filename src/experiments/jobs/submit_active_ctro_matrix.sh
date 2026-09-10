#!/bin/bash
# Submit Stage 5 MiniGrid matrix AFTER G0–G5. Does not submit Procgen.
# Usage:
#   bash src/experiments/jobs/submit_active_ctro_matrix.sh
#   ROW=4 bash src/experiments/jobs/submit_active_ctro_matrix.sh   # one row, three seeds

set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs

ROW="${ROW:-}"
if [ -n "${ROW}" ]; then
  START=$(( (ROW - 1) * 3 ))
  END=$(( START + 2 ))
  JOB=$(sbatch --parsable --array="${START}-${END}" src/experiments/jobs/active_ctro_matrix_s.sh)
  echo "matrix_row=${ROW} job=${JOB} array=${START}-${END}"
  exit 0
fi

JOB=$(sbatch --parsable src/experiments/jobs/active_ctro_matrix_s.sh)
echo "matrix_all=${JOB} array=0-20"
echo "Stop advancing if an earlier row fails. Do not treat this as a theorem claim until G0-G5 are green in NOTES.md."
