#!/bin/bash
# Submit one SLURM job per (architecture, seed). Prints job IDs for dependency chaining.
set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"
chmod +x src/theory_validation/run_exp0_expert_rstr.sh
chmod +x src/theory_validation/run_exp0_expert_vae.sh

SEEDS=(42 43 44)
JIDS=()

for S in "${SEEDS[@]}"; do
  JIDS+=($(sbatch --parsable --export=ALL,SEED="$S" src/theory_validation/run_exp0_expert_rstr.sh))
  JIDS+=($(sbatch --parsable --export=ALL,SEED="$S" src/theory_validation/run_exp0_expert_vae.sh))
done

DEP="afterok"
for J in "${JIDS[@]}"; do
  DEP="${DEP}:${J}"
done

echo "Submitted ${#JIDS[@]} expert jobs: ${JIDS[*]}" >&2
echo "export EXPERT_JOB_IDS='${JIDS[*]}'"
echo "export EXPERT_JOB_DEP='${DEP}'"
