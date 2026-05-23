#!/bin/bash
# Submit full theory validation v2 pipeline (ordered dependencies).
set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"
chmod +x src/theory_validation/submit_exp0_experts.sh
chmod +x src/theory_validation/run_exp0_expert_rstr.sh
chmod +x src/theory_validation/run_exp0_expert_vae.sh
chmod +x src/theory_validation/run_exp0_build_zref.sh
chmod +x src/theory_validation/run_exp1_2_metrics.sh
chmod +x src/theory_validation/run_exp3_collapse.sh
chmod +x src/theory_validation/run_exp4_chain.sh
chmod +x src/theory_validation/run_exp5_transfer.sh

eval "$(bash src/theory_validation/submit_exp0_experts.sh)"
J1=$(sbatch --parsable --dependency="${EXPERT_JOB_DEP}" src/theory_validation/run_exp0_build_zref.sh)
J2=$(sbatch --parsable src/theory_validation/run_exp1_2_metrics.sh)
J3=$(sbatch --parsable src/theory_validation/run_exp3_collapse.sh)
J4=$(sbatch --parsable --dependency=afterok:${J1} src/theory_validation/run_exp4_chain.sh)
J5=$(sbatch --parsable src/theory_validation/run_exp5_transfer.sh)

echo "Submitted: exp0_expert=${EXPERT_JOB_IDS} exp0_zref=$J1 exp1-2=$J2 exp3=$J3 exp4=$J4 exp5=$J5"
echo "Plots: python -m src.theory_validation.plot_validation_v2 --log-dir outputs/theory_validation_v2"
