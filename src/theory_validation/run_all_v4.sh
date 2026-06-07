#!/bin/bash
# Submit theory validation v4 pipeline with SLURM dependencies.
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"

J0=$(sbatch --parsable src/theory_validation/run_tv4_pilot_dispersion.sh)
echo "Pilot job: ${J0}"

J1=$(sbatch --parsable --dependency=afterok:${J0} src/theory_validation/run_tv4_exp0_mico_expert.sh)
echo "MICo expert job: ${J1}"

J2=$(sbatch --parsable --dependency=afterok:${J0} src/theory_validation/run_tv4_exp0_dbc_expert.sh)
echo "DBC expert job: ${J2}"

J3=$(sbatch --parsable --dependency=afterok:${J1} src/theory_validation/run_tv4_exp1_mico.sh)
echo "MICo exp1 job: ${J3}"

J4=$(sbatch --parsable --dependency=afterok:${J2} src/theory_validation/run_tv4_exp1_dbc.sh)
echo "DBC exp1 job: ${J4}"

J5=$(sbatch --parsable --dependency=afterok:${J1} src/theory_validation/run_tv4_exp4_chain_mico.sh)
echo "MICo chain job: ${J5}"

J6=$(sbatch --parsable --dependency=afterok:${J2} src/theory_validation/run_tv4_exp4_chain_dbc.sh)
echo "DBC chain job: ${J6}"

J7=$(sbatch --parsable --dependency=afterok:${J3}:${J4}:${J5}:${J6} src/theory_validation/run_tv4_plot_all.sh)
echo "Plot-all job: ${J7}"

echo "Submitted tv4 pipeline. Plot job ${J7} runs after all training completes."
