#!/bin/bash
# Submit theory validation v3 pipeline (kappa + distill arms).
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"

sbatch src/theory_validation/run_tv3_kappa_exp1.sh
sbatch src/theory_validation/run_tv3_kappa_exp2.sh
sbatch src/theory_validation/run_tv3_kappa_exp3.sh
sbatch src/theory_validation/run_tv3_distill_exp1.sh
sbatch src/theory_validation/run_tv3_distill_exp2.sh
