#!/bin/bash
# Submit theory validation v2 pipeline.
# Each run_expN.sh is self-contained; only Exp 4 needs Exp 0 expert weights on disk.
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"

sbatch src/theory_validation/run_exp0.sh
sbatch src/theory_validation/run_exp1.sh
sbatch src/theory_validation/run_exp2.sh
sbatch src/theory_validation/run_exp3.sh
sbatch src/theory_validation/run_exp4.sh
sbatch src/theory_validation/run_exp5.sh