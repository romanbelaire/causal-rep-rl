#!/bin/bash
# Submit theory validation v2 pipeline.
# Exp 0: train VAE + RSTR experts and build Z_ref (weights_expert.pt).
# Exp 4 rebuilds Z_ref from experts if run alone after both Exp 0 jobs finish.
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"

sbatch src/theory_validation/run_exp0.sh
sbatch src/theory_validation/run_exp0_rstr.sh
sbatch src/theory_validation/run_exp1.sh
sbatch src/theory_validation/run_exp2.sh
sbatch src/theory_validation/run_exp3.sh
sbatch src/theory_validation/run_exp4.sh
sbatch src/theory_validation/run_exp5.sh