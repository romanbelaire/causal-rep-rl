#!/bin/bash
# Regenerate all theory_validation_v3 plots from existing metrics CSVs (CPU-only).
set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"

python -m src.theory_validation.plot_validation_v3 --arm kappa --exp all
python -m src.theory_validation.plot_validation_v3 --arm distill --exp all
echo "All tv3 plots written to plots/theory_validation_v3"
