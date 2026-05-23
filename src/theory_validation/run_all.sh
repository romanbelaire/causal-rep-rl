#!/bin/bash
# Submit all theory validation SLURM jobs (baselines + collapse ablation).
set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"

chmod +x src/theory_validation/run_baselines.sh
chmod +x src/theory_validation/run_collapse_ablation.sh

echo "Submitting theory validation baselines..."
sbatch src/theory_validation/run_baselines.sh

echo "Submitting L_conv collapse ablation..."
sbatch src/theory_validation/run_collapse_ablation.sh

echo "After jobs finish, plot with:"
echo "  python -m src.theory_validation.plot_validation --log-dir outputs/theory_validation"
