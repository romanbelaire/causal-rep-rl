#!/bin/bash
# Regenerate all theory_validation_v4 plots from existing CSVs (CPU-only).
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --output=%u.tv4-plot-all.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv4-plot-all

set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"

module purge
module load Python/3.10.16-GCCcore-13.3.0
source rl-venv/bin/activate

python -m src.theory_validation.plot_validation_v4 --arm mico --exp all
python -m src.theory_validation.plot_validation_v4 --arm dbc --exp all
python -m src.theory_validation.interpret_tv4
echo "All tv4 plots written to plots/theory_validation_v4"
