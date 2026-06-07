#!/bin/bash
# tv4 exp4 DBC: bounding chain (seed 42)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv4-exp4-dbc.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv4-exp4-dbc

set -euo pipefail
source src/theory_validation/slurm_env_v4.sh

echo "=== tv4 exp4 DBC chain: training ==="
tv4_require_expert_weights dbc
tv4_srun_train "${TV4_CONFIG}/dbc_chain_minigrid.json" 42

echo "=== tv4 exp4 DBC chain: plots ==="
python -m src.theory_validation.plot_validation_v4 --arm dbc --exp 4
echo "tv4 exp4 DBC complete."
