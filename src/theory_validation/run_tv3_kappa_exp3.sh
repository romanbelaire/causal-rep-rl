#!/bin/bash
# tv3 kappa arm — exp3: bounding chain (seed 42)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv3-kappa-exp3.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv3-kappa-exp3

set -euo pipefail
source src/theory_validation/slurm_env_v3.sh

echo "=== tv3 kappa exp3: training ==="
tv3_require_expert_weights rstr
tv3_srun_train configs/theory_validation_v3/kappa_dir_chain_rstr_minigrid.json 42

echo "=== tv3 kappa exp3: plots ==="
python -m src.theory_validation.plot_validation_v3 --arm kappa --exp 3
echo "tv3 kappa exp3 complete."
