#!/bin/bash
# tv4 exp0: DBC expert training (seeds 42-44) + Z_ref sanity
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=%u.tv4-exp0-dbc.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv4-exp0-dbc

set -euo pipefail
source src/theory_validation/slurm_env_v4.sh

tv4_generate_expert_config dbc dbc_loss_coef 0.1 'cfg["algorithm"]["dbc_embed_ball_radius"] = 50.0'
GENERATED="${TV4_LOG}/generated_configs/expert_dbc_minigrid_stochastic_3000.json"

echo "=== tv4 exp0: DBC expert training ==="
for S in "${TV4_SEEDS[@]}"; do
  tv4_srun_train "${GENERATED}" "$S"
done

echo "=== tv4 exp0: DBC Z_ref sanity ==="
for S in "${TV4_SEEDS[@]}"; do
  python -m src.theory_validation.validate_bisim_z_ref --family dbc --seed "$S"
done
echo "tv4 exp0 DBC complete."
