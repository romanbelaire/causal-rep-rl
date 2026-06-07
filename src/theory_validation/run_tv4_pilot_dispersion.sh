#!/bin/bash
# tv4 Phase 0: reward dispersion pilot (1 seed, 100 epochs)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --output=%u.tv4-pilot-dispersion.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv4-pilot

set -euo pipefail
source src/theory_validation/slurm_env_v4.sh

echo "=== tv4 pilot: MICo dispersion gate (seed 42, 100 epochs) ==="
tv4_srun_train "${TV4_CONFIG}/pilot_mico_dispersion.json" 42
echo "Check ${TV4_LOG}/theory_v4_pilot_mico_dispersion_seed42_intervention_loss.csv for train_mico_reward_pair_dispersion"
echo "tv4 pilot complete."
