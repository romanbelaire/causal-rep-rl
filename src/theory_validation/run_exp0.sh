#!/bin/bash
# Theory validation v2 — Exp 0 (self-contained):
#   1. Train RSTR + VAE experts (seeds 42, 43, 44)
#   2. Build Z_ref tables and validate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp0.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp0

set -euo pipefail
source src/theory_validation/slurm_env.sh

echo "=== Exp 0: expert training (GELU) ==="
for S in "${TV2_SEEDS[@]}"; do
  tv2_srun_train configs/theory_validation/expert_rstr_minigrid.json "$S"
  tv2_srun_train configs/theory_validation/expert_vae_minigrid.json "$S"
done

echo "=== Exp 0: build Z_ref ==="
tv2_exp0_build_zref
echo "Exp 0 complete."
