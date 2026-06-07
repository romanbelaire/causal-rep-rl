#!/bin/bash
# Theory validation v2 — Exp 4 (self-contained):
#   1. Rebuild Z_ref tables from weights_expert.pt (requires run_exp0.sh + run_exp0_rstr.sh)
#   2. RSTR + VAE bounding-chain runs (seed 42)
#   3. Plots
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp4.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp4

set -euo pipefail
source src/theory_validation/slurm_env.sh

echo "=== Exp 4: rebuild Z_ref (run_exp0_build_zref.sh) ==="
bash src/theory_validation/run_exp0_build_zref.sh

echo "=== Exp 4: bounding-chain training ==="
tv2_srun_train configs/theory_validation/exp4_rstr_chain_minigrid.json 42
tv2_srun_train configs/theory_validation/exp4_vae_chain_minigrid.json 42

echo "=== Exp 4: plots ==="
python -m src.theory_validation.plot_validation_v2 --exp 4
echo "Exp 4 complete."
