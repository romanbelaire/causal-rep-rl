#!/bin/bash
# Theory validation v2 — Exp 1 (self-contained):
#   H1/H2 for both model families (3 seeds each):
#     - RSTR (repr loss + lconv): theory_rstr_lconv_on_v2 → z_ref_expert_family rstr
#     - VAE baseline (no repr loss): theory_vae_ppo_no_repr_loss_v2 → z_ref_expert_family vae
#   Plots: κ_concave vs rank (PR + PCA); μ_concave fallback if κ absent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp1.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp1

set -euo pipefail
source src/theory_validation/slurm_env.sh

echo "=== Exp 1: training ==="
tv2_exp12_train

echo "=== Exp 1: plots ==="
python -m src.theory_validation.plot_validation_v2 --exp 1
echo "Exp 1 complete."
