#!/bin/bash
# Exp 0: build Z_ref tables from expert checkpoints (after run_exp0_expert)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --job-name=tv2-exp0-zref

set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"
module purge
module load Python/3.10.16-GCCcore-13.3.0
source rl-venv/bin/activate

ART=artifacts/theory_validation/z_ref
mkdir -p "$ART"

SEEDS=(42 43 44)
for S in "${SEEDS[@]}"; do
  EXPERT_RSTR="outputs/theory_validation_v2/expert_rstr_minigrid_seed${S}/weights_final.pt"
  EXPERT_VAE="outputs/theory_validation_v2/expert_vae_minigrid_seed${S}/weights_final.pt"
  if [[ ! -f "$EXPERT_RSTR" || ! -f "$EXPERT_VAE" ]]; then
    echo "Missing expert weights for seed $S:"
    echo "  RSTR: $EXPERT_RSTR"
    echo "  VAE:  $EXPERT_VAE"
    echo "Run submit_exp0_experts.sh first."
    exit 1
  fi
  python -m src.theory_validation.build_z_ref \
    --config configs/theory_validation/expert_rstr_minigrid.json \
    --weights "$EXPERT_RSTR" \
    --output "$ART/rstr_seed${S}.pt" \
    --seed "$S" --n-episodes 500
  python -m src.theory_validation.build_z_ref \
    --config configs/theory_validation/expert_vae_minigrid.json \
    --weights "$EXPERT_VAE" \
    --output "$ART/vae_seed${S}.pt" \
    --seed "$S" --n-episodes 500
  python -m src.theory_validation.validate_z_ref \
    --config configs/theory_validation/expert_rstr_minigrid.json \
    --expert-weights "$EXPERT_RSTR" \
    --z-ref "$ART/rstr_seed${S}.pt" \
    --output "$ART/exp0_validation_rstr_seed${S}.json"
done
