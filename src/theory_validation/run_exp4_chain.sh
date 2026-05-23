#!/bin/bash
# Exp 4: bounding chain with Z_ref (requires artifacts from exp0)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --constraint=h100nvl|h200|h100|l40|l40s|a100|a40
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --job-name=tv2-exp4

set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"
module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0
source rl-venv/bin/activate
source src/theory_validation/env_pytorch_alloc.sh

ART=artifacts/theory_validation/z_ref
if [[ ! -f "$ART/rstr_seed42.pt" ]]; then
  echo "Missing $ART/rstr_seed42.pt — run exp0 build_zref first"
  exit 1
fi

# Patch z_ref path into temp configs or use env — configs point to seed42 by default
srun --gres=gpu:1 python -m src.main --config configs/theory_validation/exp4_rstr_chain_minigrid.json --seed 42
srun --gres=gpu:1 python -m src.main --config configs/theory_validation/exp4_vae_chain_minigrid.json --seed 42

python -m src.theory_validation.plot_validation_v2 --exp 4
