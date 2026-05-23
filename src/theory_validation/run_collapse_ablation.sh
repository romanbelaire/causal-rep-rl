#!/bin/bash
# Theory validation — controlled collapse (L_conv on vs off on RSTR VAE)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --constraint=h100nvl|h200|h100|l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.theory_validation_collapse.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=theory-val-collapse

set -euo pipefail
cd /common/home/users/r/rbelaire.2021/causal-rep

module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0
source rl-venv/bin/activate

source src/theory_validation/env_pytorch_alloc.sh

srun --gres=gpu:1 python -m src.main --config configs/theory_validation/rstr_with_lconv_minigrid.json
srun --gres=gpu:1 python -m src.main --config configs/theory_validation/rstr_no_lconv_minigrid.json
