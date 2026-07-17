#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.perf_train_procgen_baseline_s_%a.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=ppo-train-procgen

# Train one Procgen game (25M steps) with vanilla PPO across seeds 42/43/44.
# Array index selects the game. Checkpoints -> results/procgen_easy/exp_baseline/seed_{N}/{game}/
# Skips seeds with weights_final.pt; cancelled mid-run seeds restart from scratch.

module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0

export NVCC_FLAGS="-allow-unsupported-compiler"

cd /common/home/users/r/rbelaire.2021/causal-rep || exit 1
source rl-venv/bin/activate

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:~/LLM/llm/include/python3.10:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/include/python3.10
export LIBRARY_PATH=$LIBRARY_PATH:~/LLM/llm/lib:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export PYTHONUNBUFFERED=1

pip install -q procgen

TASKS=(coinrun starpilot caveflyer fruitbot chaser leaper maze miner)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="${EXP_NAME:-exp_baseline}"

mkdir -p results/slurm

nvidia-smi

for SEED in 42 43 44; do
  CKPT="results/procgen_easy/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
  if [ -f "${CKPT}" ]; then
    echo "Skipping seed ${SEED} — already finished: ${CKPT}"
    continue
  fi

  srun --gres=gpu:1 python -m src.experiments.run_performance_train \
    --suite procgen_easy \
    --task "${TASK}" \
    --seed "${SEED}" \
    --exp-name "${EXP_NAME}" \
    --agent ppo \
    --device cuda
done
