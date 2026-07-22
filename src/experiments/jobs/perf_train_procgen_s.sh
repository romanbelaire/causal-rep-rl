#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96gb
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.perf_train_procgen_s_%a.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=ctro-train-procgen

# Train ONE Procgen game (array index selects it) with CTRO (CNN-VAE stack, 25M
# steps, serial rollout num_envs=1). Each game gets its own allocation, and inside
# it the 3 seeds run as a concurrency pool (3 procs, ~1 core each) sharing one GPU.
# Checkpoints -> results/procgen_easy/exp_full/seed_{N}/{game}/
# Skips seeds with weights_final.pt. Lower MAX_PARALLEL if VRAM OOMs.

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

mkdir -p results/slurm

nvidia-smi

ALL_TASKS=(coinrun starpilot caveflyer fruitbot chaser leaper maze miner)
TASKS=("${ALL_TASKS[$SLURM_ARRAY_TASK_ID]}")
SEEDS=(42 43 44)
SUITE=procgen_easy
RESULTS_SUBDIR=procgen_easy
EXP_NAME="${EXP_NAME:-exp_full}"
EXTRA_ARGS=(--agent ctro)
# Serial rollout (num_envs=1): ~1 core/proc; this array task pools its 3 seeds.
ENVS_PER_PROC="${ENVS_PER_PROC:-1}"
THREADS_PER_PROC="${THREADS_PER_PROC:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

source src/experiments/jobs/_parallel_seeds.sh
run_training_pool
exit $?
