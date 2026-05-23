#!/bin/bash

#################################################
## VANILLA PPO BASELINE: IMPALA + MLP CRITIC  ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
##SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.vanilla_ppo_impala_mlp.%j.out
#SBATCH --requeue

#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch 
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=vanilla-ppo-impala-mlp

module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0

export NVCC_FLAGS="-allow-unsupported-compiler"

# Change to script directory (project root)
cd /common/home/users/r/rbelaire.2021/causal-rep || exit 1

source rl-venv/bin/activate

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:~/LLM/llm/include/python3.10:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/include/python3.10
export LIBRARY_PATH=$LIBRARY_PATH:~/LLM/llm/lib:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib

export DS_BUILD_OPS=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_UTILS=0
export DS_BUILD_AIO=0

nvidia-smi
nvcc --version

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

# Run training
srun --gres=gpu:1 python -m src.main --config configs/exp1_vanilla_ppo_impala_mlp.json

