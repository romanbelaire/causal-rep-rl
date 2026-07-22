#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.perf_train_dmcontrol_latent_nolink_s_%a.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=nolink-train-dmcontrol

# Negative control 2: CTRO MLP-encoder stack (shared Z, policy-on-Z) with the
# value link OFF (alpha=0, beta=0, i.e. no MICo/PL). Shows a causally-informed
# representation still fails without the value coupling.
# One DMControl task per array task (serial rollout num_envs=1); inside it the 3
# seeds run as a concurrency pool (3 procs, ~1 core each) sharing one small GPU.
# Checkpoints -> results/dmcontrol_state/exp_latent_nolink/seed_{N}/{task}/
# Skips seeds with weights_final.pt. Override MAX_PARALLEL / THREADS_PER_PROC to tune.

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
export MUJOCO_GL=egl

pip install -q 'dm_control==1.0.38' 'mujoco==3.6.0' shimmy

mkdir -p results/slurm

nvidia-smi

ALL_TASKS=(cheetah-run walker-walk hopper-hop cartpole-swingup)
TASKS=("${ALL_TASKS[$SLURM_ARRAY_TASK_ID]}")
SEEDS=(42 43 44)
SUITE=dmcontrol_state
RESULTS_SUBDIR=dmcontrol_state
EXP_NAME="${EXP_NAME:-exp_latent_nolink}"
EXTRA_ARGS=(--agent ctro --alpha 0 --beta 0)
# Serial rollout (num_envs=1): 1 core/proc; this array task pools its 3 seeds.
ENVS_PER_PROC="${ENVS_PER_PROC:-1}"
THREADS_PER_PROC="${THREADS_PER_PROC:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

source src/experiments/jobs/_parallel_seeds.sh
run_training_pool
exit $?
