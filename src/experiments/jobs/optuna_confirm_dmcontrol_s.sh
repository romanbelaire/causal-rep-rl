#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=2-00:00:00
#SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.optuna_confirm_dmcontrol_s_.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=optuna-confirm-dmc

# Full-budget (8M) retrain of an Optuna winner for one DMControl task × 3 seeds.
#
# Required env:
#   EXP_NAME          e.g. exp_optuna_confirm_exp_ctro_mlp_t12
#   TASK              e.g. cartpole-swingup
#   EXTRA_ARGS_STR    quoted CLI overrides from sensitivity confirm_commands.sh
#                     e.g. '--agent ctro --learning-rate 0.0003 ...'
#
# Example (after analyze_optuna_sensitivity):
#   bash results/optuna/dmcontrol_state/exp_ctro_mlp/cartpole-swingup/confirm_commands.sh
# or:
#   EXP_NAME=exp_optuna_confirm_ctro_t0 TASK=cartpole-swingup \
#     EXTRA_ARGS_STR='--agent ctro --learning-rate 3e-4 --entropy-coef 0.01 --num-epochs 10 --policy-hidden 64,64 --alpha 0.01 --beta 0.1' \
#     sbatch src/experiments/jobs/optuna_confirm_dmcontrol_s.sh

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

if [ -z "${EXP_NAME:-}" ] || [ -z "${TASK:-}" ] || [ -z "${EXTRA_ARGS_STR:-}" ]; then
  echo "Need EXP_NAME, TASK, and EXTRA_ARGS_STR" >&2
  exit 1
fi

# shellcheck disable=SC2206
EXTRA_ARGS=(${EXTRA_ARGS_STR})
SEEDS=(42 43 44)
SUITE=dmcontrol_state
RESULTS_SUBDIR=dmcontrol_state
TASKS=("${TASK}")
ENVS_PER_PROC="${ENVS_PER_PROC:-1}"
THREADS_PER_PROC="${THREADS_PER_PROC:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

echo "Confirm EXP_NAME=${EXP_NAME} TASK=${TASK} EXTRA_ARGS=${EXTRA_ARGS[*]}"

source src/experiments/jobs/_parallel_seeds.sh
run_training_pool
exit $?
