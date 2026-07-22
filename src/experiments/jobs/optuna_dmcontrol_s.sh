#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.optuna_dmcontrol_s_%a.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=optuna-dmcontrol

# Optuna workers for DMControl HP search (one study per agent×task).
# Array 0-7: agent = index//4, task = index%4.
# Shared SQLite study DB under results/optuna/dmcontrol_state/{agent}/{task}/study.db
#
#   N_TRIALS=20 sbatch src/experiments/jobs/optuna_dmcontrol_s.sh
#   After studies finish:
#   python -m src.experiments.analyze_optuna_sensitivity --optuna-root results/optuna --all

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

pip install -q 'dm_control==1.0.38' 'mujoco==3.6.0' shimmy optuna

mkdir -p results/slurm results/optuna

nvidia-smi

AGENTS=(exp_baseline exp_ctro_mlp)
TASKS=(cheetah-run walker-walk hopper-hop cartpole-swingup)

IDX=${SLURM_ARRAY_TASK_ID}
AGENT=${AGENTS[$((IDX / 4))]}
TASK=${TASKS[$((IDX % 4))]}
N_TRIALS="${N_TRIALS:-20}"
SEED="${SEED:-42}"
SEARCH_STEPS="${SEARCH_STEPS:-1000000}"

echo "Optuna worker idx=${IDX} agent=${AGENT} task=${TASK} n_trials=${N_TRIALS} search_steps=${SEARCH_STEPS}"

python -m src.experiments.optuna_dmcontrol \
  --agent "${AGENT}" \
  --task "${TASK}" \
  --n-trials "${N_TRIALS}" \
  --seed "${SEED}" \
  --search-steps "${SEARCH_STEPS}" \
  --device cuda \
  --results-root results \
  --optuna-root results/optuna

echo "Done agent=${AGENT} task=${TASK}"
