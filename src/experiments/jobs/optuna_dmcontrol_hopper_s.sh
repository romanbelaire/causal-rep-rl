#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.optuna_dmcontrol_hopper_s_%a.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=optuna-hopper

# Hopper-only Optuna v2:
#   - no return-collapse (MedianPruner + NaN only)
#   - 8M steps per trial
#   - warm-start from cheetah/walker best_trial.json
#   - fresh study DB under results/optuna/dmcontrol_state/{agent}/hopper-hop_v2/
#
# Array: 0=exp_baseline, 1=exp_ctro_mlp
#
#   N_TRIALS=10 sbatch src/experiments/jobs/optuna_dmcontrol_hopper_s.sh

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
AGENT=${AGENTS[$SLURM_ARRAY_TASK_ID]}
TASK=hopper-hop
N_TRIALS="${N_TRIALS:-10}"
SEED="${SEED:-42}"
# Default in optuna_dmcontrol is 8M for hopper; allow override.
SEARCH_STEPS="${SEARCH_STEPS:-8000000}"

echo "Hopper Optuna v2 agent=${AGENT} n_trials=${N_TRIALS} search_steps=${SEARCH_STEPS}"

python -m src.experiments.optuna_dmcontrol \
  --agent "${AGENT}" \
  --task "${TASK}" \
  --n-trials "${N_TRIALS}" \
  --seed "${SEED}" \
  --search-steps "${SEARCH_STEPS}" \
  --device cuda \
  --results-root results \
  --optuna-root results/optuna

echo "Done hopper agent=${AGENT}"
