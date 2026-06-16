#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.perf_train_dmcontrol_s_%a.%j.out
#SBATCH --error=%u.perf_train_dmcontrol_s_%a.%j.err
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=ctro-train-dmcontrol

# Train one DMControl task across seeds 42/43/44.
# Array index selects the task. Checkpoints -> results/dmcontrol_state/exp_full/seed_{N}/{task}/

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

TASKS=(cheetah-run walker-walk hopper-hop cartpole-swingup)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="${EXP_NAME:-exp_full}"

mkdir -p results/slurm

nvidia-smi

for SEED in 42 43 44; do
  srun --gres=gpu:1 python -m src.experiments.run_performance_train \
    --suite dmcontrol_state \
    --task "${TASK}" \
    --seed "${SEED}" \
    --exp-name "${EXP_NAME}" \
    --device cuda
done
