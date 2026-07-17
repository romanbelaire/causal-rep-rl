#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --time=0-02:00:00
#SBATCH --constraint=l40|l40s|a100|a40
#SBATCH --mail-type=END
#SBATCH --output=%u.perf_eval_dmcontrol_baseline_s_.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=ppo-perf-dmcontrol

# DMControl state-obs PPO baseline performance eval (4 tasks, full + test distributions).
#
# Expected checkpoint layout (per seed, per task):
#   ${CHECKPOINT_ROOT}/seed_${SEED}/{task}/weights_final.pt
#   ${CHECKPOINT_ROOT}/seed_${SEED}/{task}/config.json
#
# Example tasks: cheetah-run, walker-walk, hopper-hop, cartpole-swingup
#
# Override at submit time, e.g.:
#   EXP_NAME=exp_baseline SEEDS="42 43 44" sbatch perf_eval_dmcontrol_baseline_s.sh
# Skips seeds with performance_eval_metrics.csv.

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

EXP_NAME="${EXP_NAME:-exp_baseline}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-results/dmcontrol_state/${EXP_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/perf_eval/dmcontrol_state/${EXP_NAME}}"
SEEDS="${SEEDS:-42 43 44}"

mkdir -p results/slurm "${OUTPUT_ROOT}"

pip install -q 'dm_control==1.0.38' 'mujoco==3.6.0' shimmy

PROBE_CKPT="${CHECKPOINT_ROOT}/seed_42/cheetah-run/weights_final.pt"
if [ ! -f "${PROBE_CKPT}" ]; then
  echo "ERROR: missing checkpoint ${PROBE_CKPT}"
  echo "Run training first: sbatch src/experiments/jobs/perf_train_dmcontrol_baseline_s.sh"
  exit 1
fi

nvidia-smi

for SEED in ${SEEDS}; do
  CKPT_DIR="${CHECKPOINT_ROOT}/seed_${SEED}"
  OUT_DIR="${OUTPUT_ROOT}/seed_${SEED}"
  METRICS="${OUT_DIR}/performance_eval_metrics.csv"
  if [ -f "${METRICS}" ]; then
    echo "Skipping seed ${SEED} — already finished: ${METRICS}"
    continue
  fi
  mkdir -p "${OUT_DIR}"

  srun --gres=gpu:1 python -m src.experiments.run_performance_eval \
    --suite dmcontrol_state \
    --checkpoint "${CKPT_DIR}" \
    --config "${CKPT_DIR}" \
    --output-dir "${OUT_DIR}" \
    --device cuda
done
