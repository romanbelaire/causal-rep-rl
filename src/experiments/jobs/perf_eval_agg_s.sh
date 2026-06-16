#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=END
#SBATCH --output=%u.perf_eval_agg_s_.%j.out
#SBATCH --error=%u.perf_eval_agg_s_.%j.err
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=ctro-perf-agg

# Aggregate performance eval CSVs into summary tables.
#
# Override at submit time, e.g.:
#   EXP_NAME=exp_full sbatch perf_eval_agg_s.sh

module purge
module load Python/3.10.16-GCCcore-13.3.0

export NVCC_FLAGS="-allow-unsupported-compiler"

cd /common/home/users/r/rbelaire.2021/causal-rep || exit 1
source rl-venv/bin/activate

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:~/LLM/llm/include/python3.10:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/include/python3.10
export LIBRARY_PATH=$LIBRARY_PATH:~/LLM/llm/lib:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib

EXP_NAME="${EXP_NAME:-exp_full}"
PROCGEN_ROOT="results/perf_eval/procgen_easy/${EXP_NAME}"
DMCONTROL_ROOT="results/perf_eval/dmcontrol_state/${EXP_NAME}"
TABLE_DIR="results/perf_eval/tables/${EXP_NAME}"

mkdir -p "${TABLE_DIR}"

python -m src.experiments.aggregate_performance_eval "${PROCGEN_ROOT}" \
  --output "${TABLE_DIR}/procgen_easy.txt"

python -m src.experiments.aggregate_performance_eval "${DMCONTROL_ROOT}" \
  --output "${TABLE_DIR}/dmcontrol_state.txt"
