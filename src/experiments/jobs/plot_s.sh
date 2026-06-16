#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END
#SBATCH --output=%u.plot_s_.%j.out
#SBATCH --error=%u.plot_s_.%j.err
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=ctro-plot

module purge
module load Python/3.10.16-GCCcore-13.3.0

export NVCC_FLAGS="-allow-unsupported-compiler"

cd /common/home/users/r/rbelaire.2021/causal-rep || exit 1
source rl-venv/bin/activate

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:~/LLM/llm/include/python3.10:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/include/python3.10
export LIBRARY_PATH=$LIBRARY_PATH:~/LLM/llm/lib:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib

mkdir -p plots/ctro

python -m src.experiments.plot_ctro
python -m src.experiments.analyze_return_curves
python -m src.experiments.analyze_pr_manifold
python -m src.experiments.aggregate_ablation
python -m src.experiments.aggregate_results
