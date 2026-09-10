#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=p1-ale-smoke
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.err
#SBATCH --requeue
#
# Mandatory Phoenix smoke ≤2M steps, ≥15 eval checkpoints, actor metrics present.
#
# Submit:
#   sbatch src/experiments/jobs/p1_ale_phoenix_smoke_s.sh

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
module load cuda/12.6.1 2>/dev/null || true
nvidia-smi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export FREEZE_ID="${FREEZE_ID:-ltro_phase0_v1}"

SUITE=ale
TASK=ALE/Phoenix-v5
SEED=42
TOTAL_STEPS=2000000
# 2e6/1024 ≈ 1953 epochs → eval every 130 ≈ 15 evals
EVAL_FREQ=130
EXP_NAME=exp_p1_ale_smoke_2m

python -m src.experiments.run_performance_train \
  --suite "${SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --exp-name "${EXP_NAME}" \
  --agent ppo \
  --num-epochs 4 \
  --total-steps "${TOTAL_STEPS}" \
  --eval-frequency "${EVAL_FREQ}" \
  --no-early-stop \
  --no-collapse \
  --device cuda \
  --results-root results

python - <<'PY'
from pathlib import Path
import pandas as pd
p = Path("results/ale/exp_p1_ale_smoke_2m/seed_42/ALE/Phoenix-v5/metrics.csv")
df = pd.read_csv(p)
evals = df.dropna(subset=["eval_full_return_mean"])
assert len(evals) >= 15, f"expected >=15 evals, got {len(evals)}"
for col in ("diag_cumulant_nmse", "onpolicy_cumulant_nmse", "diag_actor_preact_norm"):
    assert col in df.columns, f"missing {col}"
    assert evals[col].notna().any(), f"{col} all-NaN"
print(f"smoke ok evals={len(evals)} cols_ok")
PY
