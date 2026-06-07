#!/bin/bash
# Theory validation v2 — Exp 0 RSTR expert bootstrap (standalone SBATCH):
#   1. Train RSTR-family experts (repr loss + VAE) with stochastic eval until 95% expert ckpt.
#   2. Build Z_ref tables from weights_expert.pt, not weights_final.pt.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp0-rstr.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp0-rstr

set -euo pipefail

REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "${REPO}" || exit 1

module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0

export NVCC_FLAGS="-allow-unsupported-compiler"
source rl-venv/bin/activate

export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH:-}:~/LLM/llm/include/python3.10:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/include/python3.10
export LIBRARY_PATH=${LIBRARY_PATH:-}:~/LLM/llm/lib:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/opt/apps/software/Python/3.10.16-GCCcore-13.3.0/lib

export DS_BUILD_OPS=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_UTILS=0
export DS_BUILD_AIO=0

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF

nvidia-smi
nvcc --version

TV2_LOG=outputs/theory_validation_v2
TV2_ZREF=artifacts/theory_validation/z_ref
TV2_SEEDS=(44)
GENERATED_CONFIG="${TV2_LOG}/generated_configs/expert_rstr_minigrid_stochastic_3000.json"

mkdir -p "$(dirname "${GENERATED_CONFIG}")" "${TV2_ZREF}"

python - <<'PY'
import json
from pathlib import Path

base = Path("configs/theory_validation/expert_rstr_minigrid.json")
out = Path("outputs/theory_validation_v2/generated_configs/expert_rstr_minigrid_stochastic_3000.json")

with base.open() as f:
    cfg = json.load(f)

cfg["training"]["total_epochs"] = 3000
cfg["training"]["eval_deterministic"] = False
cfg["training"]["eval_frequency"] = 50
cfg["expert"]["min_success_rate"] = 0.95
cfg["expert"]["stop_when_ready"] = True

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print(f"Wrote generated expert config: {out}")
PY

echo "=== Exp 0 RSTR: expert training (GELU, stochastic eval, 3000 epoch cap) ==="
for S in "${TV2_SEEDS[@]}"; do
  srun --gres=gpu:1 python -m src.main --config "${GENERATED_CONFIG}" --seed "$S"
done

echo "=== Exp 0 RSTR: build Z_ref (weights_expert.pt, else weights_final.pt) ==="
for S in "${TV2_SEEDS[@]}"; do
  run_dir="${TV2_LOG}/expert_rstr_minigrid_seed${S}"
  if [[ -f "${run_dir}/weights_expert.pt" ]]; then
    validate_weights="${run_dir}/weights_expert.pt"
  elif [[ -f "${run_dir}/weights_final.pt" ]]; then
    validate_weights="${run_dir}/weights_final.pt"
    echo "WARNING [exp0 RSTR]: seed ${S} — no weights_expert.pt; build/validate use weights_final.pt"
  else
    echo "Missing weights under ${run_dir}"
    exit 1
  fi

  python -m src.theory_validation.build_z_ref \
    --config "${GENERATED_CONFIG}" \
    --weights "$run_dir" \
    --output "$TV2_ZREF/rstr_seed${S}.pt" \
    --seed "$S" \
    --n-episodes 500

  python -m src.theory_validation.validate_z_ref \
    --config "${GENERATED_CONFIG}" \
    --expert-weights "$validate_weights" \
    --z-ref "$TV2_ZREF/rstr_seed${S}.pt" \
    --output "$TV2_ZREF/exp0_validation_rstr_seed${S}.json"
done

echo "Exp 0 RSTR complete."
