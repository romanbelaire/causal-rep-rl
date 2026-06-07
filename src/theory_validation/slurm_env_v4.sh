# Shared SLURM/GPU setup for theory_validation_v4 (source from run_tv4_*.sh).

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

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb=32"
unset PYTORCH_ALLOC_CONF

nvidia-smi
nvcc --version

TV4_LOG=outputs/theory_validation_v4
TV4_CONFIG=configs/theory_validation_v4
TV4_SEEDS=(42 43 44)

tv4_srun_train() {
  local config="$1"
  local seed="$2"
  srun --gres=gpu:1 python -m src.main --config "$config" --seed "$seed"
}

tv4_require_expert_weights() {
  local family="$1"
  for S in "${TV4_SEEDS[@]}"; do
    local dir="${TV4_LOG}/expert_${family}_minigrid_seed${S}"
    if [[ -f "${dir}/weights_expert.pt" ]]; then
      continue
    fi
    if [[ -f "${dir}/weights_final.pt" ]]; then
      echo "WARNING [preflight]: expert_${family}_minigrid_seed${S} — no weights_expert.pt; using weights_final.pt for Z*"
      continue
    fi
    echo "Missing expert weights in ${dir}"
    echo "Run: sbatch src/theory_validation/run_tv4_exp0_${family}_expert.sh"
    exit 1
  done
}

tv4_train_baseline_if_needed() {
  local config="$1"
  local name="$2"
  for S in "${TV4_SEEDS[@]}"; do
    if [[ ! -f "${TV4_LOG}/${name}_seed${S}_metrics.csv" ]]; then
      for S2 in "${TV4_SEEDS[@]}"; do
        tv4_srun_train "${config}" "$S2"
      done
      return
    fi
  done
  echo "Baseline ${name} metrics already present; skipping training."
}

tv4_mico_exp1_train() {
  tv4_require_expert_weights mico
  for S in "${TV4_SEEDS[@]}"; do
    tv4_srun_train "${TV4_CONFIG}/mico_alpha001_minigrid.json" "$S"
    tv4_srun_train "${TV4_CONFIG}/mico_alpha01_minigrid.json" "$S"
    tv4_srun_train "${TV4_CONFIG}/mico_alpha05_minigrid.json" "$S"
  done
  tv4_train_baseline_if_needed "${TV4_CONFIG}/vae_baseline_minigrid.json" "theory_v4_vae_baseline_mico"
}

tv4_dbc_exp1_train() {
  tv4_require_expert_weights dbc
  for S in "${TV4_SEEDS[@]}"; do
    tv4_srun_train "${TV4_CONFIG}/dbc_alpha001_minigrid.json" "$S"
    tv4_srun_train "${TV4_CONFIG}/dbc_alpha01_minigrid.json" "$S"
    tv4_srun_train "${TV4_CONFIG}/dbc_alpha05_minigrid.json" "$S"
  done
  tv4_train_baseline_if_needed "${TV4_CONFIG}/vae_baseline_dbc_minigrid.json" "theory_v4_vae_baseline_dbc"
}

tv4_generate_expert_config() {
  local family="$1"
  local loss_key="$2"
  local loss_val="$3"
  local extra_json="$4"
  local out="${TV4_LOG}/generated_configs/expert_${family}_minigrid_stochastic_3000.json"
  mkdir -p "$(dirname "${out}")"
  python - <<PY
import json
from pathlib import Path

base = Path("${TV4_CONFIG}/expert_${family}_minigrid.json")
out = Path("${out}")

with base.open() as f:
    cfg = json.load(f)

cfg["training"]["total_epochs"] = 3000
cfg["training"]["eval_deterministic"] = False
cfg["training"]["eval_frequency"] = 50
cfg["expert"]["min_success_rate"] = 0.95
cfg["expert"]["stop_when_ready"] = True
cfg["algorithm"]["${loss_key}"] = ${loss_val}
${extra_json}

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\\n")
print(f"Wrote generated expert config: {out}")
PY
}
