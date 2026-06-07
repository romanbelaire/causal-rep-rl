#!/bin/bash
# Theory validation v2 — Build Z_ref tables for all existing expert runs (no training).
#   - Scans expert_rstr_minigrid_seed* and expert_vae_minigrid_seed* under TV2_LOG
#   - Checkpoint: weights_expert.pt if present, else weights_final.pt (with warning)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp0-build-zref.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp0-build-zref

set -euo pipefail

REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "${REPO}" || exit 1

module purge
module load Python/3.10.16-GCCcore-13.3.0
source rl-venv/bin/activate

TV2_LOG=outputs/theory_validation_v2
TV2_ZREF=artifacts/theory_validation/z_ref
N_EPISODES=500

CFG_RSTR="${TV2_LOG}/generated_configs/expert_rstr_minigrid_stochastic_3000.json"
CFG_VAE="${TV2_LOG}/generated_configs/expert_vae_minigrid_stochastic_3000.json"
BASE_RSTR=configs/theory_validation/expert_rstr_minigrid.json
BASE_VAE=configs/theory_validation/expert_vae_minigrid.json

mkdir -p "${TV2_ZREF}" "$(dirname "${CFG_RSTR}")"

_write_generated_configs() {
  python - <<'PY'
import json
from pathlib import Path

def write(base_path: Path, out_path: Path) -> None:
    with base_path.open() as f:
        cfg = json.load(f)
    cfg["training"]["total_epochs"] = 3000
    cfg["training"]["eval_deterministic"] = False
    cfg["training"]["eval_frequency"] = 50
    cfg["expert"]["min_success_rate"] = 0.95
    cfg["expert"]["stop_when_ready"] = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")

write(
    Path("configs/theory_validation/expert_rstr_minigrid.json"),
    Path("outputs/theory_validation_v2/generated_configs/expert_rstr_minigrid_stochastic_3000.json"),
)
write(
    Path("configs/theory_validation/expert_vae_minigrid.json"),
    Path("outputs/theory_validation_v2/generated_configs/expert_vae_minigrid_stochastic_3000.json"),
)
PY
}

_pick_validate_weights() {
  local run_dir="$1"
  if [[ -f "${run_dir}/weights_expert.pt" ]]; then
    echo "${run_dir}/weights_expert.pt"
    return 0
  fi
  if [[ -f "${run_dir}/weights_final.pt" ]]; then
    echo "WARNING [build_zref]: ${run_dir} — no weights_expert.pt; using weights_final.pt" >&2
    echo "${run_dir}/weights_final.pt"
    return 0
  fi
  return 1
}

_build_family_seed() {
  local family="$1"
  local seed="$2"
  local config="$3"
  local run_dir="${TV2_LOG}/expert_${family}_minigrid_seed${seed}"

  if [[ ! -d "$run_dir" ]]; then
    echo "SKIP ${family} seed ${seed}: no run dir ${run_dir}"
    return 0
  fi

  local validate_weights
  if ! validate_weights="$(_pick_validate_weights "$run_dir")"; then
    echo "SKIP ${family} seed ${seed}: no weights_expert.pt or weights_final.pt under ${run_dir}"
    return 0
  fi

  echo "=== build_z_ref: ${family} seed ${seed} (${validate_weights}) ==="
  python -m src.theory_validation.build_z_ref \
    --config "$config" \
    --weights "$run_dir" \
    --output "${TV2_ZREF}/${family}_seed${seed}.pt" \
    --seed "$seed" \
    --n-episodes "$N_EPISODES"

  python -m src.theory_validation.validate_z_ref \
    --config "$config" \
    --expert-weights "$validate_weights" \
    --z-ref "${TV2_ZREF}/${family}_seed${seed}.pt" \
    --output "${TV2_ZREF}/exp0_validation_${family}_seed${seed}.json" \
    --seed "$seed" \
    --n-episodes 50
}

_discover_seeds() {
  local family="$1"
  local -n _out="$2"
  _out=()
  local d base seed
  for d in "${TV2_LOG}/expert_${family}_minigrid_seed"*/; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d")
    seed="${base#expert_${family}_minigrid_seed}"
    [[ "$seed" =~ ^[0-9]+$ ]] || continue
    if [[ -f "${d}/weights_expert.pt" || -f "${d}/weights_final.pt" ]]; then
      _out+=("$seed")
    fi
  done
  if [[ ${#_out[@]} -eq 0 ]]; then
    return 1
  fi
  # unique sorted
  readarray -t _out < <(printf '%s\n' "${_out[@]}" | sort -nu)
}

_write_generated_configs

[[ -f "$CFG_RSTR" ]] || { echo "Missing ${CFG_RSTR}"; exit 1; }
[[ -f "$CFG_VAE" ]] || { echo "Missing ${CFG_VAE}"; exit 1; }

BUILT=0
for family in rstr vae; do
  if [[ "$family" == "rstr" ]]; then
    cfg="$CFG_RSTR"
  else
    cfg="$CFG_VAE"
  fi
  seeds=()
  if ! _discover_seeds "$family" seeds; then
    echo "No expert_${family}_minigrid_seed* runs with weights under ${TV2_LOG}"
    continue
  fi
  echo "Found ${family} seeds: ${seeds[*]}"
  for S in "${seeds[@]}"; do
    _build_family_seed "$family" "$S" "$cfg"
    BUILT=$((BUILT + 1))
  done
done

if [[ "$BUILT" -eq 0 ]]; then
  echo "No Z_ref tables built — train experts with run_exp0.sh / run_exp0_rstr.sh first."
  exit 1
fi

echo "Exp 0 build_zref complete (${BUILT} tables)."
