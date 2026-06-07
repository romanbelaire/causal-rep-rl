# Shared SLURM/GPU setup for theory_validation_v3 (source from run_tv3_*.sh).

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

TV2_LOG=outputs/theory_validation_v2
TV3_LOG=outputs/theory_validation_v3
TV3_SEEDS=(42 43 44)

tv3_srun_train() {
  local config="$1"
  local seed="$2"
  srun --gres=gpu:1 python -m src.main --config "$config" --seed "$seed"
}

tv3_require_expert_weights() {
  local family="$1"
  for S in "${TV3_SEEDS[@]}"; do
    local dir="${TV2_LOG}/expert_${family}_minigrid_seed${S}"
    if [[ -f "${dir}/weights_expert.pt" ]]; then
      continue
    fi
    if [[ -f "${dir}/weights_final.pt" ]]; then
      echo "WARNING [preflight]: expert_${family}_minigrid_seed${S} — no weights_expert.pt; using weights_final.pt for Z*"
      continue
    fi
    echo "Missing expert weights in ${dir}"
    if [[ "$family" == "rstr" ]]; then
      echo "Run: sbatch src/theory_validation/run_exp0_rstr.sh"
    else
      echo "Run: sbatch src/theory_validation/run_exp0.sh"
    fi
    exit 1
  done
}

tv3_train_vae_baseline_if_needed() {
  local name="theory_v3_vae_baseline"
  for S in "${TV3_SEEDS[@]}"; do
    if [[ ! -f "${TV3_LOG}/${name}_seed${S}_metrics.csv" ]]; then
      for S2 in "${TV3_SEEDS[@]}"; do
        tv3_srun_train configs/theory_validation_v3/vae_baseline_minigrid.json "$S2"
      done
      return
    fi
  done
  echo "VAE baseline metrics already present; skipping training."
}

tv3_kappa_exp1_train() {
  tv3_require_expert_weights rstr
  tv3_require_expert_weights vae
  for S in "${TV3_SEEDS[@]}"; do
    tv3_srun_train configs/theory_validation_v3/kappa_dir_rstr_minigrid.json "$S"
  done
  tv3_train_vae_baseline_if_needed
}

tv3_distill_exp1_train() {
  tv3_require_expert_weights rstr
  tv3_require_expert_weights vae
  for S in "${TV3_SEEDS[@]}"; do
    tv3_srun_train configs/theory_validation_v3/z_distill_rstr_minigrid.json "$S"
  done
  tv3_train_vae_baseline_if_needed
}
