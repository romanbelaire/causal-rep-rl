# Shared SLURM/GPU setup for theory_validation_v2 (source from run_exp*.sh).
# Matches exp8_warmup_valueheads_minigrid.sh / scripts/run_exp8_*.sh environment.

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

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
unset PYTORCH_ALLOC_CONF

nvidia-smi
nvcc --version

TV2_LOG=outputs/theory_validation_v2
TV2_ZREF=artifacts/theory_validation/z_ref
TV2_SEEDS=(42 43 44)

tv2_srun_train() {
  local config="$1"
  local seed="$2"
  srun --gres=gpu:1 python -m src.main --config "$config" --seed "$seed"
}

tv2_exp0_build_zref() {
  mkdir -p "$TV2_ZREF"
  for S in "${TV2_SEEDS[@]}"; do
    local expert_rstr="${TV2_LOG}/expert_rstr_minigrid_seed${S}/weights_final.pt"
    local expert_vae="${TV2_LOG}/expert_vae_minigrid_seed${S}/weights_final.pt"
    if [[ ! -f "$expert_rstr" ]]; then
      echo "Missing RSTR expert weights: $expert_rstr"
      exit 1
    fi
    if [[ ! -f "$expert_vae" ]]; then
      echo "Missing VAE expert weights: $expert_vae"
      exit 1
    fi
    python -m src.theory_validation.build_z_ref \
      --config configs/theory_validation/expert_rstr_minigrid.json \
      --weights "$expert_rstr" \
      --output "$TV2_ZREF/rstr_seed${S}.pt" \
      --seed "$S" --n-episodes 500
    python -m src.theory_validation.build_z_ref \
      --config configs/theory_validation/expert_vae_minigrid.json \
      --weights "$expert_vae" \
      --output "$TV2_ZREF/vae_seed${S}.pt" \
      --seed "$S" --n-episodes 500
    python -m src.theory_validation.validate_z_ref \
      --config configs/theory_validation/expert_rstr_minigrid.json \
      --expert-weights "$expert_rstr" \
      --z-ref "$TV2_ZREF/rstr_seed${S}.pt" \
      --output "$TV2_ZREF/exp0_validation_rstr_seed${S}.json"
  done
}

tv2_exp12_train() {
  for S in "${TV2_SEEDS[@]}"; do
    tv2_srun_train configs/theory_validation/rstr_lconv_on_minigrid.json "$S"
    tv2_srun_train configs/theory_validation/vae_no_repr_loss_minigrid.json "$S"
  done
}

tv2_exp12_train_if_needed() {
  local rstr="theory_rstr_lconv_on_v2"
  local vae="theory_vae_ppo_no_repr_loss_v2"
  for S in "${TV2_SEEDS[@]}"; do
    if [[ ! -f "${TV2_LOG}/${rstr}_seed${S}_metrics.csv" ]] \
       || [[ ! -f "${TV2_LOG}/${vae}_seed${S}_metrics.csv" ]]; then
      tv2_exp12_train
      return
    fi
  done
  echo "Exp 1–2 metrics already present; skipping training."
}
