# Shared concurrency pool for performance-suite training jobs.
#
# CPU is the bottleneck. Each training process now vectorizes its rollout over
# ENVS_PER_PROC environments (DMControl: that many subprocess MuJoCo workers, one
# per core; Procgen: native in-C batching that costs ~1 core), so a process burns
# roughly ENVS_PER_PROC cores. The pool runs the full TASKS x SEEDS worklist,
# round-robin across the allocated GPUs, capping concurrency so the vectorized
# workers fit the core budget. Packing several processes per GPU also fills VRAM.
#
# Caller sets before sourcing + calling run_training_pool:
#   SUITE            e.g. dmcontrol_state | procgen_easy
#   RESULTS_SUBDIR   results/<subdir>/<exp>/seed_N/<task>/  (usually == SUITE)
#   EXP_NAME         experiment name
#   TASKS            bash array of tasks
#   SEEDS            bash array of seeds
#   EXTRA_ARGS       bash array of extra python args (e.g. --agent ppo)
# Optional env overrides:
#   ENVS_PER_PROC    cores each process consumes for its vectorized rollout;
#                    keep in sync with the suite's num_envs (default 1)
#   THREADS_PER_PROC intra-process math threads (default 1)
#   MAX_PARALLEL     max concurrent processes
#                    (default CPUS / (ENVS_PER_PROC * THREADS_PER_PROC))

run_training_pool() {
  IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
  local num_gpus=${#GPU_IDS[@]}
  local threads=${THREADS_PER_PROC:-1}
  local envs_per_proc=${ENVS_PER_PROC:-1}
  export OMP_NUM_THREADS=${threads}
  export MKL_NUM_THREADS=${threads}

  local max_parallel=${MAX_PARALLEL:-$(( ${SLURM_CPUS_PER_TASK:-32} / (envs_per_proc * threads) ))}
  [ "${max_parallel}" -lt 1 ] && max_parallel=1

  echo "Pool: gpus=${num_gpus} (${CUDA_VISIBLE_DEVICES:-none}) envs/proc=${envs_per_proc} threads/proc=${threads} max_parallel=${max_parallel}"

  local rc=0
  local launched=0
  for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      local ckpt="results/${RESULTS_SUBDIR}/${EXP_NAME}/seed_${SEED}/${TASK}/weights_final.pt"
      if [ -f "${ckpt}" ]; then
        echo "Skipping ${TASK} seed ${SEED} — already finished: ${ckpt}"
        continue
      fi

      # Throttle to max_parallel concurrent processes.
      while [ "$(jobs -rp | wc -l)" -ge "${max_parallel}" ]; do
        wait -n || rc=1
      done

      local gpu=${GPU_IDS[$(( launched % num_gpus ))]}
      launched=$(( launched + 1 ))
      echo "Launch #${launched}: ${TASK} seed ${SEED} -> GPU ${gpu}"
      CUDA_VISIBLE_DEVICES="${gpu}" python -m src.experiments.run_performance_train \
        --suite "${SUITE}" \
        --task "${TASK}" \
        --seed "${SEED}" \
        --exp-name "${EXP_NAME}" \
        --device cuda \
        "${EXTRA_ARGS[@]}" \
        > "results/slurm/${EXP_NAME}_${TASK}_seed${SEED}.${SLURM_JOB_ID:-local}.log" 2>&1 &
    done
  done

  # Drain remaining processes.
  while [ -n "$(jobs -rp)" ]; do
    wait -n || rc=1
  done

  echo "Pool finished: launched=${launched} rc=${rc}"
  return ${rc}
}
