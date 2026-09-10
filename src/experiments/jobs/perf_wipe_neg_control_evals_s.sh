#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 00:15:00
#SBATCH --ntasks-per-node=1
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=wipe-neg-evals
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --array=0-0
#SBATCH --requeue

# Remove eval CSVs for cells refreshed by the current pipeline stage.
# Default: DMControl latent_nolink only (procgen baseline usually already good).
# Override: WIPE_TARGETS="procgen_baseline dmcontrol_nolink"

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs

WIPE_TARGETS="${WIPE_TARGETS:-dmcontrol_nolink}"

wipe_glob() {
  local pattern=$1
  # shellcheck disable=SC2086
  rm -fv ${pattern}
}

for target in ${WIPE_TARGETS}; do
  case "${target}" in
    procgen_baseline)
      echo "Wiping procgen exp_baseline eval seeds..."
      wipe_glob "results/perf_eval/procgen_easy/exp_baseline/seed_*/performance_eval_metrics.csv"
      wipe_glob "results/perf_eval/procgen_easy/exp_baseline/seed_*/performance_eval_config.json"
      ;;
    dmcontrol_nolink)
      echo "Wiping dmcontrol exp_latent_nolink eval seeds..."
      wipe_glob "results/perf_eval/dmcontrol_state/exp_latent_nolink/seed_*/performance_eval_metrics.csv"
      wipe_glob "results/perf_eval/dmcontrol_state/exp_latent_nolink/seed_*/performance_eval_config.json"
      ;;
    *)
      echo "ERROR: unknown WIPE_TARGETS entry: ${target}"
      exit 1
      ;;
  esac
done

echo "Wipe complete."
