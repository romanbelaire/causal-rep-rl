#!/bin/bash
# Recover after DependencyNeverSatisfied: DMControl NOLINK eval failed on Linear
# log_std checkpoints. Procgen side already trained+eval'd.
#
# 1) Retrain remaining Linear nolink pairs
# 2) Eval nolink
# 3) Aggregate all comparison tables
# 4) Plot panels
set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs
JOBS=src/experiments/jobs

echo "Removing Linear log_std weights_final so trains re-run..."
export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
python - <<'PY'
from pathlib import Path
import torch
root = Path("results/dmcontrol_state/exp_latent_nolink")
for p in root.rglob("weights_final.pt"):
    pol = torch.load(p, map_location="cpu", weights_only=False)["policy"]
    if "action_log_std.weight" in pol or "action_log_std.bias" in pol:
        print("rm", p)
        p.unlink()
        latest = p.with_name("weights_latest.pt")
        if latest.exists():
            latest.unlink()
PY

TRAIN_NL=$(sbatch --parsable "${JOBS}/perf_train_dmcontrol_latent_nolink_finish_s.sh")
echo "DMControl latent_nolink retrain: ${TRAIN_NL}"

WIPE_NL=$(sbatch --parsable --dependency=afterok:"${TRAIN_NL}" \
  "${JOBS}/perf_wipe_neg_control_evals_s.sh")
echo "Wipe evals: ${WIPE_NL}"

EVAL_NL=$(sbatch --parsable --dependency=afterok:"${WIPE_NL}" \
  --export=EXP_NAME=exp_latent_nolink \
  --job-name=nolink-perf-dmcontrol \
  "${JOBS}/perf_eval_dmcontrol_s.sh")
echo "DMControl latent_nolink eval: ${EVAL_NL}"

# Procgen aggs ready now (eval already on disk). DMControl aggs after EVAL_NL.
AGG_JOBS=()
for EXP in exp_baseline exp_latent_nolink exp_ctro_cnn; do
  AGG=$(sbatch --parsable \
    --export=EXP_NAME="${EXP}" \
    "${JOBS}/perf_eval_agg_procgen_s.sh")
  echo "Agg procgen ${EXP}: ${AGG}"
  AGG_JOBS+=("${AGG}")
done

for EXP in exp_baseline exp_latent_nolink exp_ctro_mlp_v2; do
  AGG=$(sbatch --parsable --dependency=afterok:"${EVAL_NL}" \
    --export=EXP_NAME="${EXP}" \
    "${JOBS}/perf_eval_agg_dmcontrol_s.sh")
  echo "Agg dmcontrol ${EXP}: ${AGG}"
  AGG_JOBS+=("${AGG}")
done

# Plots need complete train metrics for re-trained nolink tasks.
PLOT_DEPS=$(IFS=:; echo "${AGG_JOBS[*]}")
PLOT=$(sbatch --parsable --dependency=afterok:"${PLOT_DEPS}" \
  "${JOBS}/perf_plot_neg_controls_s.sh")
echo "Neg-control panels: ${PLOT}"

echo
echo "Recovery submitted."
echo "  train: ${TRAIN_NL}"
echo "  eval:  ${EVAL_NL}"
echo "  plot:  ${PLOT}"
