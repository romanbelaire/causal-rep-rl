#!/usr/bin/env bash
# Sequential Minigrid runs for VAE+representation-loss (strict family), exp8 hyperparameter variants.
# Value head is affine in z (value_hidden: []) for trivial convexity in latent space.
# Intended for Slurm (see exp8_vae_strict_minigrid_hyperparam_sweep.sh) or interactive GPU nodes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

CONFIGS=(
  "configs/exp8_rstr_impala_vae_strict_soft_minigrid.json"
  "configs/exp8_rstr_impala_vae_strict_warmup400_minigrid.json"
  "configs/exp8_rstr_impala_vae_strict_reprmatch_minigrid.json"
)

for cfg in "${CONFIGS[@]}"; do
  echo "================================================================================"
  echo "$(date -Is)  START  ${cfg}"
  echo "================================================================================"
  python -m src.main --config "${cfg}"
  echo "$(date -Is)  DONE   ${cfg}"
done

echo "$(date -Is)  All exp8 Minigrid sweep configs finished."
