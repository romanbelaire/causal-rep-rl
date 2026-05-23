#!/usr/bin/env bash
# Sequential FrozenLake runs for VAE+representation-loss (strict family) hyperparameter variants.
# Intended for Slurm (see exp7_vae_strict_hyperparam_sweep.sh) or interactive GPU nodes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

CONFIGS=(
  "configs/exp4_rstr_impala_vae_strict_soft.json"
  "configs/exp4_rstr_impala_vae_strict_warmup400.json"
  "configs/exp4_rstr_impala_vae_strict_reprmatch.json"
)

for cfg in "${CONFIGS[@]}"; do
  echo "================================================================================"
  echo "$(date -Is)  START  ${cfg}"
  echo "================================================================================"
  python -m src.main --config "${cfg}"
  echo "$(date -Is)  DONE   ${cfg}"
done

echo "$(date -Is)  All sweep configs finished."
