#!/usr/bin/env bash
# RSTR VAE strict warmup400 Minigrid — value head comparison (quadratic on Z, squared norm).
# Quadratic heads: μ_latent_analytic must match μ_latent_autodiff at startup (enforced in main.py).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1

CONFIGS=(
  "configs/exp8_rstr_impala_vae_strict_warmup400_quadratic_latent_minigrid.json"
  "configs/exp8_rstr_impala_vae_strict_warmup400_quadratic_latent_mumin_minigrid.json"
  "configs/exp8_rstr_impala_vae_strict_warmup400_squared_norm_minigrid.json"
)

for cfg in "${CONFIGS[@]}"; do
  echo "================================================================================"
  echo "$(date -Is)  START  ${cfg}"
  echo "================================================================================"
  python -m src.main --config "${cfg}"
  echo "$(date -Is)  DONE   ${cfg}"
done

echo "$(date -Is)  All exp8 warmup value-head configs finished."
