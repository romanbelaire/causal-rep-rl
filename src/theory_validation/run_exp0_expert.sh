#!/bin/bash
# Deprecated monolithic launcher — use submit_exp0_experts.sh instead.
set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"
echo "Use: bash src/theory_validation/submit_exp0_experts.sh"
exec bash src/theory_validation/submit_exp0_experts.sh
