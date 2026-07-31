#!/bin/bash
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
sbatch src/experiments/jobs/full_sweep_s.sh
