#!/bin/bash

# Run all experiments

# Experiment 1
sbatch exp1_vanilla_ppo_impala_mlp.sh

# Experiment 2
sbatch exp2_repr_ppo_impala_vae.sh

# Experiment 3
sbatch exp3_rstr_impala_icnn.sh

# Experiment 4
sbatch exp4_rstr_impala_vae_strict.sh