#!/bin/bash

sbatch src/experiments/jobs/full_a0p01_b0p01_s.sh
sbatch src/experiments/jobs/full_a0p01_b0p1_s.sh
sbatch src/experiments/jobs/full_a0p01_b0p5_s.sh
sbatch src/experiments/jobs/full_a0p1_b0p01_s.sh
sbatch src/experiments/jobs/full_a0p1_b0p1_s.sh
sbatch src/experiments/jobs/full_a0p1_b0p5_s.sh
sbatch src/experiments/jobs/full_a0p5_b0p01_s.sh
sbatch src/experiments/jobs/full_a0p5_b0p1_s.sh
sbatch src/experiments/jobs/full_a0p5_b0p5_s.sh