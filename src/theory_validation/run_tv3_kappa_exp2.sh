#!/bin/bash
# tv3 kappa arm — exp2: checkpoint ablations (4 conditions × 3 seeds)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv3-kappa-exp2.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv3-kappa-exp2

set -euo pipefail
source src/theory_validation/slurm_env_v3.sh

echo "=== tv3 kappa exp2: training ==="
tv3_require_expert_weights rstr
for S in "${TV3_SEEDS[@]}"; do
  tv3_srun_train configs/theory_validation_v3/exp3_kappa_all_off.json "$S"
  tv3_srun_train configs/theory_validation_v3/exp3_kappa_only.json "$S"
  tv3_srun_train configs/theory_validation_v3/exp3_lconv_only.json "$S"
  tv3_srun_train configs/theory_validation_v3/exp3_kappa_lconv.json "$S"
done

echo "=== tv3 kappa exp2: analysis + plots ==="
python -m src.theory_validation.analyze_checkpoints \
  --log-dir outputs/theory_validation_v3 \
  --on-prefix theory_v3_exp3_kappa_all_off \
  --off-prefix theory_v3_exp3_kappa_lconv \
  --output outputs/theory_validation_v3/tv3_exp2_bootstrap_all_off_vs_kappa_lconv.csv
python -m src.theory_validation.analyze_checkpoints \
  --log-dir outputs/theory_validation_v3 \
  --on-prefix theory_v3_exp3_kappa_lconv \
  --off-prefix theory_v3_exp3_lconv_only \
  --output outputs/theory_validation_v3/tv3_exp2_bootstrap_kappa_lconv_vs_lconv.csv
python -m src.theory_validation.plot_validation_v3 --arm kappa --exp 2
python -m src.theory_validation.plot_validation_v3 --arm kappa --exp training_loss
echo "tv3 kappa exp2 complete."
