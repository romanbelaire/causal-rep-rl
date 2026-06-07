#!/bin/bash
# Submit the six remaining tv3 kappa exp2 ablation runs (job 195340 timed out after 6/12).
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"

sbatch src/theory_validation/run_tv3_kappa_exp2_lconv_only_seed43.sh
sbatch src/theory_validation/run_tv3_kappa_exp2_kappa_lconv_seed43.sh
sbatch src/theory_validation/run_tv3_kappa_exp2_all_off_seed44.sh
sbatch src/theory_validation/run_tv3_kappa_exp2_kappa_only_seed44.sh
sbatch src/theory_validation/run_tv3_kappa_exp2_lconv_only_seed44.sh
sbatch src/theory_validation/run_tv3_kappa_exp2_kappa_lconv_seed44.sh

echo "Submitted 6 tv3 kappa exp2 remainder jobs."
echo "After all finish, regenerate exp2 outputs:"
echo "  python -m src.theory_validation.analyze_checkpoints --log-dir outputs/theory_validation_v3 --on-prefix theory_v3_exp3_kappa_all_off --off-prefix theory_v3_exp3_kappa_lconv --output outputs/theory_validation_v3/tv3_exp2_bootstrap_all_off_vs_kappa_lconv.csv"
echo "  python -m src.theory_validation.analyze_checkpoints --log-dir outputs/theory_validation_v3 --on-prefix theory_v3_exp3_kappa_lconv --off-prefix theory_v3_exp3_lconv_only --output outputs/theory_validation_v3/tv3_exp2_bootstrap_kappa_lconv_vs_lconv.csv"
echo "  bash src/theory_validation/plot_all_v3.sh"
