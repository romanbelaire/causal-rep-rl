# Action-conditioned MICo

For pairs \((i,j)\) with \(a_i=a_j=a\):

\[
t_{ij}=|r_i-r_j|+\gamma(1-d_i)(1-d_j)U^-(z_i^+,z_j^+),
\]

where \(d\) is true termination (not truncation) and \(U^-\) is a frozen
target pair metric (angular-diffuse by default; Euclidean is allowed).

Unequal-action pairs are refused. This is **not** exact Wasserstein
bisimulation. Random batch permutations (legacy `src/losses/mico.py`) average
over actions and are not this estimator.

Pair mining assigns positive weight only if an upper confidence bound on a
frozen discrepancy is below \(\tau_+\). Nonfinite \(\hat\sigma_{ij}\) cannot
produce a positive pair. Warmup leaves pairs undecided. A false positive can
permanently alias control-distinct states.

The logged pair score is `D_hat_pair` (the MICo-style target discrepancy). The
conservative bound `D_hat_lower = D_hat_pair - c σ` is used only to label
negatives and, when enabled, to drive `L_sep`. Do not call either object a
metric. See `docs/reward_test_pseudometric.md`.

