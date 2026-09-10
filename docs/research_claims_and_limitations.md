# Research claims and limitations

Every report from this codebase must state:

1. Action-conditioned MICo is a sampled surrogate, not exact bisimulation.
2. v1 uses future rewards; selected predictive features are deferred.
3. The target is reachable-support control sufficiency, not causal-factor recovery.
4. Action randomization supplies action interventions but not external-shift invariance.
5. Batch \(\widehat D_{Z,\infty}^B\) is not population \(D_{Z,\infty}\).
6. KL has latent null directions; PL is a critic-landscape property.
7. No sup-norm critic-drift result establishes moving-critic PL stability.

The live PL hinge is an admissibility condition for anti-aliased PPO. It is not
an anti-aliasing theorem for pairs and it is not optional in the default
active-CTRO loss.

8. `D_hat_pair` (the same-action AC-MICo/Q score) is a learning surrogate. It
   is not automatically symmetric or triangular. The empirical pseudometric is
   `D_hat_sig` on a fixed signature map; v1 uses it as a diagnostic only.
9. `D_hat_lower` is a conservative pair margin for `L_sep`, not a metric.
10. Batch \(\widehat D_{Z,\infty}^B\) cannot create missing separation; an all-excluded
    reference set (`dz_rel_all_excluded=1`) is undefined, not a pass.

