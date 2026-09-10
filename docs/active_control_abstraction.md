# Active control abstraction (v1)

## Claim

On the reachable support of the training environment, with persistent action
coverage, the learned \(Z\) is an approximate **action-conditioned control
abstraction**. The test outcome is future reward.

## Two representation losses

**Polyak–Łojasiewicz hinge (live).** Frozen probe batch of states. One-sided
scale-corrected hinge on \(\tilde\mu_{PL}\). This is the anti-aliased PPO
admissibility condition on the value landscape \(V(Z)\). It is not a pairwise
anti-aliasing theorem and it is not demoted to a no-op.

**Action-conditioned MICo (new).** Same-action replay pairs. Bootstrapped
distance targets using future rewards. This is a sampled independent-coupling
surrogate for a controlled test, not exact bisimulation.

They share the encoder. They answer different questions. Turning one off is an
ablation, not a redefinition of the other.

## PPO stays on-policy

Replay trains Q, AC-MICo, pair selection, and \(U\). The PPO surrogate uses
only fresh \(\pi_{\rm ctrl}\) trajectories.
