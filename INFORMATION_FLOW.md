# Information flow: PPO stream, query stream, replay, targets, gates

## Two collection streams

Proximal policy optimization stays on-policy. Replay is not used for the PPO
surrogate.

```
pi_ctrl  --> PPO rollouts --> GAE --> PPO actor update (fresh data only)
                 |
                 |  (optional copy into replay with source="ppo")
                 v
beta mix --> query rollouts --> transition replay
                                      |
                                      +--> Q ensemble TD
                                      +--> action-conditioned MICo
                                      +--> pair index
                                      +--> information critic U
```

Query behavior:

\[
\beta(a\mid z)=(1-\epsilon-\eta)\pi_{\rm ctrl}(a\mid z)
+\epsilon\pi_{\rm query}(a\mid z)+\eta\,u(a),\qquad u=1/|A|,\;\eta>0.
\]

An entropy bonus does not replace the uniform floor.

## Phase schedule (active CTRO)

1. Freeze target encoder, target policy, target Q, target \(U\).
2. Collect PPO data and query data.
3. `n_aux_updates` replay updates (Q, AC-MICo, pair confidence, \(U\)).
4. PPO actor update on PPO data. PL hinge on the frozen probe batch. Actor
   gradients into the encoder are off unless `actor_updates_encoder` is true.
5. Relational distortion on frozen reference set \(B\) (observe or hard-gate).
6. Hard-copy targets at the phase boundary.

## Pair index

Replay supplies same-action candidate pairs. Diversity chooses hard-positive
*candidates*. Equivalence is an upper confidence bound on a frozen pair
discrepancy. Nonfinite uncertainty cannot yield a positive pair.

## Gates

- PPO clip and optional `target_kl` protect the **policy**.
- \(\widehat D_{Z,\infty}^B\) protects **relational latent geometry**. They are
  not interchangeable. Policy KL has encoder/head null directions.

## Entry points

- MiniGrid ablation (including E0): `src.experiments.exp_e0_vanilla`,
  `src.experiments.exp_e0_aa_ppo`, `src.experiments.exp_active_ctro`.
- Performance suites: `src.experiments.run_performance_train`.
- There is no `src.main`.
