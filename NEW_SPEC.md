# Spec: action-conditioned control abstraction

This replaces the old KL-to-causal-error bounding-chain spec as the central
claim. Historical Hessian / Fisher / occupancy machinery still exists under
`src/metrics/` and is not the v1 training objective.

## Claim (v1)

On the reachable support of the training environment, with persistent action
coverage, the representation is an **approximate action-conditioned control
abstraction**. The controlled test outcome is future **reward**. Selected
predictive features are deferred.

## What trains what

- PPO actor: on-policy, clipped, optional `target_kl`.
- Value \(V\): on-policy GAE (not replaced by Q advantages in v1).
- PL hinge: frozen reference batch; value-landscape admissibility (anti-aliased PPO).
- Q ensemble: \(Q^{\pi_{\rm ctrl}}\) from replay, one-step Huber TD.
- Action-conditioned MICo: same-action replay pairs; sampled independent-coupling
  surrogate, not Wasserstein bisimulation.
- Query \(U\): stop-gradient information-gain surrogate inside a coverage mixture.
- Relational \(\widehat D_{Z,\infty}^B\): finite-batch certificate, not a population supremum.

## Environments

v1: discrete MiniGrid and synthetic toys. Procgen after the toy package.
Continuous DMControl is out of v1 (discrete Q heads).

## Falsifiers

See [`EXPERIMENTS.md`](EXPERIMENTS.md) E0–E7 and
[`docs/experiment_protocol.md`](docs/experiment_protocol.md).
