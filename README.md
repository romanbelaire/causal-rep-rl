# Causal-rep-rl: active self-supervised CTRO

Learn an **approximate action-conditioned control abstraction** on the reachable
support of a discrete-action training environment. The agent collects action
experiments, trains an action-value ensemble from replay, uses same-action
MICo-style targets to test latent mergers, and keeps a Polyak–Łojasiewicz (PL)
hinge as a separate value-landscape admissibility condition.

Do not read this as recovery of a unique causal graph, individual
counterfactuals, or robustness to arbitrary external shifts.

## Live recipes

| Preset | Agent | What it is |
|---|---|---|
| vanilla PPO (E0 lock) | `PPO` | Clipped PPO, no PL, no MICo, no replay |
| `anti_aliased_ppo` (E0 lock) | `CTRO` | PPO + scale-corrected PL hinge on a frozen probe batch |
| `active_ctro` | `ActiveCTRO` | Dual-stream collection, replay, Q ensemble, action-conditioned MICo, optional query `U`, relational geometry |
| `ctro_full_legacy` | `CTRO` | Recoverable rejected stack (random-pair MICo + PL) |

PL and action-conditioned MICo share the encoder and are **not** substitutes.
PL is the anti-aliased-PPO admissibility condition on \(V(Z)\). Action-conditioned
MICo is a sampled same-action reward-test surrogate, not exact bisimulation.

## Setup

Bridges-2: load the project env, do not install torch on `/jet/home`.

```bash
module load pytorch/26.05-2.11-py3
source /ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin/activate
cd /jet/home/rbelaire/causal-rep-rl
```

## First commands

Vanilla PPO E0 (MiniGrid Unlock):

```bash
python -m src.experiments.exp_e0_vanilla --seed 42 --device cpu
```

Anti-aliased PPO E0:

```bash
python -m src.experiments.exp_e0_aa_ppo --seed 42 --device cpu
```

Active CTRO on MiniGrid (new architecture; GPU node for a full run):

```bash
python -m src.experiments.exp_active_ctro --seed 42
```

CPU toys (action alias, nuisance, coverage, KL-null):

```bash
CUDA_VISIBLE_DEVICES= python -m src.experiments.run_active_ctro_toys
```

Bridges-2 pipeline (toys → MiniGrid GPU → aggregate):

```bash
bash src/experiments/jobs/submit_active_ctro.sh
```

DeepMind Control / Procgen performance training is unchanged:

```bash
python -m src.experiments.run_performance_train \
  --suite dmcontrol_state --task cartpole-swingup --seed 42 \
  --exp-name exp_anti_aliased_ppo --agent ctro \
  --algo-preset anti_aliased_ppo
```

## Documentation

- [`docs/active_control_abstraction.md`](docs/active_control_abstraction.md) — v1 claim and role split
- [`docs/replay_and_offpolicy_semantics.md`](docs/replay_and_offpolicy_semantics.md)
- [`docs/action_conditioned_mico.md`](docs/action_conditioned_mico.md)
- [`docs/relational_trust_region.md`](docs/relational_trust_region.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/research_claims_and_limitations.md`](docs/research_claims_and_limitations.md)
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — staged sequence
- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md)
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — E0–E7
- [`docs/METHOD_CHANGELOG.md`](docs/METHOD_CHANGELOG.md) — recoverability of prior recipes
- [`NOTES.md`](NOTES.md) — experiment log

## Claims that this repo must not make

1. Action-conditioned MICo is a sampled surrogate, not exact bisimulation.
2. v1 uses future rewards; selected predictive features are deferred.
3. The target is reachable-support control sufficiency, not causal-factor recovery.
4. Action randomization supplies action interventions, not external-shift invariance.
5. Batch \(\widehat D_{Z,\infty}^B\) is not population \(D_{Z,\infty}\).
6. Policy KL has latent null directions; PL is a critic-landscape property, not pairwise anti-aliasing.
7. No sup-norm critic-drift result establishes moving-critic PL stability.
