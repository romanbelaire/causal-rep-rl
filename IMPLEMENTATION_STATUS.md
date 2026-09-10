# Implementation status

Status values: **implemented**, **tested** (CPU toy or invariant in module),
**experimental** (wired, MiniGrid/GPU evidence not yet the success package),
**unsupported**.

| Component | Status |
|---|---|
| PPO clip, GAE, MiniGrid Unlock, DMControl/Procgen runners | implemented |
| Anti-aliased PPO (PL hinge, frozen ref batch, head phasing) | implemented |
| Legacy random-pair MICo + log-ratio \(D_Z\) (`ctro_full_legacy`) | implemented (recovery only) |
| Split actor / representation optimizers | implemented |
| Encoder + representation-optimizer snapshot/restore | implemented |
| `actor_updates_encoder` default-off on active CTRO | implemented |
| PPO `target_kl` early stop | implemented |
| Transition replay (metadata, episode order, term vs trunc) | implemented |
| Dual PPO / query collection streams | implemented |
| Uniform coverage floor \(\eta/\|A\|\) | implemented |
| Discrete Q ensemble, target copy, one-step Huber TD | implemented |
| Action-conditioned MICo (same-action pairs) | implemented |
| Pair index UCB/LCB, undecided warmup | implemented |
| Priority mixture + coverage mix + IS weights | implemented |
| Behavior-ratio / Retrace multistep | unsupported (deferred) |
| Information critic \(U\) and \(\pi_{\rm query}\) | implemented |
| Relational \(\widehat D_{Z,\infty}^B\) observe mode | implemented |
| Relational hard gate (restore encoder **and** optimizer) | implemented |
| Action-alias / nuisance / coverage / KL-null toys | implemented (CPU) |
| E0 full MiniGrid curve lock vs historical seeds | experimental (configs locked; GPU not required for landing) |
| E6 MiniGrid + Procgen benchmarks | experimental / unsupported until toys succeed |
| Model-based planning, continuous Q, pixel-future targets | unsupported |
| Unique causal graph / external-shift robustness | unsupported |
