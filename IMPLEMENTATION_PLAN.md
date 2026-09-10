# Implementation plan: active self-supervised CTRO

Replaces the old bounding-chain / TRPO plan. The sequence is the v1 architecture
in `updated_implementation.md`, with one correction: the Polyak–Łojasiewicz (PL)
hinge stays in the live representation loss. It is not demoted to an incidental
diagnostic. Action-conditioned MICo is a separate same-action reward-test.

## Stages

0. Documentation, dual E0 config lock (vanilla PPO and anti-aliased PPO).
1. Split actor vs representation Adam; snapshot/restore encoder and representation optimizer; E0 regression with new modules off.
2. Transition replay; MiniGrid PPO stream vs query stream; terminated vs truncated in GAE.
3. Discrete Q ensemble, target copies, one-step Huber TD, disagreement metrics. Keep \(V\) for PPO and PL.
4. Same-action replay MICo; action-alias toy (E1).
5. Pair index with UCB/LCB weights; nuisance toy (E4). Undecided warmup.
6. Priority mixture plus coverage mix. Importance-sampling correction is not a behavior-policy ratio.
7. Information critic \(U\) and query mixture with uniform floor (E3).
8. Finite-reference relational distortion: observe mode, then hard gate (E5).
9. E2 calibration, E6 MiniGrid ablations vs both E0 locks. Procgen only after toys succeed. Continuous DMControl is out of v1.

## Non-goals

No model-based planning, no off-policy PPO from replay, no pixel-future targets,
no causal-graph recovery claim, no claim that PL is pairwise anti-aliasing or that
AC-MICo is exact bisimulation.
