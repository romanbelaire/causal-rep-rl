# Replay and off-policy semantics

## Schema

Each stored transition has: `obs`, `next_obs`, `action`, `reward`,
`terminated`, `truncated`, `episode_id`, `step_id`, `behavior_log_prob`,
`control_log_prob`, `source` (`ppo` or `query`), `policy_version`,
`encoder_version`.

Termination and truncation are never merged. Episode order is preserved for a
later multistep algorithm. v1 TD is one-step.

## Two corrections (not interchangeable)

1. **Replay-sampling** importance weights \((N p_i)^{-\alpha_{\rm is}}\).
2. **Behavior-to-target policy** ratios (Retrace / V-trace). Not implemented in
   v1. Do not use (1) as a substitute for (2).

## Coverage

Priority cannot create support that behavior never visited. Sampling mixes a
priority distribution with a coverage distribution \(\beta_{\rm cov}>0\). Query
behavior includes a uniform floor \(\eta/|A|\).
