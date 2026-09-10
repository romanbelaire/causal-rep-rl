# Active Self-Supervised CTRO: Implementation Guide

**Audience:** Cursor agent implementing the next architecture in `romanbelaire/causal-rep-rl`.

**Status:** implementation handoff. This document separates theorem-backed targets, empirical surrogates, and claims that must *not* be made.

## 1. Goal

The existing codebase combines PPO, a VAE/value critic, random-pair MICo, and optional PL coupling. The next architecture should instead learn a **self-supervised action-conditioned control abstraction**:

> Actively collect action experiments; train an action-value critic from replay; use action-conditioned MICo-style targets to test candidate latent mergers; and preserve learned relational geometry across updates.

The intended v1 claim is deliberately narrow:

> On the reachable support of the training environment, with persistent action coverage, the representation is an approximate action-conditioned control abstraction.

Do not claim recovery of a unique causal graph, individual counterfactuals, or robustness to arbitrary external shifts.

### Fixed v1 decisions

1. Start in discrete-action MiniGrid; preserve extensibility to continuous actions.
2. Use future **rewards** as the controlled test outcome. Defer selected predictive features.
3. Use action-conditioned MICo as the empirical, bootstrapped approximation—not exact bisimulation.
4. Keep PPO actor learning on-policy. Replay trains auxiliary critics and representation losses.
5. Add target Q critics before attempting adaptive critic schedules.
6. Keep PL only as an optional critic-conditioning diagnostic; it is not an anti-aliasing theorem.

## 2. Current-state assessment

### Reusable pieces

| Path | Current role | Use in migration |
|---|---|---|
| `src/agents/ppo.py` | On-policy PPO / GAE / latent policy input | Preserve for return policy learning; refactor optimizer boundaries. |
| `src/agents/ctro.py` | PPO + MICo + PL; EMA encoder target | Reuse target/checkpoint patterns. |
| `src/losses/mico.py` | Random-pair MICo | Replace with action-conditioned replay-pair loss. |
| `src/utils/bisimulation_utils.py` | Encoder-target helpers | Generalize for target encoder. |
| `src/experiments/runner.py` | MiniGrid collection and training harness | Split into PPO collection, query collection, replay updates, evaluation. |
| `src/architectures/forward_model.py` / `latent_dynamics.py` | Currently unused models | Do not use for planning in v1; reserve for later predictive-feature work. |

### Must change

1. `compute_mico_loss` randomly permutes transitions and ignores their actions. It therefore gives a policy-averaged signal, not action-conditioned equivalence.
2. Rollout data are discarded each epoch: there is no persistent replay, Q critic, target critic, or ensemble.
3. The target encoder exists, but there is no target critic for stable action-value uncertainty.
4. The current documentation's pointwise `E||Z_new-Z_old||^2` is weaker than relational distortion and no relational gate exists in `src/`.
5. The VAE reconstruction loss may retain visual nuisances. It must be optional.
6. The unified optimizer makes a hard encoder-update rollback unsafe. Split actor and representation/critic optimizer states first.

## 3. Target architecture

```text
  fresh PPO rollouts --> PPO actor update --> PPO clip / optional target KL
          |
          |                       replay ----------------> Q ensemble
  pi_ctrl + query mixture --------------------+               |
          |                                  pair index        +--> target Q ensemble
          v                                      |              |
     environment action experiments               +--> AC-MICo --+--> information critic U
                                                                  |
  online encoder -------------------------------> target encoder |
          |                                                       |
          +---- frozen finite reference batch B --> relational D_Z,infinity^B gate
```

Suggested module layout:

```text
src/
  agents/active_ctro.py
  replay/transition_replay.py
  replay/priorities.py
  replay/pair_index.py
  architectures/critics/q_ensemble.py
  architectures/critics/information_critic.py
  losses/action_conditioned_mico.py
  losses/q_td.py
  losses/information_value.py
  metrics/relational_trust_region.py
  metrics/action_coverage.py
  metrics/pair_equivalence.py
  utils/target_network.py
```

Keep components composable; do not begin with a broad rewrite of PPO.

## 4. Data collection and replay

### 4.1 Do not make PPO off-policy by accident

PPO remains on-policy. Use two streams initially:

1. **PPO stream:** fresh trajectories under the saved PPO behavior policy; use only for GAE and PPO actor updates.
2. **Query stream:** exploratory trajectories stored in replay; use replay for Q, MICo, pair selection, and `U`.

The query behavior is

\[
\beta(a\mid z)=(1-\epsilon-\eta)\pi_{\rm ctrl}(a\mid z)
 +\epsilon\pi_{\rm query}(a\mid z)+\eta u(a\mid z),\qquad\eta>0.
\]

For discrete actions, `u(a|z)=1/|A|`, hence

\[
\beta(a\mid z)\ge \eta/|A|.
\]

An entropy bonus does not replace this coverage floor.

### 4.2 Replay schema

```python
Transition(
    obs, next_obs, action, reward,
    terminated, truncated,
    episode_id, step_id,
    behavior_log_prob,     # log beta(a|z) at collection
    control_log_prob,      # log pi_ctrl(a|z), when defined
    source,                # "ppo" or "query"
    policy_version,
    encoder_version,
)
```

Preserve episode order for later multistep targets. Never conflate termination and truncation.

### 4.3 Replay priorities

Use a floor plus explicit coverage mixture:

\[
p_i=\epsilon_p+\lambda_{\rm IG}I_i+\lambda_{\rm gap}G_i
 +\lambda_{\rm pair}P_i+\lambda_{\rm geom}R_i,
\]

\[
q_{\rm replay}=(1-\beta_{\rm cov})q_{\rm priority}+\beta_{\rm cov}q_{\rm cover},
\qquad \beta_{\rm cov}>0.
\]

- `I_i`: ensemble Bellman-target disagreement.
- `G_i`: action-gap uncertainty.
- `P_i`: uncertainty in a candidate action-conditioned pair equivalence.
- `R_i`: geometry-risk / near-alias involvement.
- `q_cover`: reservoir, uniform, or state-stratified sampling.

Log each component separately. Priority cannot create support that behavior never visited.

Keep two corrections distinct:

- Replay-sampling correction, e.g. `(N p_i)^(-alpha_is)`.
- Behavior-to-target policy correction for multistep off-policy evaluation, e.g. Retrace/V-trace ratios.

They are not interchangeable.

## 5. Q ensemble and targets

### 5.1 Q semantics

Train `Q^{pi_ctrl}`, not `Q*`, in the initial PPO architecture. PPO needs values relative to the current return policy.

For a discrete action head `Q_j(z)`, define

\[
V_Q(z)=\sum_a\pi_{\rm ctrl}(a\mid z)Q(z,a),
\qquad A_Q(z,a)=Q(z,a)-V_Q(z).
\]

Use Q advantages first for diagnostics and acquisition. Do not replace PPO GAE with them until an ablation supports doing so.

### 5.2 TD update

Use target encoder `Z^-`, target policy `pi^-`, and a target ensemble:

\[
y_j=r+\gamma(1-\mathbb{1}_{\rm terminated})
\sum_{a'}\pi^-_{\rm ctrl}(a'\mid z^+)\bar Q_j(z^+,a').
\]

Start with one-step Huber TD. Add multistep Retrace only after sequence replay and behavior-ratio tests exist. Use independent ensemble heads and bootstrap masks/resampling, then log disagreement calibration. A target-ensemble uncertainty score is

\[
I_i=\operatorname{Var}_j\left[r_i+\gamma V_{\bar Q_j}(z_i^+)\right].
\]

Raw TD error alone is not epistemic uncertainty.

## 6. Action-conditioned MICo and pair construction

### 6.1 Loss contract

The v1 primary loss must pair transitions with the **same action**. For `(i,j)` with `a_i=a_j=a`, use

\[
t_{ij}=|r_i-r_j|+\gamma(1-d_i)(1-d_j)U^-(z_i^+,z_j^+),
\]

where `U^-` is a frozen target representation-distance surrogate. Terminal handling must be explicit and tested. This is a sampled independent-coupling MICo-style surrogate, not exact Wasserstein bisimulation.

Retain angular/diffuse distance only behind an interface that also permits Euclidean or learned nonnegative pair metrics.

### 6.2 Interpretation

The ideal controlled-test discrepancy is

\[
D_{\mathcal U}(h,h')=\sum_{u\in\mathcal U}w(u)\operatorname{IPM}
\left(P(Y_u\mid h,do(u)),P(Y_u\mid h',do(u))\right),
\]

where `Y_u` is a future reward sequence. Action-conditioned bootstrapped MICo is the empirical v1 approximation. Do not claim equality without separating action tests, persistent action coverage, consistent conditional estimation, and sufficient capacity.

### 6.3 Pair mining

Replace random batch permutations in the primary loss:

1. Select anchor `i` from replay.
2. Select a same-action candidate `j`, preferring observationally or currently-latently diverse candidates.
3. Estimate frozen pair discrepancy and uncertainty from target-ensemble quantities.
4. Assign positive weight only if an **upper confidence bound** is below `tau_pos`.
5. Retain confidently high-discrepancy pairs as negatives/repulsion diagnostics.

\[
w^+_{ij}=\mathbf{1}\{\widehat D_{ij}+c\widehat\sigma_{ij}\le\tau_+\}
w_{\rm diversity}(i,j),
\]

\[
w^-_{ij}=\mathbf{1}\{\widehat D_{ij}-c\widehat\sigma_{ij}\ge\tau_-\}.
\]

Raw diversity chooses hard-positive candidates; it never defines equivalence. During warmup, undecided pairs must remain undecided. A false positive can permanently alias control-distinct states.

## 7. Information-value critic and query behavior

Define a separate `U(z,a)` with a stop-gradient information reward:

\[
i(z,a)=\lambda_Qi_Q(z,a)+\lambda_{\rm gap}i_{\rm gap}(z,a)
       +\lambda_{\rm pair}i_{\rm pair}(z,a).
\]

- `i_Q`: target-ensemble Bellman disagreement.
- `i_gap`: uncertainty in relative action ordering.
- `i_pair`: expected reduction in uncertainty about relevant candidate pairs.

This is an information-gain **surrogate**, not an exact mutual-information estimator. With target objects frozen for a collection phase, train

\[
U(z,a)\leftarrow i(z,a)+\gamma_U(1-d)\max_b\bar U(z^+,b).
\]

Use `pi_query(a|z) proportional to exp(U(z,a)/tau_U)` or epsilon-greedy over U, always inside the behavior mixture and coverage floor from Section 4.

## 8. Relational latent trust region

Replace old average pointwise displacement with finite-reference relational distortion:

\[
\widehat D_{Z,\infty}^{B}=
\max_{i\ne j,\;d_{\rm old}(s_i,s_j)\ge\tau_d}
\left|\frac{d_{\rm new}(s_i,s_j)}{d_{\rm old}(s_i,s_j)}-1\right|.
\]

`B` is a frozen reference set. `tau_d` avoids unstable ratios near old zero-distance pairs; log the excluded-pair fraction. This is a finite-batch certificate only, never a population supremum.

Implementation stages:

1. **Observe mode:** calculate and log after every representation update.
2. **Hard-gate mode:** snapshot encoder parameters and representation optimizer state; apply candidate update; calculate metric; restore both on violation of `epsilon_z`.

This is why optimizer separation is mandatory. Keep PPO clipping and optionally add standard policy `target_kl` early stopping. KL and relational geometry protect different objects.

## 9. Optimizer/update schedule

Replace the unified optimizer with at least:

```text
actor_optimizer:           policy-head parameters
representation_optimizer:  encoder + V/Q/MICo/U parameters
```

Initial schedule:

1. Freeze target networks; collect PPO and query data.
2. Run `n_aux_updates` replay updates: Q ensemble, MICo, pair confidence, U.
3. Run PPO actor update on fresh PPO data only.
4. Initially stop actor gradients into the encoder, or make it a default-off config option.
5. Measure/gate the representation update with relational distortion.
6. Refresh target networks at phase boundary (hard copy first, EMA later as ablation).

## 10. Configuration

Add a new configuration family rather than overloading legacy `alpha` / `beta` settings.

```yaml
algorithm:
  ppo: {clip_epsilon: 0.2, target_kl: null, actor_updates_encoder: false}
  replay: {capacity: 200000, batch_size: 256, coverage_mix: 0.10,
           priority_floor: 0.001, importance_exponent: 0.4}
  q_ensemble: {members: 5, hidden_sizes: [256, 256], target_update: hard,
               target_interval_phases: 1, td_huber_delta: 1.0}
  mico_ac: {coefficient: 1.0, pair_metric: angular_diffuse,
            positive_ucb_threshold: 0.1, negative_lcb_threshold: 0.3}
  information_critic: {gamma_u: 0.95, temperature: 0.25,
                       lambda_q: 1.0, lambda_gap: 0.5, lambda_pair: 1.0}
  exploration: {query_mix_epsilon: 0.15, uniform_floor_eta: 0.05}
  relational_tr: {enabled: false, mode: observe, reference_size: 512,
                  min_old_distance: 1e-3, epsilon_z: 0.10}
  representation: {vae_coef: 0.0, pl_coupling_coef: 0.0}
```

Persist all config values and target/network versions in checkpoints and run metadata.

## 11. Required metrics

### Coverage and replay

- action counts/probabilities by state stratum; entropy and actual uniform-floor mass;
- PPO/query data fraction; replay effective sample size; priority component distributions;
- query-action distribution and information-score calibration.

### Q and pair uncertainty

- TD loss per ensemble member; target-online drift; Bellman-target variance;
- action gaps and action-gap uncertainty;
- same-action pair count by action; positive/negative/undecided fractions;
- pair residual/uncertainty calibration against later observed reward-test disagreement.

### Geometry and policy

- `D_Z,infinity^B`, excluded near-zero fraction, and a quantile distortion summary;
- relational-gate accept/reject rate; latent rank/participation ratio;
- PPO clip fraction, approximate KL, and return.

Do not present policy KL as a latent-geometry certificate, nor PL as a pairwise anti-aliasing certificate.

## 12. Tests

### Unit tests

1. Ring buffer preserves all metadata and episode order.
2. Priority floor and coverage mixture give every stored item nonzero sample probability.
3. Replay-sampling and action-policy corrections are separately computed.
4. Q target bootstraps over truncation but not true termination.
5. Ensemble members receive different bootstrap masks/samples.
6. Action-conditioned MICo refuses/buckets unequal-action pairs.
7. Uninitialized or nonfinite pair uncertainty cannot yield a positive pair.
8. Targets are frozen inside a collection phase.
9. Relational metric never divides by zero and reports excluded pairs.
10. Hard-gate rejection restores encoder *and optimizer* state exactly.

### Synthetic tests

- **Action alias:** equal current greedy action/value, different reward under another action. AC-MICo/Q must separate it.
- **Nuisance:** visually distant states with identical reward futures for all actions. Pair mining must merge them.
- **Coverage:** discriminating action is rare. Demonstrate failure with no floor and recovery with it.
- **KL null direction:** compensating encoder/head map preserves logits while contracting a latent coordinate. KL stays zero; relational metric detects it.
- **Near-zero pairs:** verify stable `tau_d` exclusion.

## 13. Experiment program

All experiments require: claim, baseline, fixed budget, metrics, and falsification criterion.

| ID | Claim / comparison | Primary falsifier |
|---|---|---|
| E0 | Refactor with new features disabled matches current PPO/CTRO behavior. | Baseline regression beyond ordinary seed variation. |
| E1 | Same-action MICo/Q resolves action aliasing better than random action-agnostic MICo. | No gain on action-alias toy. |
| E2 | Replay target ensemble produces better-calibrated uncertainty than fresh-buffer scalar V. | Disagreement does not predict later TD/pair error. |
| E3 | Query U resolves discriminating action outcomes more efficiently than matched uniform/entropy exploration. | No sample-efficiency gain on controlled toys. |
| E4 | Confidence-filtered hard positives merge nuisances without false merges. | Increased false merges or no nuisance-invariance benefit. |
| E5 | Relational metric detects KL-null geometry changes; hard gate retains abstraction diagnostics. | Cannot distinguish analytic null direction from benign head change. |
| E6 | Full ablation improves task return *and* action-conditioned probe error. | Only exploration statistics improve. |
| E7 optional | Held-out named visual/context shift is handled. | No benefit on stated shift; no broad causal claim. |

Run E0--E5 on MiniGrid/toys before Procgen or DMControl. For all benchmark claims, report per-seed results, intervals, wall-clock overhead, action coverage, and every component ablation.

## 14. Documentation changes

Revise existing documentation:

| File | Required update |
|---|---|
| `README.md` | Remove stale planning language; add current architecture, setup, first active-CTRO command. |
| `NEW_SPEC.md` | Replace old KL-to-causal-error chain as the central claim with action-conditioned control abstraction and tests. |
| `representation_space_trust_region.md` | Supersede expected pointwise displacement with finite-reference relational distortion; state its limitations. |
| `IMPLEMENTATION_PLAN.md` | Replace with the staged sequence below. |
| `IMPLEMENTATION_STATUS.md` | Mark every component implemented/tested/experimental/unsupported. |
| `EXPERIMENTS.md` | Add E0--E7, hypotheses, metrics, and failure conditions. |
| `INFORMATION_FLOW.md` | Document PPO data, query data, replay, pair index, targets, and gates. |

Add:

```text
docs/active_control_abstraction.md
docs/replay_and_offpolicy_semantics.md
docs/action_conditioned_mico.md
docs/relational_trust_region.md
docs/experiment_protocol.md
docs/research_claims_and_limitations.md
```

Every document/report must state:

1. AC-MICo is a sampled surrogate, not exact bisimulation.
2. v1 uses future rewards; selected predictive features are deferred.
3. The target is reachable-support control sufficiency, not causal-factor recovery.
4. Action randomization supplies action interventions but not external-shift invariance.
5. Batch `D_Z,infinity^B` is not population `D_Z,infinity`.
6. KL has latent null directions; PL is a critic-landscape property.
7. No sup-norm critic-drift result establishes moving-critic PL stability.

## 15. Implementation order

1. Lock baseline config/results; update stale documentation.
2. Split actor and representation optimizers; prove disabled-feature regression (E0).
3. Implement replay schema, uniform/resevoir sampling, terminal tests, logs.
4. Implement discrete Q ensemble, target copies, one-step TD, uncertainty metrics.
5. Replace random MICo with same-action replay MICo; run E1.
6. Implement pair index, confidence weights, nuisance/action-alias toys.
7. Implement priority mixture and correction tests.
8. Add U and separate query collection; run E3.
9. Add relational distortion in observe mode; validate KL-null toy.
10. Enable relational hard gate only after observe-mode scale is known.
11. Run full ablations and only then benchmark/shift experiments.

## 16. Non-goals

- No model-based planning or learned rollout optimization in v1.
- No transformer-based latent interventions.
- No policy-gradient training from arbitrary replay without a separately justified off-policy algorithm.
- No raw future-pixel targets in v1.
- No external-shift causal robustness claim without named-shift data.
- No theorem claiming the learned critic is `V*` or that PL discovers bisimulation.

## 17. Definition of first success

The first successful result is the following empirical package:

1. On controlled action-alias and nuisance toys, the method separates different action-conditioned reward futures and merges observationally different reward-test-equivalent states.
2. The query policy reaches that discrimination in fewer environment steps than matched uniform/entropy baselines while preserving explicit action support.
3. Ensemble/pair uncertainty predicts later disagreement sufficiently to make confidence filtering useful.
4. The relational metric catches known encoder/head KL-null geometry changes.
5. Benchmark gains survive seed-level component ablation against the current PPO/CTRO baseline.

Only after this package succeeds should the project add selected predictive features or pursue stronger external-intervention claims.
