# Experimental Program for the Latent Trust Region Algorithm

## Objective

The experimental program is designed to support three claims:

1. **Ordinary-case safety:** the latent trust region algorithm performs at least as well as PPO, within a predeclared practical margin, on environments where PPO works normally.
2. **Collapse-regime advantage:** the algorithm outperforms PPO when aggressive policy optimization causes PPO to lose useful representational capacity.
3. **Distributional robustness:** the algorithm generalizes better than PPO to unseen Procgen levels, with ablations showing whether the gain comes from the causal component, latent stability, or their combination.

For brevity, this guide calls the complete proposed method **LTRO**. Replace that name with the final algorithm name in code and manuscripts. The comparison labeled **CTRO** means the existing method without the new latent trust-region term \(D_Z\).

The program has five phases:

- Phase 0: lock and validate the algorithm;
- Phase 1: demonstrate an advantage under representation collapse;
- Phase 2: establish non-inferiority in ordinary environments;
- Phase 3: test distributional robustness on Procgen;
- Phase 4: integrate the evidence and test the proposed causal chain.

Do not begin a large phase until the preceding phase's acceptance gate is met.

---

# Global experimental rules

## G1. Freeze the implementation before confirmatory testing

Create a tagged version of the code and a frozen configuration file containing:

- architecture;
- optimizer and learning-rate schedule;
- rollout length and minibatch size;
- PPO clip range;
- number of epochs in ordinary conditions;
- all CTRO coefficients;
- \(D_Z\) definition;
- reference-set construction;
- \(s_{\mathrm{ref}}\), \(\delta\), \(\eta\), and dual-controller settings;
- gradient clipping;
- evaluation schedule.

Any change after the freeze must produce a new version identifier. Never silently replace failed runs with a changed implementation.

## G2. Separate development from confirmation

Use three levels of data:

1. **Development:** debugging and choosing hyperparameters.
2. **Confirmation:** testing frozen choices across seeds.
3. **Held-out environments or games:** testing whether choices transfer.

Do not tune \(\eta\), auxiliary coefficients, or architecture independently for every test environment. If one environment-specific scale is necessary, derive it using a predeclared measurement rule rather than return maximization.

## G3. Match experimental budgets

All methods in a cell must receive the same:

- environment steps;
- observation preprocessing;
- architecture, except where the method necessarily adds a component;
- optimizer family;
- rollout data;
- number and timing of evaluations;
- random seeds where feasible.

Report both environment steps and wall-clock time. LTRO may use additional computation, but that cost must be visible.

## G4. Use paired seeds

Within each environment, use the same seed list for all methods. Pairing reduces noise because methods encounter comparable initializations and environment randomness.

Use at least:

- 5 seeds for development and screening;
- 10 seeds for headline confirmatory comparisons when affordable;
- more seeds for unusually noisy environments rather than relying on a few outliers.

Never delete a valid low-performing seed. Rerun only documented infrastructure failures, using the same seed when possible.

## G5. Predeclare primary outcomes

Each phase has a small set of primary outcomes. The broader metric collection is diagnostic and should not be used to search for whichever metric favors LTRO.

Across phases, report:

- individual seed results;
- mean, median, standard deviation, and confidence interval;
- learning curves;
- final-window evaluation performance;
- sample efficiency;
- wall-clock cost.

## G6. Keep evaluation separate from training

At each evaluation checkpoint:

- pause learning;
- run deterministic evaluation and, if relevant, stochastic evaluation;
- use held-out environment seeds or levels;
- do not update normalization statistics from held-out evaluations;
- report a fixed number of complete episodes.

Final performance should be the mean return over the final 10% or 20% of evaluation checkpoints, not the final training episode.

---

# Common methods

Use the following core methods where applicable:

| Label | Description | Purpose |
|---|---|---|
| PPO | Strong tuned PPO baseline | Reference algorithm |
| PFO | PPO with Proximal Feature Optimization | Existing representation-stability baseline |
| CTRO | Existing causal method without \(D_Z\) | Isolates the causal component |
| LTRO | CTRO plus adaptive latent trust region | Complete proposed method |
| LTRO-fixed | CTRO plus fixed \(D_Z\) penalty | Tests whether the dual controller is necessary |

Optional diagnostic baselines:

- PPO with a parameter-matched auxiliary head;
- PPO with generic weight decay or feature normalization;
- PPO with a smaller latent dimension;
- LTRO with the causal component removed;
- LTRO with warm-started \(D_Z\).

Do not include every optional baseline everywhere. Use them only when they answer a specific ambiguity.

---

# Common latent metrics

Compute metrics on both current on-policy observations and a fixed diagnostic dataset.

## Primary collapse measurements

1. **Functional target-fitting error:** freeze the encoder and train a new probe for held-out value targets or random cumulants using a fixed optimization budget.
2. **Near-zero distinct-pair rate:**

\[
C_{0.01}=\Pr\!\left[d(z_i,z_j)<0.01s_{\mathrm{ref}}\right].
\]

Report this for all pairs and, diagnostically, for pairs with nonidentical observations.
3. **Task-relevant probe accuracy:** use environment-specific labels such as optimal action, key possession, door state, velocity, or object identity.

## Secondary measurements

- stable rank and numerical rank;
- dormant-unit fraction;
- singular-value spectrum;
- participation ratio;
- latent and pre-activation norms;
- cosine similarity between features;
- CKA to the initial and previous representations;
- exact duplicate-observation rate;
- occupancy entropy;
- \(D_Z\), \(D_Z/\eta^2\), and signed log-distance change;
- \(\lambda\) and constraint-satisfaction rate;
- encoder-gradient norms for each loss component.

Rank alone must not define collapse. A representation is considered harmfully collapsed only when it loses functional or task-relevant information.

---

# Phase 0 — Lock and validate LTRO

## Goal

Choose one stable, defensible LTRO configuration without tuning it to maximize return.

## P0.1 Final \(D_Z\) definition

Use the reference-scaled log-distance form:

\[
D_Z=
\mathbb E_{(i,j)}
\left[
\log\left(\frac{d_{\mathrm{new}}(i,j)}{s_{\mathrm{ref}}}+\delta\right)
-
\log\left(\frac{d_{\mathrm{old}}(i,j)}{s_{\mathrm{ref}}}+\delta\right)
\right]^2.
\]

Requirements:

- \(s_{\mathrm{ref}}\) is calculated from the initial or declared healthy reference encoder and then fixed for the run;
- use the same pair indices for old and new embeddings;
- detach old embeddings;
- do not exclude near-zero pairs;
- leave duplicate observations in the sampling distribution;
- log duplicate and near-zero rates separately;
- use \(\delta=0.01\) unless a previously frozen value is already established.

## P0.2 Tight-radius experiment

Reuse the completed cartpole off, penalty, and loose-dual results. Add:

| Arm | \(\eta\) | \(\eta^2\) | Seeds |
|---|---:|---:|---:|
| dual-tight | 0.045 or rounded 0.05 | approximately 0.002–0.0025 | 5 |
| dual-medium | 0.070 | approximately 0.0049 | 5 |
| dual-loose | 0.140 | approximately 0.0196 | existing 5 |
| fixed penalty | not applicable | \(\lambda=1\) | existing 5 |
| off | not applicable | \(\lambda=0\) | existing 5 |

If compute is limited, run only \(\eta=0.05\).

## P0.3 Select the operational radius

Choose the radius using only these criteria:

- \(\lambda\) is not at its upper boundary for more than 10% of updates;
- \(\lambda\) is above its lower floor for approximately 25–75% of updates;
- median late \(D_Z/\eta^2\) is between 0.5 and 2;
- the geometry-gradient norm does not dominate the task-gradient norm for most updates;
- no clear training failure occurs;
- runtime overhead remains acceptable.

Do not select \(\eta\) according to the highest cartpole return.

If no radius satisfies these conditions, prefer the fixed penalty and state that adaptive dual control did not provide stable operational value.

## P0.4 Implementation and speed checks

Measure update time under:

1. CTRO without \(D_Z\);
2. \(D_Z\) computed with fixed \(\lambda\);
3. full adaptive dual.

Target less than 30% overhead. Investigate any larger difference before continuing. Confirm that:

- old reference features are cached where possible;
- no full \(n\times n\) pairwise matrix is constructed unnecessarily;
- one combined backward pass is used;
- logging does not force repeated GPU synchronization.

## Phase 0 acceptance gate

Proceed only if:

- all metrics are finite;
- no pairs are excluded;
- the controller does not saturate at its upper bound;
- the selected configuration materially changes \(D_Z\) or collapse diagnostics;
- ordinary learning is not obviously prevented;
- the implementation version and configuration are frozen.

### Phase 0 deliverable

A short calibration report containing:

- the selected radius or fixed penalty;
- the selection rule;
- constraint and gradient plots;
- runtime overhead;
- the frozen configuration identifier.

---

# Phase 1 — Representation-collapse stress test

## Goal

Show that LTRO outperforms PPO specifically when PPO loses useful representational capacity.

This is the most important mechanistic phase.

## P1.1 Environment selection

Choose environments satisfying all of the following:

- tuned PPO learns the task under the standard setting;
- PPO representation degradation can be induced reproducibly;
- observations require meaningful feature learning;
- evaluation is not prohibitively expensive.

Use at least:

- two pixel-based environments;
- two state-based or discrete-control environments if they are within the claimed scope.

Do not use an environment as a headline collapse benchmark unless PPO collapse is reproducible across multiple seeds.

## P1.2 Primary stress dial

Vary PPO optimization epochs per rollout:

\[
4,\;8,\;16,\;32.
\]

Keep learning rate, rollout data, architecture, minibatch size, environment steps, and clip range fixed. This intentionally increases optimization pressure per data collection cycle.

Use clip range as a secondary stress dial only after completing the epoch sweep:

\[
0.1,\;0.2,\;0.4,\;\text{unclipped diagnostic}.
\]

The unclipped condition is diagnostic and need not be described as ordinary PPO.

## P1.3 Experiment matrix

For every selected environment and epoch value, run:

| Method | Required? | Reason |
|---|---|---|
| PPO | Yes | Baseline collapse behavior |
| PFO | Yes | Prior representation-protection method |
| CTRO | Yes | Contribution of causal method without latent constraint |
| LTRO | Yes | Full method |
| LTRO-fixed | One or two environments | Tests necessity of dual adaptation |

Use 5 seeds for the full screening matrix. Increase to 10 seeds for headline environments and stress levels selected by a predeclared rule, such as the lowest epoch setting where at least half of PPO seeds meet the collapse definition.

## P1.4 Predeclare collapse

Define harmful representation collapse before examining the confirmatory returns.

Recommended definition:

A run is in collapse when both conditions hold for two consecutive checkpoints:

1. functional probe error exceeds a threshold derived from the healthy PPO reference distribution; and
2. either near-zero distinct-pair rate or dormant-unit fraction exceeds its healthy threshold.

Set healthy thresholds using standard-epoch PPO development runs. For example:

- probe error greater than the healthy mean plus two standard deviations;
- \(C_{0.01}\) greater than 10%, or greater than the healthy 95th percentile.

Do not require return to define collapse. Return is the downstream outcome.

## P1.5 Primary outcomes

1. Final-window evaluation return.
2. Functional probe error.
3. Collapse incidence across seeds.
4. Collapse onset step.
5. Return conditional on whether PPO collapsed in the matching environment/stress condition.

## P1.6 Required analysis

For each environment:

- plot return versus steps for all methods and epoch settings;
- plot probe error, \(C_{0.01}\), rank, dormant units, and \(D_Z\);
- show collapse incidence by method and stress level;
- show collapse onset distributions;
- compute LTRO–PPO return difference at each stress level;
- test whether LTRO's advantage increases with PPO collapse severity;
- inspect temporal ordering: representation failure should precede or accompany return degradation.

The key interaction is:

\[
\text{method advantage}
\times
\text{representation-failure severity}.
\]

LTRO should not merely have a higher overall mean; its advantage should be strongest where PPO representations fail.

## P1.7 Falsification conditions

The collapse-mechanism claim is weakened if:

- PPO return falls without functional representation loss;
- representation metrics worsen only after return falls;
- LTRO preserves geometry but not functional probes;
- PFO prevents collapse as well as LTRO with lower cost;
- LTRO's advantage is unrelated to PPO collapse severity.

## Phase 1 acceptance gate

The strong claim passes if:

- PPO collapse is reproducible under increased optimization pressure;
- LTRO significantly reduces collapse incidence or delays onset;
- LTRO retains functional capacity;
- LTRO outperforms PPO in the collapse conditions;
- this result holds across more than one environment;
- LTRO remains competitive with PFO and CTRO.

### Phase 1 deliverable

A collapse-stress report with one headline figure containing:

1. stress level;
2. functional collapse;
3. evaluation return;
4. LTRO–PPO advantage.

---

# Phase 2 — Ordinary-environment non-inferiority

## Goal

Show that LTRO does not sacrifice ordinary PPO performance when representation collapse is absent or mild.

## P2.1 Predeclare the non-inferiority margin

Before running the comparison, choose the largest practically acceptable loss relative to PPO.

Recommended normalized margin:

\[
\Delta=5\% \text{ of PPO performance},
\]

or a task-specific equivalent declared before results are inspected.

The claim is not “no significant difference.” The claim passes only if the lower confidence bound for LTRO–PPO is above \(-\Delta\).

## P2.2 Benchmark suite

Use a suite broad enough to cover the intended scope. A reasonable minimum is:

- 4–6 continuous-control tasks from MuJoCo or DMControl;
- 3–5 discrete or visual tasks if the method claims general PPO compatibility;
- at least one sparse-reward task;
- at least one environment where representation learning is important.

Do not rely on the current broken or weak PPO configurations. First verify that PPO reaches a recognized implementation-level target or matches a trusted internal reproduction.

Partition environments:

- development environments for any final implementation choices;
- held-out confirmation environments receiving the frozen method unchanged.

## P2.3 Methods and seeds

Required:

- PPO;
- CTRO;
- LTRO.

Include PFO on representative environments or the entire suite if affordable. Use at least 10 paired seeds for the main aggregate claim, or justify a smaller number using a power analysis based on pilot variance.

## P2.4 Primary outcomes

1. Aggregate normalized final-window evaluation return.
2. Per-environment LTRO–PPO return difference.
3. Area under the evaluation learning curve.
4. Wall-clock and sample overhead.

Normalize scores only with a rule declared before analysis. Also show raw scores so normalization cannot conceal failures.

## P2.5 Statistical decision

For each environment and the aggregate:

- calculate paired LTRO–PPO differences;
- construct a confidence interval by paired bootstrap over seeds;
- declare non-inferiority only if the lower bound exceeds \(-\Delta\).

Also report the number of environments where LTRO:

- is superior;
- is non-inferior;
- is inconclusive;
- is inferior.

One severe failure cannot be hidden by aggregate averaging. Predeclare a safety rule such as:

> No environment may show a confirmed degradation larger than 20% without being treated as a scope limitation.

## P2.6 Diagnose failures

For any environment where LTRO underperforms PPO by more than the margin, inspect:

- whether \(D_Z\) was active throughout early learning;
- geometry-gradient dominance;
- task-relevant probes;
- visitation diversity;
- whether a warm start removes the failure;
- whether the task admits a naturally compressed representation.

Do not silently retune only the failing environment. Report the failure or introduce a universal rule, such as a predeclared warm-start criterion, and rerun the whole confirmation set affected by that rule.

## Phase 2 acceptance gate

The ordinary-case claim passes if:

- aggregate LTRO performance is non-inferior to PPO;
- most individual environments are non-inferior or inconclusive rather than inferior;
- no unexplained catastrophic failure remains;
- sample and wall-clock overhead are acceptable for the claimed use case.

### Phase 2 deliverable

A benchmark table with raw returns, normalized returns, confidence intervals, runtime, and non-inferiority status for every environment.

---

# Phase 3 — Procgen distributional robustness

## Goal

Show that LTRO preserves training performance while improving performance on unseen levels, and determine which component causes the gain.

## P3.1 Protocol

Use a fixed train/test level split:

- train only on the declared training levels;
- evaluate on both training and unseen test levels;
- never update the agent or running statistics using test-level data;
- keep the number of training levels and environment steps identical across methods;
- use the same level seeds for every method.

Use the standard Procgen evaluation settings implemented by the selected codebase and record them completely. Do not mix protocols from different implementations.

## P3.2 Game selection

Best option: run the full Procgen suite.

If compute is limited:

1. predeclare a diverse subset before running comparisons;
2. include games with navigation, object interaction, timing, and visually varied layouts;
3. do not choose games because preliminary LTRO results were favorable;
4. label the result as a subset study.

Use the same frozen LTRO configuration across games, apart from scale values derived by the predeclared \(s_{\mathrm{ref}}\) rule.

## P3.3 Methods

Required ablations:

| Method | Causal component | \(D_Z\) | Purpose |
|---|---:|---:|---|
| PPO | No | No | Baseline |
| PFO | No | Feature regularizer | Representation-stability baseline |
| CTRO | Yes | No | Causal contribution |
| PPO+\(D_Z\) | No | Yes | Latent trust region without causal objective |
| LTRO | Yes | Yes | Full method |

This factorial structure separates:

- generic regularization;
- causal representation learning;
- latent stability;
- interaction between causal learning and latent stability.

Use at least 5 seeds per game for screening and 10 seeds for the final aggregate if computationally possible.

## P3.4 Primary outcomes

1. Aggregate normalized unseen-level return.
2. Aggregate generalization gap:

\[
G=R_{\mathrm{train\ levels}}-R_{\mathrm{test\ levels}}.
\]

3. Training-level return.
4. Number of games where LTRO improves unseen-level return without degrading training return.

Secondary outcomes:

- sample efficiency on training levels;
- functional probes on held-out observations;
- representation similarity across level variations;
- performance under visual or layout changes, if part of the declared protocol;
- wall-clock overhead.

## P3.5 Causal attribution

Better test return alone does not prove that the causal component is responsible. Look for this pattern:

1. CTRO outperforms PPO on unseen levels.
2. PPO+\(D_Z\) provides some stability benefit or no harm.
3. LTRO outperforms both CTRO and PPO+\(D_Z\), indicating a useful interaction.
4. Training-level return remains comparable.
5. Functional probes show preservation of level-invariant, task-relevant information.

Use a factorial analysis with two binary factors—causal component and \(D_Z\)—to estimate their main effects and interaction. Treat game and seed as sources of variation.

## P3.6 Failure interpretations

- **Higher train and test return:** possibly general optimization improvement, not specifically robustness.
- **Same train return, higher test return:** strongest robustness pattern.
- **Lower train and higher test return:** possibly excessive regularization; inspect whether the tradeoff is practically acceptable.
- **Improvement from PPO+\(D_Z\) but not CTRO:** stability, not causal structure, explains the gain.
- **Improvement from CTRO but no added LTRO benefit:** causal objective helps, but latent trust region is unnecessary on Procgen.
- **Only a few games improve:** report task dependence rather than claiming general robustness.

## Phase 3 acceptance gate

The distributional-robustness claim passes if:

- LTRO improves aggregate unseen-level performance;
- training-level performance remains non-inferior;
- the generalization gap decreases;
- gains are distributed across multiple games;
- ablations attribute at least part of the gain to the causal component or its interaction with \(D_Z\).

### Phase 3 deliverable

A Procgen table and figure showing, per game and in aggregate:

- training return;
- unseen-level return;
- generalization gap;
- CTRO, \(D_Z\), and interaction effects.

---

# Phase 4 — Evidence integration and causal-chain analysis

## Goal

Determine whether the complete evidence supports the proposed mechanism:

\[
\text{optimization pressure or distribution shift}
\rightarrow
\text{loss of useful representation}
\rightarrow
\text{performance loss},
\]

and whether LTRO interrupts that chain.

## P4.1 Required temporal evidence

For collapse-stress runs, align checkpoints by environment steps and determine:

- when functional probe error begins to rise;
- when near-zero pair rate or dormant units increase;
- when return begins to decline;
- when \(\lambda\) activates;
- whether LTRO prevents or delays those transitions.

Representation failure should precede or coincide with performance decline. If it consistently follows performance decline, it is more likely an effect than a cause.

## P4.2 Conditional effect analysis

Divide conditions using the predeclared PPO collapse criterion:

- PPO-healthy conditions;
- PPO-collapse conditions.

Estimate LTRO–PPO performance separately in both groups.

The desired pattern is:

- non-inferiority in PPO-healthy conditions;
- clear superiority in PPO-collapse conditions.

## P4.3 Cross-phase evidence table

Create a final claim table:

| Claim | Required evidence | Pass condition | Result |
|---|---|---|---|
| Ordinary-case safety | Phase 2 | Non-inferior aggregate and no unexplained severe failures | Pending |
| Collapse advantage | Phase 1 | Better capacity and return when PPO collapses | Pending |
| Distributional robustness | Phase 3 | Better unseen-level return with comparable train return | Pending |
| Causal attribution | Phases 1, 3, 4 | Temporal ordering and component ablations | Pending |
| Practical viability | All phases | Acceptable runtime and tuning burden | Pending |

Do not use one strong phase to compensate rhetorically for a failed claim in another phase.

---

# Recommended execution order

## Immediate

1. Run Phase 0 tight and medium radii.
2. Select the operational radius without using return as the selection criterion.
3. Freeze the LTRO configuration and code version.

## Next priority

4. Run Phase 1 on one reproducible pixel-based collapse environment.
5. Verify that PPO loses functional capacity before scaling the matrix.
6. Add PFO, CTRO, and LTRO.
7. Expand to additional environments only after the mechanism reproduces.

## Then

8. Run the Phase 2 ordinary-environment suite with the frozen configuration.
9. Address universal failure modes, not environment-specific tuning.

## Finally

10. Run the Procgen screening subset or full suite.
11. Promote to confirmatory seeds with unchanged settings.
12. Complete the Phase 4 integrated analysis.

---

# Minimal run registry

Every run must store:

- run identifier;
- code commit or version tag;
- complete configuration;
- environment and environment version;
- method and ablation label;
- seed;
- training and test level seeds where relevant;
- start and completion time;
- environment steps;
- wall-clock duration;
- hardware;
- completion status and failure reason;
- checkpoint paths;
- metric file paths.

The aggregation script must refuse to combine runs with different frozen configuration identifiers unless the comparison explicitly studies that difference.

---

# Final decision standard

Evidence for LTRO as a principled extension or successor to PPO is convincing only if all three patterns appear:

1. **When PPO remains healthy, LTRO is non-inferior.**
2. **When PPO loses useful representational capacity, LTRO preserves that capacity and achieves higher return.**
3. **On unseen Procgen levels, LTRO generalizes better, and ablations connect the gain to the causal component and/or its interaction with latent stability.**

The strongest defensible conclusion would then be:

> LTRO extends PPO with a latent trust region that preserves ordinary-task performance while improving performance in regimes characterized by representation failure and distribution shift.

Calling it a universal replacement for PPO would additionally require broader architecture, observation-modality, and large-scale benchmark evidence beyond this core program.
