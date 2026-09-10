# Next round of experiments: Anti-aliased PPO

Supersedes the previous version of this document, which planned experiments for a method
that is no longer the method. Companion to `aliasing_framework.tex`, which is the
authoritative statement of definitions, theorems, and evidence status.

Recoverability of the old CTRO recipe: [`METHOD_CHANGELOG.md`](METHOD_CHANGELOG.md).

**What changed.** The four-arm comparison rejected the bisimulation term against its own
pre-declared retention criterion and rejected spectral normalization by dominance. The
multi-task test rejected the hypothesis that the latent trust region increases separation.
The surviving method is PPO plus a single scale-corrected PL hinge. The surviving theory
is the impossibility proposition, the invariance-group proposition, and the two transfer
theorems.

---

## 1. The method

PPO, unchanged, including ratio clipping. One auxiliary term on the encoder.

```python
# once, at startup: sample and freeze
ref_states = sample_states(N=2048)              # never resampled

# per update, encoder phase (value head frozen)
z      = encoder(ref_states)
v      = V(z)                                    # frozen head
g2     = grad_norm_sq(v, z)                      # ||grad_Z V||^2, per state
v_ref  = ema(v.quantile(0.99), tau=0.01)         # reference level
f      = (v_ref - v).clamp_min(1e-3)
L      = pdist(z).median()                       # scale correction
mu_t   = L**2 * g2 / (2 * f)                     # scale-corrected diagnostic
loss  += lam_pl * (mu_min - mu_t).clamp_min(0).mean()
```

Head phasing: `E_enc = 4` epochs updating encoder and policy head with the value head
frozen, then `E_val = 4` epochs updating the value head with the encoder frozen. Assert
that gradients do not reach frozen parameters in either phase.

The hinge is one-sided. Do not maximize `mu_t` — unbounded maximization admits the
contraction solution, which is the failure mode the method exists to prevent.

Preset: `--algo-preset anti_aliased_ppo`. Legacy stack: `--algo-preset ctro_full_legacy`.

### Deleted from the previous stack

| Component | Reason |
|---|---|
| MICo loss and its target network | Failed pre-declared retention criterion |
| Spectral normalization | Dominated on both return and separation |
| `D_Z` term, EMA target encoder, adaptive `lambda_dz`, distance floor | Not in the method; see §4 |
| Pair distribution `nu`, local-pair sampling | Needed only to state and test Theorem 2 |

### Hyperparameters to fix before any run

`mu_min`, `lam_pl`, `N`, `tau` for the reference-level EMA, `f_min`, `E_enc`, `E_val`.
Set these once on a single task and hold them fixed across all reported runs. Report the
sweep used to set them.

---

## 2. Blocking checks

Nothing below is interpretable until these are resolved.

**B1. Participation ratio near one.** Report the full eigenvalue spectrum of the latent
covariance, the histogram of pairwise distances, and the ratio of the fifth percentile of
pairwise distances to the median, on the reference batch. If the fifth percentile is
within numerical tolerance of zero, the encoder is collapsed and every geometric quantity
computed on it is meaningless. Fix the architecture before proceeding.

**B2. Denominator provenance of the existing results.** Determine whether the four-arm
comparison ran after the reference-level rewrite or on the earlier Bellman-residual
denominator. If the latter, arm A is not the method and its numbers do not transfer.

**B3. Scale correction provenance.** Determine whether the separation and PL statistics in
the completed runs were scale-corrected. One arm contracted the latent by a factor of
three to eight in median pairwise distance; comparisons of uncorrected statistics across
arms of differing latent scale are not interpretable.

**B4. PPO baseline budget.** Targets for a correctly configured baseline on state
observations: cartpole-swingup above 800, walker-walk above 900, cheetah-run above 400.
A comparison against a baseline below these carries no information. Hopper-hop is reported
as unsolved rather than used as a separator.

Status write-up: [`blocking_checks_aa_ppo.md`](blocking_checks_aa_ppo.md).

---

## 3. Experiments

Every hypothesis below is stated in two forms. **H1** is the outcome predicted by the
theory. **H0** is the statement that must be false for H1 to carry information. Both are
pre-declared here and are not to be revised after seeing results.

### 3.0 Three of these are null claims, and need equivalence tests

Several predictions in this plan are claims of *no difference*. These are stated as
**equivalence hypotheses** and tested by two one-sided tests against a pre-declared margin
`Delta`, or equivalently by requiring the confidence interval on the difference to lie
entirely inside `[-Delta, +Delta]`. Each margin must be fixed before running.

### E-A. The shear experiment

**Priority: first.** Cheap, requires no new algorithm, and is the only direct empirical
demonstration of the central theoretical claim.

Take a trained encoder. Apply a compensated reparameterization by
`A = diag(1, c, 1, ..., 1)` — multiply the encoder output by `A`, multiply the input
weights of both heads by `A^-1`. Sweep `c` over roughly `{1, 2, 5, 10, 30}`. Five
checkpoints from independent seeds.

- **E-A.1** Behavioural invariance (equivalence): max abs deviation of logits/values
  vs `Delta = 1e-5` relative.
- **E-A.2** Informational invariance (equivalence): MI / sufficiency / bisimulation
  proxies unchanged within TBD margins.
- **E-A.3** Conditioning statistic: scale-corrected `mu_t` decreases as `1/c^2`.
- **E-A.4** Separation: scale-corrected p05 decreases in `c`.
- **E-A.5** Harm to subsequent learning (load-bearing): resume return lower for larger `c`.
- **E-A.6** Orthogonal gauge: functions, `mu_t`, and separation unchanged.

Script: `python -m src.experiments.shear_gauge_experiment` (CPU for .1–.4/.6).
Resume jobs: `src/experiments/jobs/ea5_resume_shear_s.sh`.

**Run E-A before committing compute to E-B.**

### E-B. Headline comparison

Anti-aliased PPO against unmodified PPO. Matched architecture, matched tuning budget.
Tasks: cartpole-swingup, walker-walk, cheetah-run, plus a four-game Procgen subset.
Seeds: ten. Aggregation: IQM with stratified bootstrap CIs.

### E-C. Mechanism (logged during E-B)

Scale-corrected p05 separation; participation ratio; `g2` and `f` as separate series;
median pairwise distance.

### E-D. The contraction account

Hinge on corrected `mu_t` versus uncorrected `mu_PL`; track median pairwise distance and
return.

---

## 4. Disposition of prior results

See [`METHOD_CHANGELOG.md`](METHOD_CHANGELOG.md).

---

## 5. Open theory items

**The estimator gap.** Theorem 2 concludes about a neighbour-quantile slope; the code logs
the full gradient norm then a fifth percentile over states. Until a neighbour-quantile
estimator or a variant theorem exists, no measurement bears on Theorem 2.

**The value-drift cross-term.** Transfer theorems hold the value assignment fixed.

**Learned critic against optimal value.** No argument connects them.

**Bounded extent in the collapse-implies-aliasing argument.** Pigeonhole needs a
bounded-extent hypothesis.

---

## 6. Not being run, and why

- Rerunning the four-arm ablation grid.
- The rate sweep, until the estimator gap is closed.
- Hopper-hop as a separator.
- Any experiment requiring the `D_Z` apparatus, unless log re-analysis recovers a
  preservation result.
- Distributional-shift generalization runs.

---

## 7. One literature check before committing to the title

Confirm the literature review distinguishes the single PL hinge from existing rank and
plasticity regularizers via the impossibility proposition.
