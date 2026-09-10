# $D_Z$ experiment guide (implementation companion)

**Experiment ordering and hypotheses:** see `next_round_experiments.md` (supersedes the
experiment-planning content here). Tier 0 results: `t0_findings.md`. Conceptual
identity: `ctro_conceptual_foundations.md`.

Phases, configs, and commands below remain the implementation companion. Code lives in:

| Piece | Path |
|-------|------|
| μ_PL rewrite | `src/metrics/pl_ratio.py`, `src/losses/pl_coupling.py` |
| D_Z loss (reference-scaled log) | `src/losses/dz_trust_region.py` |
| Agent | `src/agents/ctro.py`, `src/agents/ppo.py` |
| Geometry gate | `python -m src.experiments.diagnose_latent_geometry` |
| Math checks | `python -m src.experiments.validate_dz_geometry` |
| μ_PL offline | `python -m src.experiments.recompute_mu_pl` |
| Aggregation | `python -m src.experiments.aggregate_performance_eval` (IQM+bootstrap) |
| Phase 3 runner | `src/experiments/jobs/dz_phase3.sh` |

## $D_Z$ definition

Freeze at run start
$s_{\mathrm{ref}}=\mathrm{median}_{(i,j)}\lVert z_i-z_j\rVert_2/\sqrt{d}$
on the initial reference encoder (abort if $s_{\mathrm{ref}}<10^{-8}$).
Never recompute $s_{\mathrm{ref}}$.

$$
\bar d=\frac{\lVert z_i-z_j\rVert_2}{\sqrt{d}\,s_{\mathrm{ref}}},
\qquad
D_Z^{\log}
=
\mathbb{E}_{(i,j)}\bigl(\log(\bar d_{\mathrm{new}}+\delta)-\log(\bar d_{\mathrm{old}}+\delta)\bigr)^2
$$

with absolute $\delta=$ `dz_delta` (default `0.01`).
All sampled pairs enter the mean (local/global mixture via `lambda_loc`); no near-zero exclusions.
Dual form: $L+\lambda(D_Z^{\log}-\eta^2)$, $\lambda\ge 0$.

Collapse diagnostic: $C_{0.01}=\Pr[\bar d_{\mathrm{old}}<0.01]$ (`dz_C_0p01`).

Fix 2 (exclude near-zero pairs from the mean) is not implemented unless log-ratio shows pathology.

## Phase 0 gates

```bash
# CPU only
CUDA_VISIBLE_DEVICES= python -m src.experiments.validate_dz_geometry --device cpu
python -m src.experiments.recompute_mu_pl --run-dir results/.../seed_42/cartpole-swingup --device cpu
python -m src.experiments.diagnose_latent_geometry --run-dir results/.../seed_42/cartpole-swingup --device cpu
```

Baseline targets documented in `docs/baseline_dmc_p0_3.md`.

## η calibration (required after scaled log)

Previous η values are obsolete under $s_{\mathrm{ref}}$ scaling. Calibrate:

```bash
# λ frozen at 0: log D_Z under CTRO encoder motion, no penalty
sbatch src/experiments/jobs/dz_eta_calib_pixels_cartpole_s.sh
# First 20% of post-first-update D_Z: η² = p25, η to 2 sig digits
python -m src.experiments.fit_dz_eta_from_calib \
  --run-dir results/dmcontrol_pixels/exp_ctro_dz_calib/seed_42/cartpole-swingup \
  --early-frac 0.2
```

Then confirmation with that $\eta$:

```bash
ETA_DZ=<calibrated> sbatch src/experiments/jobs/dz_confirm_pixels_cartpole_s.sh
```

## Phase 1 flags

```text
--dz-enabled --eta-dz <calibrated> --lambda-dz 1
--no-dz-adapt --lambda-dz 0   # calib: observe D_Z without penalty
--dz-delta 0.01               # absolute δ on d_bar
--no-policy-clip              # Run 1: D_Z is the only policy constraint
--head-phasing                # optional head phasing
```

## Logging (per update in metrics.csv)

`D_Z`, `dz_s_ref`, `dz_delta_eps`, `dz_log_delta_p{50,90,95,99}`,
`dz_signed_log_median`, `dz_d_bar_{old,new}_median`, `dz_C_0p01`,
`dz_expand_frac`, `dz_contract_frac`, `lambda_dz`, `eta_dz`, `dz_eta_sq`,
`dz_collapse_unjustified_frac`, `kl`, participation ratio,
`on_policy_to_buffer_dist`, `ref_buffer_refresh`,
`mu_pl_*`, **`grad_sq_*`**, **`f_*`**, `v_ref`, `f_floor_rate`,
`mu_pl_valid_fraction`.

Every metrics dump also writes `f_gap_hist/epoch_*_step_*.npz` (raw \(f=\hat V_q-V\)
histogram, no clamp) for post-hoc \(\tau\) sensitivity without re-running.

Collapse fractions are relative to the local/global pair mixture (not stratified-by-decile).
Always decompose $\mu_{PL}$ via `grad_sq` (numerator) and `f` (denominator); the ratio alone
is ambiguous.

## Run order

validate → η calib (λ=0) → re-pilot → (gate) → headline 24-cell → eval.

```bash
bash src/experiments/jobs/dz_phase3.sh validate
sbatch src/experiments/jobs/dz_eta_calib_pixels_cartpole_s.sh
# after η set:
ETA_DZ=... sbatch src/experiments/jobs/dz_pilot_pixels_cartpole_s.sh
```

## Anticipated failures

λ at upper clip → raise η; unjustified collapse high → inspect MICo targets / encoder;
log-ratio gradient pathology at tiny $d$ → consider Fix 2 (exclude floor-active).
Run 1 KL explosion under bounded $D_Z$ → report negative result, fall back to Run 2.

## Pilot finding (2026-08-13): fixed-$\eta$ pilot vs $\lambda=0$ calib

Cancelled fixed-$\eta$ pilot `dz-pilot-pix` / `43377648` (`exp_ctro_dz`, cartpole-swingup
seed 42, $\eta=\sqrt{\mathrm{p25}}=0.10123514$) after ~18.6h / epoch ~1596.
Compared against $\lambda=0$ calib `exp_ctro_dz_calib` on the same stack/seed.

**Gates on geometry looked fine and are not the story.** $\lambda_{D_Z}$ stayed interior
(never pinned at $10^4$); $D_Z$ was held at $\approx 0.010$ on 97.8% of steps; collapse /
floor-activation stayed low (~1%). Returns still collapsed (~150 early → ~40 late).

**The single-seed rank-collapse reading was withdrawn after the five-seed run.**
The pilot's participation-ratio contrast was seed noise. Across seeds 42–46, off and
fixed are indistinguishable (1.86 ± 0.21 vs 1.88 ± 0.16; paired p=0.835). This
reproduces the earlier Minigrid result: global feature rank does not separate the
conditions.

**$\mu_{PL}$ / cross-term — do not over-read.** Early `f_floor_rate` ≈ 0.9, so early
$\mu_{PL}\approx\mathrm{grad\_sq}/(2 f_{\mathrm{floor}})$ is a clamp artifact
(~1800 for $\mathrm{grad\_sq}\approx 3.6$), not a measurement. Much of the apparent
1468 → 74 drop is the floor *releasing* as $f$ grows past the clamp. The value-drift
cross-term (proofs §5) remains open and worth proving, but **this run is not evidence
that it is the binding term.** P0.1's code path is correct; its acceptance criterion
(floor rate < 1%) is **not** met in the formation phase, so formation-phase $\mu_{PL}$
is not yet trustworthy until floored samples are masked out of the diagnostic.

**Calib `grad_sq` / `f` decomposition (on disk).** 0–100 → 400–500:

| | $\mathrm{grad\_sq}$ | $f$ | PR | return |
|---|---|---|---|---|
| pilot | 3.6 → 9.6 (×2.7) | 0.027 → 0.93 (×34) | 6.4 → 2.1 | 152 → 73 |
| calib | 1.2 → 8.1 (×6.9) | 0.062 → 0.50 (×8) | 2.1 → 2.1 | 133 → 246 |

Calib $f$ rose ~8×, not ~34× — the lock widened the value gap relative to unconstrained
CTRO in this single seed. Absolute late $\mathrm{grad\_sq}$ is similar (~8–10).
Do not infer a rank mechanism from this table; the five-seed comparison supersedes it.

**Fixed $\eta$ cannot span formation and refinement.** Unconstrained calib $D_Z$ spans
~0.49 → ~0.014 (≈50×). $\eta$ fit on post-warmup p25 matches the *settled* regime only.

**Rejected:** disable $D_Z$ until epoch 500 (abandons the trust region during formation).

**Next:** adaptive radius floored at the calibrated settled rate
$\eta_t=\max(\eta_{\min},\,c\cdot\overline{D_Z}^{(t-1)})$ with $c\approx 1.5$,
$\eta_{\min}=0.101$, EMA from *before* the current update, slow timescale. Always
$\eta_t\ge\eta_{\min}$ (never tighter than calib). Three-arm short horizon: $D_Z$ off;
fixed $\eta$; adaptive $\eta$. Instrument PR, pairwise distance histograms at
checkpoints, and realized $\eta_t$ on the adaptive arm. Mask floored samples out of
$\mu_{PL}$ percentiles and log the mask rate.

### Prerequisite checks

- **P0.1 conditional-domain definition:** let
  $\mathcal S_\tau=\{s:f(s)=\hat V_q-V(s)>\tau\}$, with
  $\tau=\max(10^{-12},10^{-3}|\hat V_q|)$. Report
  $\mu_{PL,0.05}$ only over $\mathcal S_\tau$, plus
  `mu_pl_valid_fraction` $=|\mathcal S_\tau|/|\mathcal S|$. States outside this
  domain make the PL inequality vacuous and are excluded from both the diagnostic
  and (after the 2026-08-14 critical-path fix) PL coupling loss; no denominator is
  clamped in new runs. DMC uses the full buffer
  (`metric_pl_max_samples=None`). The old “floor rate <1%” acceptance criterion
  does not apply to this conditional estimand; Run 4 must report coverage and fail
  if it falls below its declared minimum.
- **Phase 2 gauge:** `validate_dz_geometry` passes (`D_Z≈0` under orthogonal gauge;
  displacement foil large). Rotation-locking eliminated.

### Three-arm short horizon (seed 42) and pilot≠fixed

First array `43469318` failed on all-floored $\mu_{PL}$ raise; requeue `43493816`
completed all three arms (~8–9h). Overlay:
`plots/dz3_pairwise_distance_overlays.png`.

| arm | last-50 ret | late PR | final dist p05 | late `dz_floor_activation` |
|---|---|---|---|---|
| off | 165 | 1.35 | $2.0\times 10^{-4}$ (not zero) | ~0.057 (not saturated) |
| fixed | 206 | 2.20 | 0.94 | ~0.002 |
| adapt | 158 | 1.88 | 0.48 | ~0.001 |

**Off p05 is a small positive, not genuine zero.** Floor-activation ~6% matches a
left-tail mass below $d_{\mathrm{floor}}$, not a constantly-pinned floor guard.

**Adaptive $\eta$:** behaved as designed (seeded permissive → $\eta_{\min}$ by epoch
~100) and the design was wrong for formation — reported as a single-seed ablation
only, not expanded.

**Why fixed ≠ cancelled pilot (gates interpretation).** Same $\eta$, $\lambda$ adapt,
stack, seed 42. Config diffs through epoch 732 are non-causal (`early_stop`,
`total_epochs`, unused `dz_eta_*` keys). Trajectories diverge from epoch 6
(returns/$\lambda$/$D_Z$). Both V100; `cudnn.deterministic=True`. Conclusion:
**single-seed GPU nondeterminism**, not a silent HP change. The cancelled pilot's
return/PR collapse is one draw; `exp_dz3_fixed` seed 42 is a counter-draw. Geometry
evidence for now is PR + distance percentiles, not $\mu_{PL}$.

**Next (compute):** five-seed off vs fixed (`exp_dz5_*`, seeds 42–46, job
`dz_off_vs_fixed_5seed_s.sh`) under scale-relative floor + full-buffer $\mu_{PL}$.
Adaptive not expanded.

### Five-seed result: local distinguishability, not global capacity

Job `43539537` completed all ten cells (off/fixed × seeds 42–46, 1.5M steps).
The policy KL/PPO clip remained active in both arms, so this is **Run 2**:
$D_Z$ as an additional constraint. Run 1 (remove KL/PPO clipping) remains open.

| metric | off | fixed | paired result |
|---|---:|---:|---|
| last-50 return, mean | 112.8 | 163.0 | Δ=+50.2; paired t p=0.069 |
| last-50 return, IQM | 112.8 | 163.3 | paired bootstrap IQM Δ 95% CI [-4.9, 103.1] |
| final pair-distance p05, mean | 0.143 | 0.419 | Δ=+0.275; all 5 seeds positive |
| final pair-distance p05, IQM | 0.143 | 0.378 | paired bootstrap IQM Δ 95% CI [0.041, 0.561] |
| late participation ratio, mean | 1.863 | 1.881 | Δ=+0.018; paired t p=0.835 |

The mechanism supported by these data is **local distinguishability** (protection
against state aliasing in the lower tail of pairwise distances), not rank/capacity
preservation. Participation ratio measures global covariance dimension and does not
move. Pair-distance p05 separates every matched seed, and return moves in the same
direction in 4/5 seeds. Lead with the geometric result; treat return as corroborating
at five seeds.

Inference uses matched-seed resampling because each off/fixed pair shares a seed.
With only five pairs, report the bootstrap interval and direction count rather than
turning the bootstrap tail probability into an asymptotic p-value. Welch's unpaired
test gives p=0.051 for return and p=0.045 for p05, but discards the pairing.

The five-seed diagnostic already excluded $f\le\tau$ samples, although those runs'
training PL loss still used the old clamp; that implementation was shared by both
arms and is now superseded. Formation coverage was
16.1% (off) and 17.3% (fixed), so early conditional $\mu_{PL}$ is based on the
surviving domain—not on manufactured clamped ratios—but must always be shown with
coverage. Per-sample `grad_sq` and `f` were not persisted on those runs; subsequent
runs write `f_gap_hist/` every metrics dump.

### Run 1 pilot (submitted)

Job script `dz_run1_pilot_s.sh`: seed-42 pilot with required stress control
(no clip, no $D_Z$) and $\eta\in\{0.101,0.32,0.71\}$. Pre-declared outcomes in
[`docs/dz_run1_pilot_criteria.md`](dz_run1_pilot_criteria.md). Theory does not
predict success; a negative is consistent with gauge freedom under $D_Z$.
