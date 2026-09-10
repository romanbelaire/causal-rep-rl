# Tier 0 findings

Zero-compute checks from `next_round_experiments.md`, grounded in
`ctro_conceptual_foundations.md`. Date: 2026-08-15.

---

## T0.1 — Separate Minigrid cells (contraction prediction)

**Artifacts.** `plots/ctro/t0_1_separated_cells.png`,
`plots/ctro/t0_1_endpoint_pairwise.csv`.

**What was available.** Training logs have return and $\mu_{PL}$ for all four cells,
but **never logged pairwise latent distances**. Intermediate checkpoints were not
kept (only `weights_final` / `weights_latest`), so a true over-training pairwise
series cannot be recovered. Endpoint pairwise distances were recomputed on a shared
`MiniGrid-Unlock-v0` probe buffer (same procedure as `analyze_pr_manifold.py`),
CPU only.

**Separated late-window means (seeds 42–44).**

| Cell | late $\mu_{PL}$ q05 | late return | final median pairwise | final p05 | mean $\|Z\|$ |
|---|---:|---:|---:|---:|---:|
| BASELINE | 0.044 | 0.522 | 0.070 | 0.0021 | 0.069 |
| MICO_ONLY | 0.067 | 0.292 | 0.021 | 0.0035 | 0.037 |
| PL_ONLY | 0.349 | 0.606 | 0.056 | 0.0019 | 0.079 |
| FULL (α=β=0.5) | 0.272 | 0.550 | 0.037 | 0.0028 | 0.084 |

**Against the §5.1 hypothesis.**

- Predicted: PL-only contracts (median pairwise falls), $\mu_{PL}$ rises, return does
  not follow.
- Observed: PL-only has the **highest** late $\mu_{PL}$ **and** the highest late
  return among the four cells. Endpoint median pairwise is **not** smaller than
  baseline (0.056 vs 0.070) and is **larger** than FULL (0.037). The smallest
  distances are under MICO_ONLY.
- So on this Minigrid VAE stack the contraction mechanism is **not** confirmed.
  The pooled summary that “PL-on cells raise $\mu_{PL}$ and return” was not hiding a
  PL-only collapse; separating the cells makes PL-only look like the strong cell.

**Decision (as pre-declared).** This is a **null for §5.1 on Minigrid**, with the
documented caveat that `vae_coef=0.1` independently regularizes latent scale. The
scale-invariant $\tilde\mu_{PL}$ (T1.1) remains justified as closing a loophole in
the surrogate, but Minigrid no longer supplies empirical pressure for it. **E3**
(non-VAE stacks) is now the decisive test of the contraction story, not a
nice-to-have.

---

## T0.2 — Sandwich claim vs Castro et al. (MICo)

**Source.** Castro et al., *MICo: Improved representations via sampling-based state
similarity for MDPs* (NeurIPS 2021).

**Exact statements that matter.**

1. **Proposition 4.8.** For any policy $\pi$ and states $x,y$,
   $|V^\pi(x)-V^\pi(y)| \le U^\pi(x,y)$, where $U^\pi$ is the MICo **diffuse** fixed
   point (independent couplings; self-distances need not be zero).
2. **Diffuse metrics (Def. 4.9).** $U^\pi$ is not a metric in the usual sense:
   $U^\pi(x,x)$ can be positive.
3. **Reduced MICo.** The quantity realized by angular embedding distances is
   $\Pi U^\pi(x,y) = U^\pi(x,y) - \tfrac12 U^\pi(x,x) - \tfrac12 U^\pi(y,y)$.
4. **Lemma 5.1.** The value upper bound does **not** transfer to $\Pi U^\pi$ in
   general: there exist MDPs with $|V^\pi(x)-V^\pi(y)| > \Pi U^\pi(x,y)$.

**Contact with our implementation.** `src/losses/mico.py` matches a diffuse
potential $U_\omega$ (norm terms + angular distance) to the MICo Bellman target.
The angular / embedding part is in the $\Pi U$ family. Prop 4.8 therefore justifies
an upper bound on value differences in terms of **$U_\omega$ (or $U^\pi$)**, not in
terms of Euclidean $\|Z_i-Z_j\|$ alone.

**Decision.** The informal §4.2 sandwich — “MICo upper-bounds how fast value changes
with *latent distance*” — does **not** survive contact with the exact form if
“latent distance” means embedding Euclidean / angular distance. It **does** survive
if the upper bound is stated against the diffuse MICo potential that the loss
actually fits. §4.2 and §6 must be rewritten before drafting around them:
condition-number targeting on the ratio of PL steepness to **embedding** distance is
not licensed by Prop 4.8; targeting against $U$ (or a calibrated proxy) might be.

**E5 status.** Blocked until that rewrite. Do not run E5 on the current sandwich
wording.

---

## T0.3 — Run 2 statistical protocol

**Artifact.** `docs/run2_analysis_protocol.json` (machine-readable) and the
summary below. Job `43539537`, cartpole-swingup, seeds 42–46, arms
`exp_dz5_off` vs `exp_dz5_fixed`.

**Protocol (freeze this for E1).**

1. **Point estimate:** interquartile mean (IQM) — mean of observations in
   $[q_{25}, q_{75}]$ inclusive; if $n<4$, ordinary mean
   (`aggregate_performance_eval.interquartile_mean`).
2. **Interval:** percentile bootstrap 95% CI, $n_{\mathrm{boot}}=10000$, RNG seed 0.
3. **Pairing:** resample **matched-seed differences** with replacement, then take
   IQM of the resampled deltas. Do not use unpaired Welch tests as the primary
   report.
4. **Sign consistency:** report counts of positive / negative paired deltas
   alongside the interval.
5. **Multi-task extension (E1):** stratified bootstrap — within each draw, resample
   seeds inside each task, pool the sampled values, then IQM
   (`stratified_bootstrap_iqm_ci` with task strata). Still report per-task tables.

**Recomputed paired $\Delta$ IQM (fixed − off).**

| Metric | off IQM | fixed IQM | $\Delta$ IQM | 95% CI | signs |
|---|---:|---:|---:|---|---|
| last-50 return | 112.8 | 163.3 | +51.5 | [-0.85, 97.1] | +4 / −1 |
| pairwise p05 | 0.143 | 0.378 | +0.240 | [0.046, 0.561] | +5 / −0 |
| pairwise p50 | 1.220 | 1.570 | +0.453 | [-0.47, 0.92] | +3 / −2 |
| late PR | 1.871 | 1.981 | +0.040 | [-0.19, 0.28] | +3 / −2 |

Same qualitative call as before: **p05 is the lead confirmed geometric effect**;
return leans positive but CI includes zero; PR does not separate. Small CI
numerics differ from the earlier NOTES entry because of bootstrap seed / draw count;
the protocol above is now the source of truth for E1.

---

## Consequences for Tier 1 / Tier 2 ordering

| Item | Effect of Tier 0 |
|---|---|
| T1.1–T1.2 scale-invariant / value-normalized $\mu_{PL}$ | Still do — closes a real surrogate loophole even without Minigrid evidence |
| T1.4 median pairwise logging | Required — T0.1 could not recover a training series |
| T1.3 $f$ histograms | Unchanged; still required for E4 |
| E3 | Elevated: only remaining empirical test of §5.1 contraction |
| E5 | Blocked pending §4.2 / §6 rewrite after T0.2 |
| E1 | Use T0.3 protocol exactly |
| E2 | Still next after Tier 1; MICo’s identification role is untouched by T0.2 |
