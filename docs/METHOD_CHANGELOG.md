# Method change log

Recoverability record for the live training recipe. Chat history is not the
record of record. Each entry states what changed, why, and how to restore the
previous recipe without code archaeology.

Companion: [`next_round_experiments.md`](next_round_experiments.md) (run list),
[`NOTES.md`](../NOTES.md) (job IDs).

---

## 2026-08-18 — active CTRO architecture (PL kept, AC-MICo added)

**What changed.** Dual-stream collection, transition replay, discrete Q
ensemble, same-action MICo, pair-confidence index, query information critic,
and finite-reference relational \(\widehat D_{Z,\infty}^B\). Actor and
representation Adam are split. The Polyak–Łojasiewicz hinge remains a live
representation loss (anti-aliased PPO admissibility). It is not set to zero.

**Why.** v1 claim is an approximate action-conditioned control abstraction on
reachable support, not the rejected random-pair MICo identification story.

**How to restore anti-aliased PPO (E0).** `--algo-preset anti_aliased_ppo` or
`python -m src.experiments.exp_e0_aa_ppo --seed 42`. Vanilla PPO:
`python -m src.experiments.exp_e0_vanilla --seed 42`.

**How to run active CTRO.** `python -m src.experiments.exp_active_ctro --seed 42`.

---

## 2026-08-16 — switch to anti-aliased PPO (scale-corrected PL hinge only)

**What changed.** The live method is PPO (ratio clipping on) plus a single
one-sided Polyak–Łojasiewicz (PL) hinge on the encoder, using the scale-corrected
diagnostic \(\tilde\mu = L^2\|\nabla_Z V\|^2/(2f)\). MICo, spectral normalization
on the value head, and the latent trust region \(D_Z\) are removed from the live
recipe (code paths retained behind config).

**Why.** The four-arm comparison rejected the bisimulation term against its
pre-declared retention criterion and rejected spectral normalization by
dominance. The multi-task \(D_Z\) test rejected the hypothesis that the latent
trust region increases separation. Surviving theory: impossibility proposition,
invariance-group proposition, and the two transfer theorems.

**Plumbing changes (same date).**

| Item | Before (legacy CTRO) | After (anti-aliased PPO) |
|---|---|---|
| Aux losses | MICo + PL (+ optional \(D_Z\)) | PL only (`alpha=0`, `dz_enabled=False`) |
| PL states | On-policy minibatch latents | Frozen reference batch (`N=2048`, never refreshed) |
| Denominator \(f\) | Exclude states with \(f\le\tau\) (relative floor) | `clamp_min(f_min)` with `f_min=1e-3` |
| Scale correction | Optional `--pl-scale-invariant`; E2 also used value-normalize | Always \(L^2\) in the hinge; no value-range division in the loss |
| Head phasing | Optional / off in E2 | Required: `E_enc=4`, `E_val=4` |

**How to restore the old CTRO recipe.**

```bash
python -m src.experiments.run_performance_train \
  --suite dmcontrol_pixels --task cartpole-swingup --seed 42 \
  --exp-name exp_ctro_full_legacy --agent ctro \
  --algo-preset ctro_full_legacy
```

Optional \(D_Z\) on top of legacy: add `--dz-enabled --eta-dz <calibrated>`.
Spectral arm of E2: `--algo-preset ctro_full_legacy` is not required; use explicit
flags `--alpha 0 --beta 0.532 --pl-scale-invariant --pl-value-normalize
--value-spectral-norm` as in `e2_run3_pixels_cartpole_s.sh`.

**How to run the live method.**

```bash
python -m src.experiments.run_performance_train \
  --suite dmcontrol_state --task cartpole-swingup --seed 42 \
  --exp-name exp_anti_aliased_ppo --agent ctro \
  --algo-preset anti_aliased_ppo
```

**Prior-result disposition (do not re-run unless noted).**

| Result | Disposition |
|---|---|
| Rank fails to separate | Keep (field-metric claim) |
| Run 1: \(D_Z\) cannot replace policy clip | Keep as appendix |
| Four-arm (E2): arm A strongest | Provisional appendix after B2/B3 |
| Contraction on non-VAE PL-only | Input to E-D |
| Multi-task \(D_Z\) | Re-analyze logs for degradation slope; do not rerun |
| Rate sweep | Discard as Theorem-2 test (estimator gap) |

---

## Blocking provenance (B2 / B3) — 2026-08-16

**B2.** E2 (`e2_run3_pixels_cartpole_s.sh`, job 43614235) used the reference-level
denominator (`v_ref` EMA of the 0.99 quantile), not the earlier Bellman-residual
denominator. Metrics columns include `v_ref`, `train_v_ref`, `f_*`. Arm A is the
scale-corrected PL hinge on that denominator; numbers transfer as that recipe,
not as the older Bellman form.

**B3.** E2 logged both raw `mu_pl_*` and scale-corrected `mu_pl_tilde_*`, plus
`latent_pair_L` / `latent_pair_p{05,50}`. Arm median pairwise distance \(L\)
differed by roughly \(3\times\)–\(20\times\) across arms (PL-tilde \(L\sim17\)–\(37\),
full CTRO \(L\sim0.85\)–\(2.2\)). Cross-arm comparisons of **uncorrected** \(\mu_{PL}\)
or raw distances are not interpretable; use \(\tilde\mu\) or \(L\)-normalized
separation.

Full B1–B4 write-up: [`blocking_checks_aa_ppo.md`](blocking_checks_aa_ppo.md).
