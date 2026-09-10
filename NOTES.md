# Experiment log

## 2026-09-02 — ALE Stage B cancel + preliminary CTRO/LTRO collapse diagnosis — job 44918769 — cancelled (partial)

Cancelled the six still-running Stage B stress cells (CTRO/LTRO × 16 epochs × seeds 42–44) mid-resume to stop further GPU burn and diagnose why those arms collapse while PFO does not. At cancel they were at about 75–83M steps; all ep4 arms and all PFO ep16 arms already had finals.

Preliminary analysis on Phoenix ep16: PFO never meets the ret<5 and cumulant-NMSE>10 collapse rule; PPO collapses by about 4.5–6M steps; CTRO collapses on a similar schedule; LTRO only delays onset to about 8–12.5M steps then fails the same way. Configs show the Moalla-relevant feature is actor `actor_preactivation` on a separate NatureCNN trunk, while MICo, the Polyak–Łojasiewicz (PL) hinge, and \(D_Z\) all run on `critic.encode()` (critic 512-d features). Logged PL hinge loss is identically zero (ratios far above \(\mu_0\)), and mean \(D_Z\approx0.002\) versus \(\eta^2=0.0196\), so the fixed geometry term is not binding on the critic either. Prior Phase 0 cartpole-pixel evidence therefore does not transfer: that stack regularized a shared Impala critic latent path, not the separate actor representation Moalla/PFO protect. Next mechanistic check should apply geometry or PFO-style trust on `actor_preactivation` (or share trunks) rather than retuning \(\lambda\) on the critic.

## 2026-08-31 — ALE Stage B Phoenix resume incomplete cells — job 44918769 — cancelled (partial)

Resubmits the unfinished credit-reduced Stage B Phoenix cells from job **44570971** as array tasks `6,9-23` (sixteen cells). Finished `pfo_ep4` seeds 43–44 (tasks 7–8) are omitted. Each incomplete run still has `weights_latest.pt`, and the job script auto-resumes under `--resume` while skipping any cell that already has `weights_final.pt`. Remaining work is about 9–60M environment steps (ep4 nearer finish; ep16 stress arms still near mid-run). Expected result if throughput matches the first wave: all sixteen reach 100M within another two-day wall, after which Stage B analysis and figures can run with Stage A PPO as the baseline arm.

Analysis: twelve of sixteen resume tasks completed; the six CTRO/LTRO ep16 cells were cancelled on 2026-09-02 for diagnosis (see entry above).

## 2026-08-31 — ALE Stage A gate aggregation — no job — done

Ran `python -m src.experiments.analyze_phase1_ale_stage_a` on the finished Stage A checkpoints (all twelve cells at ~100M steps with `weights_final.pt`). Output: `results/ale/phase1_ale_stage_a_gate.json`. Phoenix qualifies with seed_hits=3/3: under 16 update epochs, final-window eval return collapses from about 26–30 to about 0.4–0.6, diagnostic cumulant normalized mean squared error (NMSE) rises by orders of magnitude, and at least two actor geometry diagnostics worsen in every seed. NameThisGame does not qualify (seed_hits=0): stress raises return (about 55–73 to about 95–98) even though cumulant NMSE explodes, so the joint return+NMSE+diagnostics pattern fails. Stage B aggregation is not runnable yet because job **44570971** left only two of eighteen Phoenix cells finished.

## 2026-08-27 — ALE Stage B reduced Phoenix matrix (PFO/CTRO/LTRO) — job 44570971 — timed out (incomplete)

Runs a credit-reduced Stage B on Phoenix only: PFO, CTRO, and LTRO-fixed × update epochs {4, 16} × seeds {42, 43, 44} (18 cells; array tasks 6–23). Plain PPO is not retrained; Stage A `exp_p1a_ppo_ep{4,16}` supplies the PPO arm for interaction \(I\) and method comparisons. NameThisGame Stage B is deferred until the Stage A gate finishes. Expected pattern if the hypothesis is right: under 16-epoch stress, PFO and especially LTRO-fixed should keep return and cumulant NMSE healthier than Stage A PPO, while the LTRO−PPO gap is larger at stress than at 4 epochs so \(I>0\); CTRO is the geometry-ablation between PPO and LTRO.

Analysis: only `exp_p1b_pfo_ep4` seeds 43 and 44 reached `weights_final.pt`. The other sixteen cells timed out between about 40M and 91M steps (stress arms especially short). Resume or longer walltime is required before Stage B analysis or figures.

## 2026-08-27 — ALE Stage B method matrix (both games) — jobs 44570738 / 44570740 — cancelled

Full 48-cell Stage B (both games × four methods including re-trained PPO) was cancelled before start to save GPU hours after Stage A showed a clear Phoenix collapse and unfinished NameThisGame cells.

## 2026-08-27 — ALE Stage A NameThisGame resume — job 44570605 — done

Resubmits Stage A array tasks 7–11 after they timed out on **44279818** and were not covered by resume job **44487871**. The five cells are NameThisGame with 4 update epochs (seeds 43–44) and NameThisGame with 16 update epochs (seeds 42–44). Each still has `weights_latest.pt` and will auto-resume under `--resume`; remaining work is about 15–52M environment steps. Walltime is two days (QOS rejected three). If resume works as intended, these complete the NameThisGame half of Stage A so the gate can run on both games.

Analysis: all five tasks completed; NameThisGame Stage A is now fully on disk with finals, enabling the 2026-08-31 gate.

## 2026-08-26 — ALE Stage A resume timed-out cells — job 44487871 — done

Resubmits array tasks 3–6 of the Stage A Moalla PPO screen after they hit the two-day wall on job **44279818**. The four cells are Phoenix with 16 update epochs (seeds 42–44) and NameThisGame with 4 update epochs (seed 42); each still has `weights_latest.pt` and will auto-resume under `--resume`. Remaining work is about 17–35M environment steps, which should finish inside another two-day allocation if throughput matches the first wave. If the hypothesis about walltime was right, these resumes complete without another timeout while tasks 7–11 of **44279818** continue independently.

Analysis: all four resume tasks completed without another timeout.

## 2026-08-23 — ALE Phase 1 Stage A PPO Moalla screen — job 44279818 — done

This Stage A array trains plain PPO on `ALE/Phoenix-v5` and `ALE/NameThisGame-v5` for 100M environment steps with update epochs in {4, 16} and paired seeds {42, 43, 44} (12 cells). The goal is to reproduce Moalla-style representation stress under a cheaper epoch dial than 32, using separate NatureCNN actor/critic, actor `actor_preactivation` features, and random-cumulant NMSE. If the collapse-regime hypothesis is right, stressed (16-epoch) runs will show worse or less stable return, worse cumulant NMSE, and at least two supporting actor diagnostics in at least two of three seeds while standard PPO still learns. Gate script: `src/experiments/analyze_phase1_ale_stage_a.py`. Freeze stamp remains `ltro_phase0_v1` without retuning. As of 2026-08-26: tasks 0–2 (Phoenix ep4) completed; tasks 3–6 timed out and were resumed as **44487871**; tasks 7–11 still running.

Analysis: the full twelve-cell matrix finished via resumes **44487871** and **44570605**. Gate result (2026-08-31): Phoenix qualifies; NameThisGame does not.

## 2026-08-23 — ALE Phoenix ≤2M smoke — job 44279817 — done

Mandatory short smoke before Stage A: Phoenix PPO for 2M steps with eval every 130 updates so we get at least 15 eval rows, plus on-policy and fixed-diag actor cumulant/geometry metrics. This checks that ALE wrappers, NatureCNN stacks, PFO plumbing (coef 0 here), freeze stamping, and metric columns exit cleanly on GPU before burning the 100M Stage A budget. CPU wire check `exp_p1_ale_cpu_wire8` already produced 15 eval rows with `diag_cumulant_nmse` present under the login memory cgroup.

## 2026-08-23 — Phase 0 freeze commit and tag ltro-phase0-v1 — no job — done

Locked LTRO-fixed before Phase 1 v2 collapse results: fixed geometry penalty \(\lambda=1\) adding \(D_Z\) to CTRO; \(\eta=0.14\) is monitoring-only because the loss is \(\lambda(D_Z-\eta^2)\) rather than a hinge. Adaptive dual control was rejected after job **44194678** because no dual radius met P0.3. Annotated tag `ltro-phase0-v1` points at commit `efcbef09246b38641b388b836944caed6190957d`; see `docs/phase0_freeze_note.md`.

## 2026-08-23 — Phase 1 v2 PPO epoch sweep (eval-schedule fix) — job 44236259 — done

The v2 sweep completed all 20 cells (`exp_p1v2_ppo_ep{4,8,16,32}` × seeds 42–46) with 15 eval checkpoints each, confirming the runner fix. Thresholds were fit from healthy ep4 and written to `configs/frozen/phase1_collapse_thresholds.json`. The predeclared collapse gate **failed**: zero seeds met the dual functional+geometry criterion for two consecutive evals at any epoch ≥8, so we should not launch the PPO/CTRO/LTRO-fixed confirmation matrix on this dial alone. At ep32, mean eval return fell to 129 (from ~187 at ep4) and seed 42 showed saturated \(C_{0.01}=1\) and dormant fraction 1.0 from ~200k steps while functional probe MSE stayed below the healthy threshold—geometry collapse without functional probe failure under the current definition.

## 2026-08-23 — Phase 0 radius selection finalize — job 44194678 — done

Dual-tight (\(\eta=0.05\)) and dual-medium (\(\eta=0.07\)) all five seeds completed. No adaptive dual arm met the P0.3 band (tight: late ratio good but \(\lambda\) almost always above floor and geometry grads dominate; medium/loose miss ratio and/or geom). Per the guide fallback we freeze **LTRO-fixed**: \(\eta=0.14\), \(\lambda=1\), `dz_adapt=false` in `configs/frozen/ltro_phase0_v1.json`. Adaptive dual control did not provide stable operational value on cartpole-pixels under these criteria.

## 2026-08-22 — Phase 1 PPO epoch stress sweep — job 44195299 — done

This run checks whether increasing PPO optimization epochs per rollout induces reproducible representation collapse on cartpole-swingup pixels before we add other methods. We train plain PPO for 1.5M steps with `--num-epochs` in {4, 8, 16, 32} and five paired seeds, logging functional probe MSE, dormant-unit fraction, and \(C_{0.01}\) at eval checkpoints. If the collapse-mechanism hypothesis is right, at least one stress level at or above 8 epochs will put half or more of the seeds into the predeclared collapse definition, with representation failure at or before return drop.

Analysis: all cells finished except ep32 seed44, which aborted near the end when a fully collapsed latent made \(s_{\mathrm{ref}}=0\). More importantly, probe and eval metrics appeared only on the final checkpoint because eval was nested under the step-interval logger, so mid-run eval epochs were skipped. Final-only snapshots are not usable for collapse-onset gating; see the 2026-08-23 v2 resubmit.

## 2026-08-22 — Phase 0 dual-tight/medium radius sweep — job 44194678 — done

This run tests whether an adaptive dual controller for the latent trust-region term \(D_Z\) stays inside the operational band at tighter radii than the Stage B loose dual (\(\eta=0.14\)). We train CTRO with adaptive \(\lambda\) at \(\eta=0.05\) (`exp_dzb_dual_tight`) and \(\eta=0.07\) (`exp_dzb_dual_medium`) on cartpole-swingup pixels for five paired seeds (42–46), 1.5M steps each. If the Phase 0 selection rule is right, at least one radius will keep \(\lambda\) off the upper bound, keep late \(D_Z/\eta^2\) near one, and avoid geometry-gradient dominance without using return to choose \(\eta\).

All ten array tasks completed (~8.5 h each). Selection is recorded in the 2026-08-23 finalize entry: no dual arm passed; freeze to LTRO-fixed.

## 2026-08-20 — reference-scaled D_Z calib fc_mu crash / resubmit — jobs 44000036 / 44000037 — done

Calib **43999499** failed on the first ownership step with `CNNEncoderCritic` has no attribute `fc_mu` (hint: `fc_z`). Ownership grad collection in `ppo._run_minibatch` hardcoded `fc_mu`; `_encoder_params` already gated `fc_mu` but omitted `fc_z`. Fixed by including `fc_z` in `_encoder_params` and routing ownership through that helper (same fix in `active_ctro`). Confirm **43999500** hit `DependencyNeverSatisfied`. Resubmitted calib **44000036** and confirm **44000037** (`afterok:44000036`). Both finished on disk: Stage A wrote \(\eta=0.14\) under `exp_ctro_dz_calib_sref`, and Stage B completed off / dual-loose (`exp_dzb_fixed`) / fixed-penalty (`exp_dzb_penalty`) for seeds 42–46. These arms are reused for Phase 0 radius selection; dual-tight and dual-medium are the remaining P0.2 cells.

## 2026-08-20 — reference-scaled D_Z calib crash / resubmit — jobs 43999499 / 43999500 — failed (calib)

Calib **43998468** failed immediately with `z_new/z_old shape mismatch (256, 128) vs (256, 256)`, so confirm **43998469** hit `DependencyNeverSatisfied`. Cause: `VAEEncoderTarget` only copied `fc_mu`, but pixel `CNNEncoderCritic` maps Impala emb (256) → latent (128) via `fc_z`, so the target path returned 256-d CNN features while the online encode path returned 128-d latents. Fixed by deepcopying and soft-updating `fc_z` in `bisimulation_utils.VAEEncoderTarget`. Resubmitted calib **43999499** and confirm **43999500**, which then failed on the `fc_mu` ownership bug above.

## 2026-08-20 — reference-scaled D_Z^log Stages A+B — jobs 43998468 / 43998469 — failed (calib)


Replaced batch-median log-ratio \(D_Z\) with a frozen reference scale \(s_{\mathrm{ref}}=\mathrm{median}\|z_i-z_j\|/\sqrt{d}\) and absolute \(\delta=0.01\) on scaled distances \(\bar d\). The dual term is now \(\lambda(D_Z^{\log}-\eta^2)\). Collapse is logged as \(C_{0.01}=\Pr[\bar d_{\mathrm{old}}<0.01]\). CPU geometry checks in `validate_dz_geometry` all passed. A frozen cartpole-pixels diagnostic batch was written to `results/dmcontrol_pixels/diag/cartpole-swingup/dz_diag.pt` and eval checkpoints log `diag_dz_C_0p01`. Calib job **43998468** (`exp_ctro_dz_calib_sref`, \(\lambda=0\)) failed on the CNN target-latent dim bug above; confirm **43998469** never ran.
## 2026-08-20 — reward-test pseudometric stages 1–5 — no MiniGrid claim job — done

Implemented the staged reward-test split: `D_hat_pair` / `D_hat_lower` logging, frozen-reference validity (unique hashes and old-distance quantiles), target-landscape PL, optimizer-ownership deltas, behavior-stream stats, `L_sep` on confident negatives, diagnostic `D_hat_sig`, toys T1–T5, and the post-gate MiniGrid matrix configs/jobs. CPU toys including T1–T5 passed via `python -m src.experiments.run_active_ctro_toys` (no Slurm job). E0 vanilla and anti-aliased PPO three-epoch smokes passed on CPU (`results/active_ctro/gates/g0`). Shared-ref diagnostics ran on CPU as `exp_diag_reference_ownership` (seed 42, 3 epochs). No new full Unlock active job was submitted.

Gate G0 holds: the live AA-PPO PL hinge, floor, and E-A/E-B criteria were not changed. Gate G1 is a documented cause, not a usable relational scale: the shared 256-row freeze has only 28 unique observations (duplicate fraction 0.89), median pairwise latent distance \(\approx 7\times 10^{-4}\) below `min_old_distance` \(=10^{-3}\), and covariance eigenvalues below the \(10^{-6}\) floor, which is why the finished 1500-epoch active package logged `dz_rel_all_excluded=1`. Gates G2–G4 passed on T1–T4. Gate G5: ownership logs show the actor still steps (non-zero `own_actor_delta`) while KL stays tiny and the query stream is about one-third of collected steps; the stall is the latent-pair scale / duplicate-ref geometry, not a frozen actor. Stage 5 matrix entry points exist (`exp_matrix_row`, `submit_active_ctro_matrix.sh`) and a CPU smoke of row 5 passed; they must not be used as theorem evidence until a later run starts from a reference set with a documented usable old-distance scale.

## 2026-08-20 — active CTRO MiniGrid E0/E6 finished — jobs 43866692 / 43908525 — done

Toys **43865880** stayed green. The first MiniGrid resubmit **43866692** finished vanilla PPO and anti-aliased PPO (array 0–5, ~3.5–4 h) and failed active CTRO (array 6–8) when every off-diagonal pair on the frozen reference batch had old latent distance below `min_old_distance` (\(10^{-3}\)). After that crash was changed to report `dz_rel_all_excluded` instead of raising, array **43908525** skipped the finished E0 checkpoints and completed the three active-CTRO seeds (~4.7 h). Aggregation **43908526** was still pending on RM-shared; the table was written on CPU from the finished checkpoints at `results/active_ctro/minigrid/tables/e0_e6_summary.txt`.

Final mean eval / train return (seeds 42–44): vanilla PPO \(0.314\pm0.169\) / \(0.883\pm0.006\); anti-aliased PPO \(0.364\pm0.091\) / \(0.932\pm0.010\); active CTRO \(0.000\pm0.000\) / \(0.180\pm0.034\). Active CTRO never left the early-training plateau (train return ~0.21 by epoch 200, then slightly worse), while both E0 locks rose through ~0.6–0.8 by epoch 500. The relational metric stayed fully excluded (`dz_rel_all_excluded=1`) so \(\widehat D_{Z,\infty}^B\) is undefined; policy KL stayed ~0.0007 versus ~0.013 (vanilla) and ~0.005 (anti-aliased). The MiniGrid hypothesis that active CTRO would beat both E0 locks is not supported on this Unlock package.

## 2026-08-19 — active CTRO MiniGrid resubmit — jobs 43866692 / 43866693 — failed (active arm)

The aggregate job **43865882** stayed pending with `DependencyNeverSatisfied` because
the MiniGrid array **43865881** failed (`afterok` requires all array tasks to succeed).
Toys **43865880** completed. All nine MiniGrid tasks crashed during metric logging:
`CTROMetricEvaluator.evaluate` received CPU rollout tensors while the critic was on
CUDA. Fixed in `runner.py` by passing `buffer_dev` tensors into metric evaluation.
Cancelled the dead agg job, resubmitted MiniGrid **43866692** and agg **43866693**
(`MINIGRID_ONLY=1`, toys already done). E0 arms of **43866692** completed; active CTRO
raised on an empty relational-pair set (see 2026-08-20 entry). Agg **43866693** was
cancelled when the array failed.

## 2026-08-19 — active CTRO experiment pipeline — jobs 43865880 / 43865881 / 43865882 — failed (MiniGrid)

Added a Bridges-2 pipeline for the active self-supervised CTRO experiments:
CPU toys (E1–E5 plus invariants) on RM-shared, MiniGrid Unlock E0/E6 as a
nine-cell GPU array (vanilla PPO, anti-aliased PPO, active CTRO × seeds 42–44),
then CPU aggregation. Submit with `bash src/experiments/jobs/submit_active_ctro.sh`.
The first submit failed because RM-shared caps memory at 2000M per core; `--mem=8G`
with four cores exceeded that. The scripts now follow the existing RM-shared
pattern (`--ntasks-per-node`, no explicit `--mem`). Resubmitted: toys **43865880**,
MiniGrid array **43865881** (afterok on toys), aggregate **43865882** (afterok on
MiniGrid). Toys completed; MiniGrid array failed on CPU/CUDA device mismatch in
metric eval (see resubmit entry above). Agg **43865882** cancelled.

## 2026-08-18 — active CTRO migration landing — no job — done

Implemented the active self-supervised CTRO architecture on top of the existing
PPO/CTRO stack without rewriting PPO: split actor and representation Adam optimizers,
transition replay with terminated/truncated metadata, dual PPO and query collection
streams, discrete Q ensemble with target copies, same-action action-conditioned MICo,
pair-confidence index, information critic U, and finite-reference relational
\(\widehat D_{Z,\infty}^B\) in observe mode. The Polyak–Łojasiewicz hinge remains
a live representation loss (anti-aliased PPO admissibility), distinct from
AC-MICo. E0 locks vanilla PPO and anti-aliased PPO with all new modules disabled.
CPU smoke runs passed for `validate_invariants`, E1–E5 toys, and three-epoch MiniGrid
smokes (`exp_e0_vanilla`, `exp_e0_aa_ppo`, `exp_active_ctro`, seed 42, device cpu).
No GPU job submitted; Procgen benchmarks remain gated until the toy package is
reviewed on longer MiniGrid runs.

## 2026-08-16 — anti-aliased PPO method switch — no job — pending

The live training recipe is now proximal policy optimization with ratio clipping
plus a single scale-corrected Polyak–Łojasiewicz hinge on a frozen reference batch
of states (anti-aliased PPO). The previous Causal Trust Region Optimization stack
(MICo metric loss, spectral normalization on the value head, and the latent-distance
trust region \(D_Z\)) remains in the codebase behind `--algo-preset ctro_full_legacy`
and is documented for recovery in `docs/METHOD_CHANGELOG.md`. Blocking provenance
checks B2 and B3 on the four-arm E2 logs are recorded there: E2 already used the
reference-level value denominator and logged scale-corrected \(\tilde\mu_{PL}\) and
latent length \(L\). Experiment order and pre-declared hypotheses are in
`docs/next_round_experiments.md`. No GPU job is submitted in this entry; E-A must
gate E-B.

**Implementation landed (same date).** Frozen PL reference buffer, clamp denominator,
`--algo-preset {anti_aliased_ppo,ctro_full_legacy}`, CPU shear tool
(`shear_gauge_experiment`), and gated Slurm templates
`ea5_resume_shear_s.sh`, `eb_aa_ppo_vs_baseline_s.sh`, `eb_procgen_aa_ppo_s.sh`,
`ed_contraction_s.sh`, `b4_baseline_walker_cheetah_s.sh`. B1 on E2 PL-tilde seed 42
passed (`gate_b1_ok`; PR≈7.6). B4 cartpole `exp_baseline_v3` passes (>800); walker
and cheetah baselines remain open. E-A.1 smoke on that checkpoint passed behavioural
invariance; do not submit E-B until E-A.5 supports H1.

## 2026-08-15 — Tier 1 diagnostics + E2/E1/E4 submissions — jobs 43614235 / 43614236 / 43614237 — done

Implemented Tier 1 logging and surrogate fixes before the next GPU wave: scale-invariant
$\tilde\mu_{PL}=\mu_{PL}L^{2}$ and value-normalized $\tilde\mu_{PL}/V_{\mathrm{scale}}$
in `compute_mu_pl_bootstrap` (logged alongside raw $\mu_{PL}$); optional
`--pl-scale-invariant` / `--pl-value-normalize` for the PL hinge; per-metrics-dump
`latent_pair_p{05,50,95}` series (T1.4); existing per-dump `f_gap_hist/` retained
(T1.3); `--value-spectral-norm` for E2 arm C. CPU geometry validation and a
scale-invariance smoke for $\tilde\mu_{PL}$ both passed. Submitted
`e2_run3_pixels_cartpole_s.sh` as job **43614235** (12 cells),
`e1_dz_multitask_pixels_s.sh` as **43614236** (40 cells), and
`e4_rate_pixels_cartpole_s.sh` as **43614237** (15 cells).

All sixty-seven cells finished with `status=ok`. **E2 (method identity):** on the
primary metric (fifth-percentile pairwise distance), full CTRO did **not** beat
PL-plus-spectral-norm; the paired difference was negative in all three seeds
(IQM $-1.69$, 95% CI $[-2.96,-0.41]$). PL-only with the scale-invariant hinge
had the highest return and by far the largest distances. That fails the
pre-declared “MICo earns its keep via identification on p05” test; spectral
normalization is at least competitive for the smoothness role, and the
bisimulation target’s third role is not supported here. **E3:** PL-only on the
non-VAE pixel stack did contract median pairwise distance over training (late/early
ratios $0.13$–$0.36$), but return rose rather than failing to follow, so the
§5.1 “contraction hurts return” story is only half confirmed. **E1:** pooled
across four DeepMind Control pixel tasks, fixed $D_Z$ **lowered** p05 relative
to off (stratified IQM $\Delta=-0.141$, CI $[-0.219,-0.055]$; only $6/20$
seed-task pairs positive). The cartpole Run 2 aliasing-protection claim did not
replicate as a multi-task mechanism; return leaned up (walker significant) but
the geometric lead result flipped sign. **E4:** $\tilde\mu$ versus $\eta$ was
non-monotone and positively correlated with $\eta$; linear SSE beat $\sqrt\eta$,
but neither is a credible confirmation of Theorem 2. File the rate test as a
mismatch that points at missing terms (value-drift) rather than as empirical
support. E5 remains blocked.

## 2026-08-15 — Tier 0 conceptual checks (T0.1–T0.3) — no job — done

Zero-compute checks from `docs/next_round_experiments.md`, reported in
`docs/t0_findings.md`. T0.1 separated the Minigrid two-by-two cells and recomputed
endpoint median pairwise latent distance on a shared probe buffer: PL-only does not
contract relative to baseline or full CTRO on the VAE stack, while late return rises
with $\mu_{PL}$, so the §5.1 contraction prediction is not confirmed on Minigrid and
E3 on non-VAE stacks becomes decisive. T0.2 checked the §4.2 sandwich against Castro
et al.: the MICo value bound holds for the diffuse potential $U^\pi$, not for reduced
embedding distance, so §4.2 was rewritten and E5 stays blocked. T0.3 froze the Run 2
analysis protocol (paired IQM + stratified bootstrap) in
`docs/run2_analysis_protocol.json` for reuse in E1; pairwise p05 remains the lead
confirmed geometric effect.

## 2026-08-14 — D_Z off versus fixed, five seeds — job 43539537 — done

This experiment compared Causal Trust Region Optimization with the latent-distance
constraint \(D_Z\) disabled against the same optimizer with a fixed calibrated
\(D_Z\) radius, using cartpole-swingup seeds 42–46 for 1.5 million environment
steps. We ran it to test whether the single-seed rank and return effects reproduced;
if the constraint prevented global representation collapse, we expected higher
participation ratio (the effective covariance rank), larger lower-tail pairwise
distances, and higher return in the fixed arm.

All ten cells completed. Participation ratio did not separate the arms
(off 1.86 ± 0.21; fixed 1.88 ± 0.16), so the single-seed rank-collapse
interpretation was rejected. The fifth percentile of pairwise latent distance,
which measures local state distinguishability, increased in every matched seed
(mean 0.143 to 0.419; paired-bootstrap interquartile-mean difference 95% interval
[0.041, 0.561]). Last-50 return increased from 112.8 to 163.0 on average and in
four of five seeds, but its paired-bootstrap interquartile-mean difference interval
included zero ([-4.9, 103.1]). The supported mechanism is protection against local
state aliasing, not preservation of global feature rank; return is corroborating
rather than the lead result at five seeds.

The policy proximal-policy-optimization clip, which is the KL-like policy trust
region used here, remained active in both arms. This is therefore the Run 2
additional-constraint claim; Run 1, where \(D_Z\) replaces that policy constraint,
remains untested.

## 2026-08-14 — Run 1 pilot (no-clip η sweep + stress) — job 43566262 — done

Submitted `sbatch src/experiments/jobs/dz_run1_pilot_s.sh` to test whether a latent
distance trust region can replace the PPO policy clip on cartpole-swingup seed 42.
The pilot includes three fixed \(\eta\) values (0.101 calibrated under clip; 0.32 mid;
0.71 matching early unconstrained \(D_Z\)) plus a required stress arm with clipping
and \(D_Z\) both off, so a Run-1 success cannot be read as “clipping was unnecessary.”
Pre-declared explosion / bounded-loose / bounded-tight / stress-succeeds outcomes are in
`docs/dz_run1_pilot_criteria.md`, using the Run 2 fixed KL distribution
(median 0.440, p95 2.383, p99 5.871). A negative is consistent with theory; a positive
would motivate a follow-up lemma on bounded policy KL under bounded latent distortion.

All four array tasks completed (`status=ok`, 1.5M steps). By the pre-declared rule,
every arm is an **explosion**: run-wide median policy KL exceeded Run 2 p99 (5.871),
and more than 1% of logged updates exceeded \(3\times\) that threshold (17.61). Medians
were stress 9.61, \(\eta=0.101\) 8.36, \(\eta=0.32\) 24.38, \(\eta=0.71\) 7.40; the
fraction of updates with KL above 17.61 was 15–61%. Last-50 mean episode return was
stress 86.1, \(\eta=0.101\) 74.8, \(\eta=0.32\) 159.8, \(\eta=0.71\) 87.8, so mid-\(\eta\)
kept return near the Run 2 fixed mean while still exploding on KL. Stress did not
succeed alone (return below 129.2 and KL explosion). File as: latent \(D_Z\) did not
replace the policy trust region in this configuration; the negative matches the
theory expectation from gauge freedom under \(D_Z\). No \(\eta\)-ambiguity exception
(only the clip-calibrated arm exploding) applies.
