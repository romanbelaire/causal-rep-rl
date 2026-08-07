# Matched parity, on-Z stress, and DMControl pixels — run report

**Date of analysis:** 2026-08-07  
**Suite roots:** `results/dmcontrol_state/`, `results/dmcontrol_pixels/`  
**Machine summary CSV:** [`results/parity_stress_pixels_summary.csv`](results/parity_stress_pixels_summary.csv)

Metrics below are from train logs (`metrics.csv` columns `mean_episode_return`, `mu_pl_q05`, `feature_rank_participation_ratio`), final logged value for finished runs. Budget is 8M env steps (`total_epochs = 8_000_000 // 2048`), seeds `{42, 43, 44}`, serial `num_envs=1`.

---

## 1. What we did

After fixing continuous PPO (action/log-prob ranks under `num_envs=1`, terminated vs truncated, action clip, state-independent `log_std`) and running Optuna on DMControl state, we built and ran a three-arm recipe:

| Arm | `EXP_NAME`s | Purpose |
|-----|-------------|---------|
| **Shared parity** | `exp_shared_baseline`, `exp_shared_latent_nolink`, `exp_shared_ctro` | Same Optuna t8 HPs for all methods; fair 8M×3 seed comparison |
| **On-Z stress** | `exp_stress_latent_nolink`, `exp_stress_ctro` | Harder policy-on-Z schedule to hunt μ_PL / return failure vs CTRO protection |
| **DMControl pixels** | `exp_baseline`, `exp_latent_nolink`, `exp_ctro` under suite `dmcontrol_pixels` | Same four tasks with 84×84 RGB so representation learning is forced |

### Shared hyperparameters (CTRO Optuna trial 8)

| Knob | Shared parity | Stress |
|------|---------------|--------|
| `learning_rate` | `1.0128e-4` | `5e-4` |
| `entropy_coef` | `0.0408` | `0` |
| `num_epochs` | `20` | `40` |
| `policy_hidden` | `256,256` | `256,256` |
| CTRO `α`, `β` | `0.00205`, `0.532` | same |
| CTRO warmup | `alpha/beta_warmup_epochs=500` (~1M steps) | none |
| latent_nolink `α`, `β` | `0`, `0` | `0`, `0` |

Primary **on-Z control is latent_nolink** (same stack as CTRO, value link off). Baseline PPO is a methodological reference (`π(a|s)`), not architecture-matched.

### Code / infra shipped for this recipe

- Job launchers: `perf_shared_hps_dmcontrol_s.sh`, `perf_stress_onz_dmcontrol_s.sh`, `perf_train_dmcontrol_pixels_{baseline,latent_nolink,ctro}_s.sh`
- α/β linear warmup in `src/agents/ctro.py` + CLI flags
- `DMControlPixelWrapper` + suite `dmcontrol_pixels` (non-VAE `ctro_cnn` encoder stack)
- Continuous Impala / latent Impala: state-independent `log_std` + correct `get_action` squeeze
- Panel presets: `dmcontrol_state_shared`, `dmcontrol_state_stress`, `dmcontrol_pixels` in `plot_performance_panels.py`
- Recipe notes in `EXPERIMENTS.md`

### Jobs (submit chronology)

| Job / array | Contents | Notes |
|-------------|----------|-------|
| `230934` | Shared HPs, 12 tasks × 3 seeds | First wave; completed |
| `231516` | Stress, 8 tasks × 3 seeds | Second wave |
| `231517` | Pixels baseline | Second wave; many seed OOMs |
| `231944`, `231945` | Pixels latent_nolink / CTRO | Submitted after QOS headroom |

Submission was staggered by `QOSMaxSubmitJobPerUserLimit` / `QOSMaxJobsPerUserLimit`.

---

## 2. Why we did it

Desired publishable shape:

1. **Normal regime (state, matched HPs):** CTRO ≈ latent_nolink ≈ baseline on return, with CTRO keeping healthy μ_PL.
2. **Failure regime:** stressed policy-on-Z where latent_nolink μ_PL and return drop while CTRO holds.
3. **Representation-sensitive domain:** same DMC tasks with pixels so Z is not already Markov physics state.

Background that motivated the design:

- Pre-fix continuous PPO under `num_envs=1` corrupted action/log-prob ranks; hopper and other tasks were unreliable.
- Post-fix Optuna showed CTRO preferred **tiny α** (~0.002); large constant α hurt return.
- Diagnostics often showed encoder **not collapsing** under CTRO (high μ_PL, PR); the gap vs PPO was more a **policy-on-Z + MICo tax** than pure collapse.
- Therefore **latent_nolink** is the right control for “does the value link help,” not raw-obs PPO.

---

## 3. Completion status

| Experiment | Metrics CSVs | `weights_final.pt` |
|------------|-------------:|-------------------:|
| `exp_shared_baseline` | 12/12 | **12/12** |
| `exp_shared_latent_nolink` | 12/12 | **12/12** |
| `exp_shared_ctro` | 12/12 | **12/12** |
| `exp_stress_latent_nolink` | 12/12 | **9/12** (all 3 cheetah pruned) |
| `exp_stress_ctro` | 12/12 | **9/12** (all 3 cheetah pruned) |
| pixels `exp_baseline` | 11/12 | **4/12** |
| pixels `exp_latent_nolink` | 11/12 | **4/12** |
| pixels `exp_ctro` | 12/12 | **4/12** |

**Shared:** complete.  
**Stress cheetah:** all 6 runs (3 seeds × 2 agents) **pruned** by return-collapse floor (`mean_ret < 1.0` for 3 log intervals).  
**Pixels:** majority failed with **CUDA OOM** while multiple seeds shared one GPU (`MAX_PARALLEL=3` on CNN-sized models). Failures often cite another process holding ~41 GiB on a ~44 GiB device.

---

## 4. What happened — state shared parity (main result)

Final train return / μ_PL_q05 / PR (mean ± std over 3 seeds).

### cartpole-swingup

| Method | Return | μ_PL_q05 | PR |
|--------|--------|----------|-----|
| baseline | **858.6 ± 1.0** | 1556 ± 933 | 2.43 ± 0.48 |
| latent_nolink | 634.0 ± 106 | 66 ± 49 | 3.67 ± 0.15 |
| CTRO | 612.2 ± 276 | 458 ± 184 | 3.90 ± 0.31 |

### cheetah-run

| Method | Return | μ_PL_q05 | PR |
|--------|--------|----------|-----|
| baseline | **763.4 ± 52** | 31.0 ± 6.2 | 4.60 ± 0.62 |
| latent_nolink | 631.6 ± 20 | 12.2 ± 3.8 | 10.9 ± 0.8 |
| CTRO | 610.4 ± 121 | **49.9 ± 21.8** | 9.44 ± 1.7 |

### walker-walk

| Method | Return | μ_PL_q05 | PR |
|--------|--------|----------|-----|
| baseline | **849.2 ± 117** | 22.6 ± 28 | 3.69 ± 0.61 |
| latent_nolink | 621.9 ± 251 | 6.6 ± 1.7 | 8.34 ± 2.6 |
| CTRO | 801.7 ± 191 | **35.7 ± 11** | **13.4 ± 0.9** |

### hopper-hop

| Method | Return | μ_PL_q05 | PR |
|--------|--------|----------|-----|
| baseline | **158.3 ± 96** | 191 ± 148 | 2.26 ± 0.95 |
| latent_nolink | 2.6 ± 3.3 | 872 ± 565 | 7.86 ± 1.8 |
| CTRO | 1.4 ± 1.0 | 1797 ± 428 | 3.33 ± 0.71 |

### Interpretation (shared)

- **Not parity with baseline.** On cartpole, cheetah, walker, raw-obs PPO clearly wins return. On-Z methods sit ~15–30% lower on cartpole/cheetah; walker is closer for CTRO but seed-noisy.
- **CTRO ≈ latent_nolink on return** (within noise), but **CTRO holds higher μ_PL** on cheetah and walker (and cartpole vs nolink). That matches “value link geometries Z” more than “value link alone matches PPO return.”
- **Hopper is a failure for all policy-on-Z at these HPs.** Both latent_nolink and CTRO end near-zero hop return while baseline reaches ~150–200. High μ_PL on failed hopper runs is **not** predictive of good locomotion here (PL can look “healthy” while the policy does not hop).
- Seed variance is large for CTRO cartpole (seed 44 ≈ 294 return) and walker/cheetah CTRO.

**Bottom line for the “normal regime” goal:** matched t8 HPs do **not** deliver CTRO ≈ baseline. They do show CTRO can match latent_nolink return while improving PL metrics on some tasks—useful for the value-link narrative, but the **PPO tax remains**.

---

## 5. What happened — on-Z stress

Stress knobs: `lr=5e-4`, `entropy=0`, `epochs=40`, torso `256,256`; CTRO keeps t8 α/β without warmup.

### Finished tasks (final returns mean ± std, n=3)

| Task | latent_nolink return | CTRO return | latent_nolink μ_PL | CTRO μ_PL |
|------|---------------------:|------------:|-------------------:|----------:|
| cartpole-swingup | 184.7 ± 47.5 | 160.5 ± 45.9 | 17.3 ± 10.3 | **316.6 ± 78.6** |
| walker-walk | 28.7 ± 14.1 | 43.9 ± 26.9 | high/noisy | ~36.5 (varies) |
| hopper-hop | ~1.5 | ~0.04 | high | very high (incl. outlier 5e4) |

(Shared-regime cartpole was ~600–850; stress cartpole is ~120–220 for all on-Z agents.)

### cheetah — all pruned (collapse)

| Agent | Seeds | Approx failure step | Reason |
|-------|-------|---------------------|--------|
| latent_nolink | 42, 43, 44 | ~0.44–0.65M | return collapse floor `<1.0` |
| CTRO | 42, 43, 44 | ~0.59–2.16M | same floor; CTRO occasionally lasted longer then still died |

CTRO seed 42 cheetah showed exploding PPO KL in logs before prune (training instability under stress schedule, not a clean “geometry protects return” story).

### Interpretation (stress)

- Stress **did** break policy-on-Z: returns collapsed vs shared on every task; cheetah was pruned for **both** agents.
- **CTRO did not “hold” return** relative to latent_nolink on cartpole; walker favors CTRO only weakly and both are catastrophic vs shared (~30–40 vs ~600–900).
- CTRO **did keep much higher μ_PL** on cartpole under stress—geometry metric moves without return rescue under this schedule.
- Success criterion from the plan (“latent_nolink μ_PL↓ + return↓; CTRO holds both”) is **not met** for this stress knob set. Stress is too harsh / wrong axis (lr↑ + epochs↑ + ent=0 causes PPO train blow-up, not a mild geometry collapse regime).

---

## 6. What happened — DMControl pixels

### Infrastructure outcome

Most runs died with **CUDA OOM** because three Impala/CNN training processes shared one GPU. OOM notes explicitly point at a co-resident process using ~41 GiB.

Finished 8M runs (only):

| Method | Finished seed/task cells |
|--------|---------------------------|
| baseline | seed 44: all 4 tasks |
| latent_nolink | seed42 cartpole, cheetah, hopper; seed43 walker |
| CTRO | seed44 cartpole, cheetah, walker; seed43 hopper |

### Finished returns (incomplete design; do not treat as multi-seed science)

| Method | Task | Seed | Return | μ_PL_q05 |
|--------|------|------|-------:|---------:|
| baseline | cartpole | 44 | 171.7 | 11.0 |
| baseline | cheetah | 44 | 145.6 | 1.1 |
| baseline | walker | 44 | 134.0 | 9.7 |
| baseline | hopper | 44 | 1.9 | 1367 |
| latent_nolink | cartpole | 42 | 160.1 | 1.01 |
| latent_nolink | cheetah | 42 | 108.3 | 0.001 |
| latent_nolink | walker | 43 | 144.3 | 0.0004 |
| latent_nolink | hopper | 42 | 0.03 | 0.97 |
| CTRO | cartpole | 44 | 78.7 | 22.6 |
| CTRO | cheetah | 44 | 69.4 | 2.2 |
| CTRO | walker | 44 | 99.6 | 4.9 |
| CTRO | hopper | 43 | 1.8 | 662 |

Pixel returns are far below state PPO even when jobs finish. Latent_nolink μ_PL is near-zero on finished locomotion tasks; CTRO μ_PL is higher but with tiny n and OOM-biased survivors.

**Bottom line for pixels:** experiment is **invalid as a multi-seed comparison** until re-run with one train process per GPU (`MAX_PARALLEL=1`) or larger VRAM allocations.

---

## 7. Synthesis vs original goals

| Goal | Verdict |
|------|---------|
| Shared HPs three-way complete | **Yes** — all 36 state runs finished |
| CTRO ≈ latent_nolink ≈ baseline (return) | **No** — baseline ahead; CTRO ≈ nolink on return |
| CTRO healthier μ_PL in normal regime | **Partial** — yes on cheetah/walker (and cartpole vs nolink); hopper μ_PL high for bad policies |
| Stress: CTRO holds when nolink fails | **No** for this stress recipe; both crash; cheetah both pruned; geometry can remain high while return dies |
| Same-task pixels representation comparison | **Not yet** — mostly OOMs; requeue with `MAX_PARALLEL=1` |

Supporting diagnostics that still hold:

- Policy-on-Z is a real tax on DMControl **state** relative to feedforward PPO, even after continuous-PPO fixes.
- Tiny α + β warmup does not close the return gap to baseline.
- Hopper on-Z remains fragile under these configs; μ_PL alone is a poor success proxy when locomotion rewards are sparse or zero.

---

## 8. Recommended next steps

1. **Pixels requeue:** `MAX_PARALLEL=1` (and/or one seed per SLURM array task) so CNN training does not share a 40+ GB GPU; clear failed dirs without `weights_final.pt` first if skip logic only checks finals.
2. **Do not treat stress arm as publishable** until redesigned — e.g. milder lr/epoch stress, disable return-collapse prune for stress cheetah, or stress representation noise rather than PPO optimizer aggression.
3. **Hopper / on-Z story:** either specialized HPs, longer entropy schedule, or report baseline-only for hopper locomotion and use other tasks for three-way geometry.
4. **Panels** (once cleaned):  
   `python -m src.experiments.plot_performance_panels --suite dmcontrol_state_shared --task walker-walk`  
   (stress/pixels incomplete or pruned.)
5. If seeking **return parity with baseline**, need further tuning or architectural changes beyond matched t8 + tiny α; current evidence is “CTRO regulates Z metrics vs latent_nolink without matching PPO return.”

---

## 9. Artifact paths

```
results/dmcontrol_state/exp_shared_{baseline,latent_nolink,ctro}/seed_{42,43,44}/{task}/
results/dmcontrol_state/exp_stress_{latent_nolink,ctro}/seed_{42,43,44}/{task}/
results/dmcontrol_pixels/exp_{baseline,latent_nolink,ctro}/seed_{42,43,44}/{task}/
results/parity_stress_pixels_summary.csv   # machine table used for this report
```

Launchers: `src/experiments/jobs/perf_shared_hps_dmcontrol_s.sh`,  
`perf_stress_onz_dmcontrol_s.sh`,  
`perf_train_dmcontrol_pixels_*_s.sh`.
