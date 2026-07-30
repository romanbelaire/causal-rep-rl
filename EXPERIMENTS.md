# Experiment Configurations

This document describes the 4 experiment setups and their corresponding configs and SLURM scripts.

## Experiment 1: Vanilla Baselines

### 1.1 PPO + IMPALA + MLP Critic
- **Config**: `configs/exp1_vanilla_ppo_impala_mlp.json`
- **SLURM**: `slurm/exp1_vanilla_ppo_impala_mlp.sh`
- **Description**: Standard PPO baseline with IMPALA policy and feedforward MLP critic
- **Architecture**:
  - Policy: IMPALA (2 residual blocks)
  - Critic: Feedforward MLP (256, 256)
- **Algorithm**: PPO (clip_epsilon=0.2)

### 1.2 TRPO + IMPALA + MLP Critic
- **Config**: `configs/exp1_vanilla_trpo_impala_mlp.json`
- **SLURM**: `slurm/exp1_vanilla_trpo_impala_mlp.sh`
- **Description**: Standard TRPO baseline with IMPALA policy and feedforward MLP critic
- **Architecture**:
  - Policy: IMPALA (2 residual blocks)
  - Critic: Feedforward MLP (256, 256)
- **Algorithm**: TRPO (max_kl=0.01)

---

## Experiment 2: Representational Baselines

### 2.1 PPO + IMPALA + VAE Critic
- **Config**: `configs/exp2_repr_ppo_impala_vae.json`
- **SLURM**: `slurm/exp2_repr_ppo_impala_vae.sh`
- **Description**: PPO with VAE-based critic for causal representation learning
- **Architecture**:
  - Policy: IMPALA (2 residual blocks)
  - Critic: VAE (latent_dim=32, beta=1.0)
- **Algorithm**: PPO (clip_epsilon=0.2)

### 2.2 TRPO + IMPALA + VAE Critic
- **Config**: `configs/exp2_repr_trpo_impala_vae.json`
- **SLURM**: `slurm/exp2_repr_trpo_impala_vae.sh`
- **Description**: TRPO with VAE-based critic for causal representation learning
- **Architecture**:
  - Policy: IMPALA (2 residual blocks)
  - Critic: VAE (latent_dim=32, beta=1.0)
- **Algorithm**: TRPO (max_kl=0.01)

---

## Experiment 3: Representation-Space Trust Region (RSTR)

### 3.1 RSTR + IMPALA + ICNN Critic
- **Config**: `configs/exp3_rstr_impala_icnn.json`
- **SLURM**: `slurm/exp3_rstr_impala_icnn.sh`
- **Description**: Representation-space trust region using ICNN critic (enforces convexity)
- **Architecture**:
  - Policy: IMPALA (2 residual blocks)
  - Critic: ICNN (mu=0.1 for strong convexity, exponential positivity)
- **Algorithm**: TRPO (max_kl=0.01) - Note: Full RSTR algorithm can be implemented later
- **Metrics**: Hessian spectrum enabled (for convexity analysis)

**Note**: Currently uses TRPO with ICNN critic. The ICNN enforces input convexity which is key for representation-space trust regions. Full RSTR algorithm implementation (with gradient/Hessian thresholding) can be added to the codebase.

---

## Experiment 4: Representational RSTR (RRSTR)

### 4.1 RRSTR + IMPALA + VAE Critic + Strict Clipping
- **Config**: `configs/exp4_rstr_impala_vae_strict.json`
- **SLURM**: `slurm/exp4_rstr_impala_vae_strict.sh`
- **Description**: Representational RSTR with VAE critic and strict policy clipping
- **Architecture**:
  - Policy: IMPALA (2 residual blocks)
  - Critic: VAE (latent_dim=32, beta=1.0)
- **Algorithm**: PPO with strict clipping (clip_epsilon=0.1, tighter than standard 0.2)

---

## Experiment 8: VAE strict Minigrid — value head comparison (warmup400)

All runs match [`configs/exp8_rstr_impala_vae_strict_warmup400_minigrid.json`](configs/exp8_rstr_impala_vae_strict_warmup400_minigrid.json) (PPO, `representation_loss_coef_warmup_epochs: 400`, `latent_dim: 8`) except the value head.

| Run | Config | Value head | μ w.r.t. encoder latent Z |
|-----|--------|------------|---------------------------|
| Baseline | `exp8_rstr_impala_vae_strict_warmup400_minigrid.json` | affine `V(Z)=wᵀZ+b` | μ ≡ 0 |
| Quadratic (free A) | `exp8_rstr_impala_vae_strict_warmup400_quadratic_latent_minigrid.json` | `V(Z)=ZᵀAᵀAZ+bᵀZ+c` | `μ_latent_analytic = 2σ_min(A)²` |
| Quadratic (μ_min floor) | `exp8_rstr_impala_vae_strict_warmup400_quadratic_latent_mumin_minigrid.json` | same + `A ← A_free + √(μ_min/2)·I` | floor ≥ 0.02 at init |
| Squared norm | `exp8_rstr_impala_vae_strict_warmup400_squared_norm_minigrid.json` | `V(Z)=‖f(Z)‖²` (MLP `f`) | Jacobian proxy + autodiff (logged) |

**μ validation:** For `quadratic_latent*`, training fails at startup if `μ_latent_analytic` ≠ `μ_latent_autodiff` (no nonlinear layer between Z and the quadratic form). Squared norm logs both but does not enforce equality.

**Launch:**
```bash
bash scripts/run_exp8_warmup_valueheads_minigrid.sh   # interactive
sbatch exp8_warmup_valueheads_minigrid.sh              # Slurm (3 jobs sequential)
```

Legacy `quadratic_bottleneck*` configs are deprecated (μ was w.r.t. an internal bottleneck, not Z).

---

## Minigrid 2×2 CTRO ablation (primary ablation)

The controlled ablation for the central claim runs on Minigrid (fast to train) with a fixed
VAE + latent-IMPALA stack, varying only the two causal-representation losses in

\[
L_{\mathrm{CTRO}} = L_{\mathrm{PPO}} + \alpha\, L_{\mathrm{MICo}} + \beta\, L_{\mathrm{PL}}.
\]

### Design

A 2×2 over `{MICo off/on} × {PL off/on}`, 3 seeds (42/43/44), everything else held fixed:

| Cell | Exp name | \(\alpha\) (MICo) | \(\beta\) (PL) | Job |
|------|----------|-------------------|----------------|-----|
| BASELINE | `exp_baseline` | 0 | 0 | [`jobs/baseline_s.sh`](src/experiments/jobs/baseline_s.sh) |
| MICO_ONLY | `exp_mico_only` | 0.1 | 0 | [`jobs/mico_only_s.sh`](src/experiments/jobs/mico_only_s.sh) |
| PL_ONLY | `exp_pl_only` | 0 | 0.1 | [`jobs/pl_only_s.sh`](src/experiments/jobs/pl_only_s.sh) |
| FULL | `exp_full/alpha_{a}_beta_{b}` | swept | swept | [`jobs/full_*_s.sh`](src/experiments/jobs) |

Note BASELINE here is PPO on the **same VAE + policy-on-\(Z\) stack** (losses off), not the raw-obs
PPO used on the performance suites — so this isolates the losses, not the architecture.

### Result (3 seeds, from `plots/ctro/ablation_table.txt`)

| Cell | Return | \(\mu_{PL}\) (q05) | PR |
|------|--------|--------------------|-----|
| BASELINE | 0.064 ± 0.059 | 0.030 | 1.00 |
| MICO_ONLY | 0.006 ± 0.006 | 0.044 | 1.01 |
| PL_ONLY | 0.191 ± 0.191 | 0.227 | 1.00 |
| FULL (α=0.5, β=0.5) | 0.231 ± 0.231 | 0.290 | 1.05 |

### Reading (research narrative)

- **PL is the driver.** The two cells that turn PL on (`PL_ONLY`, `FULL`) are the only ones that lift
  both \(\mu_{PL}\) and return; `MICO_ONLY` moves neither (it is the negative control for "structure
  without value link"). This is the empirical form of the claim that constraining \(Z\) through value
  geometry — non-vanishing \(\mu\) — is what protects performance.
- **Rank collapse does not discriminate here.** PR ≈ 1 in every cell, so feature-rank collapse (the
  Moalla/Lyle story, documented elsewhere) does not separate good from bad policies in this ablation.
  Our contribution is the value-geometry axis, not rank.
- **Caveat.** Returns are noisy (SEM ≈ mean) and Panel B's \(\mu_{PL}\)-vs-return trend is weak because
  a few high-\(\mu_{PL}\), low-return outliers flatten it; Panel C (BASELINE vs PL_ONLY, dual-axis) is
  the cleanest single figure — \(\mu_{PL}\) and return co-move only when PL is on.

### Figures / reproduce

Panels live in `plots/ctro/`: `ablation_final_bars.png` (A), `mu_pl_vs_return.png` (B),
`dual_axis_baseline_vs_pl.png` (C).

```bash
python -m src.experiments.plot_ctro --results-root results --output-dir plots/ctro
```

---

## Performance suite: DMControl CTRO vs PPO (MLP, policy-on-Z)

State-based continuous control comparison on `dmcontrol_state` (cartpole-swingup, cheetah-run, hopper-hop, walker-walk). Config lives in [`src/experiments/config.py`](src/experiments/config.py) (`DMCONTROL_*`); stacks are built in [`src/experiments/performance_models.py`](src/experiments/performance_models.py).

### Motivation

CTRO is PPO plus causal-representation losses on a shared latent \(Z\):

\[
L_{\mathrm{CTRO}} = L_{\mathrm{PPO}} + \alpha\, L_{\mathrm{MICo}} + \beta\, L_{\mathrm{PL}}
\]

Minigrid ablations used a **VAE** encoder/decoder with policy-on-\(Z\). Porting that stack unchanged to DMControl **hurt** vs the PPO baseline: DMControl observations are already compact Markov physics states (obs dim ~5–24). A VAE bottleneck + recon/KL is an unnecessary generative prior there, and early training showed policy-ratio / representation collapse (especially cheetah).

What CTRO actually needs is not a VAE, but:

1. A map \(s \mapsto Z(s)\) shared by \(\pi(a|Z)\) and \(V(Z)\)
2. **MICo** (bisimulation geometry on \(Z\)) and **PL** (value-gradient / Bellman-gap coupling)

Reconstruction/KL are optional VAE regularizers, not part of the CTRO objective.

### Method: MLP CTRO (`exp_ctro_mlp`)

| Piece | Choice |
|-------|--------|
| Critic | `MLPEncoderCritic`: `obs → encoder MLP → Z → linear value_head → V` ([`mlp_encoder_critic.py`](src/architectures/critics/mlp_encoder_critic.py)) |
| Policy | `MLPPolicy` on **\(Z\)** (`policy_on_latent=True`) |
| \(Z\) dim | Last `encoder_hidden` size (`[256, 256]` → 256) |
| Policy torso | `[64, 64]`, `tanh` (same widths as PPO policy, input dim = latent) |
| VAE / recon | None (`vae_coef=0`, no decoder) |
| Losses | MICo (`α=0.01`) + PL (`β=0.1`); `entropy_coef=0` to match PPO |
| Stack type | `ctro_mlp` |
| Checkpoints | `results/dmcontrol_state/exp_ctro_mlp/seed_{N}/{task}/` |

Policy-on-\(Z\) is intentional: CTRO’s claim is that the policy acts in the causally structured representation, not only that the critic is regularized.

### Method: PPO baseline (`exp_baseline`)

| Piece | Choice |
|-------|--------|
| Critic | `FeedforwardCritic` on raw obs |
| Policy | `MLPPolicy` on raw obs (`policy_on_latent=False`) |
| Extra losses | None (`α=β=vae_coef=0`, `entropy_coef=0`) |
| Stack type | `ppo_mlp` |
| Checkpoints | `results/dmcontrol_state/exp_baseline/seed_{N}/{task}/` |

Comparison is **methodological** (shared \(Z\) + MICo/PL vs end-to-end MLP on state), not architecture-identical.

### Continuous PPO correctness (DMControl)

Matched to CleanRL continuous PPO on four points that were previously wrong under `num_envs=1`:

1. **Action / log-prob ranks** — `MLPPolicy.get_action` squeezes the batch dim for 1D obs so buffer stores `[T,A]` / `[T]` (not `[T,1,A]` / `[T,1,1]`).
2. **Terminated vs truncated** — dm_control time limits set `truncated=True` (`discount>0`); GAE bootstraps `V(s')` on truncate and only zeroes bootstrap on true `terminated`.
3. **Action bounds** — `DMControlWrapper.step` clips to `action_spec` (CleanRL `ClipAction`).
4. **State-independent `log_std`** — `nn.Parameter` vector, not a per-state Linear head.

### Normalization (DMControl / Procgen performance)

| Signal | Mode | Behavior |
|--------|------|----------|
| Observations | `obs_norm=running_mean_std` | Welford \((x-\mu)/\sqrt{\sigma^2+\epsilon}\), then clip ±`obs_norm_clip` (default 10) |
| Rewards | `reward_norm=return_var_scale` | CleanRL discounted-return std scale \(r/\sqrt{\mathrm{Var}(R)+\epsilon}\); **no** mean subtract, **no** reward clip |

This is the intended recipe for “Reward Normalization: False (use running variance scaling without clipping)”. Modes are asserted in [`PerformanceNormalizer`](src/utils/normalization.py); unknown modes raise.

### Optuna HP search (DMControl)

Truncated search (1M steps) over shared PPO knobs (+ \(\alpha,\beta\) for CTRO), with MedianPruner and return-collapse early abort. One study per `(agent, task)`.

**Hopper exception:** return-collapse is **off** (`DMCONTROL_COLLAPSE_FLOORS["hopper-hop"]=None`), search budget is **8M** steps, studies live under `hopper-hop_v2` (fresh DB), and cheetah/walker `best_trial.json` params are enqueued as warm starts.

```bash
# 8 array workers: exp_baseline|exp_ctro_mlp × 4 tasks (1M; hopper still uses task list but prefer hopper job)
N_TRIALS=20 sbatch src/experiments/jobs/optuna_dmcontrol_s.sh

# Hopper v2 only (8M, no collapse, transfer warm-start)
N_TRIALS=10 sbatch src/experiments/jobs/optuna_dmcontrol_hopper_s.sh

# Sensitivity + confirm command scripts
python -m src.experiments.analyze_optuna_sensitivity --optuna-root results/optuna --all

# Full 8M × 3-seed confirm (from generated confirm_commands.sh), or:
EXP_NAME=exp_optuna_confirm_ctro_t0 TASK=cartpole-swingup \
  EXTRA_ARGS_STR='--agent ctro --learning-rate 3e-4 --entropy-coef 0.01 --num-epochs 10 --policy-hidden 64,64 --alpha 0.01 --beta 0.1' \
  sbatch src/experiments/jobs/optuna_confirm_dmcontrol_s.sh
```

Search space / collapse floors: [`src/experiments/optuna_dmcontrol.py`](src/experiments/optuna_dmcontrol.py), [`DMCONTROL_COLLAPSE_FLOORS`](src/experiments/config.py). Failed/pruned runs write `run_status.json` and **do not** write `weights_final.pt`.

### What stayed on VAE

- **Minigrid** CTRO ablations: VAE + latent IMPALA (controlled loss ablations).
- **Procgen** CTRO: still CNN-VAE + latent policy (`PROCGEN_CTRO_ALGO_CONFIG` keeps `vae_coef=0.1`). Pixel domains still motivate a reconstructible latent; DMControl state does not.

### Launch

```bash
sbatch src/experiments/jobs/perf_train_dmcontrol_s.sh           # CTRO MLP → exp_ctro_mlp
sbatch src/experiments/jobs/perf_train_dmcontrol_baseline_s.sh # PPO → exp_baseline
sbatch src/experiments/jobs/perf_eval_dmcontrol_s.sh            # eval CTRO (default EXP_NAME=exp_ctro_mlp)
sbatch src/experiments/jobs/perf_eval_dmcontrol_baseline_s.sh
```

Legacy VAE DMControl runs (if any) used `exp_full` / `stack_type=ctro_mlp_vae`; eval still reloads that stack from checkpoint config for backward compatibility.

---

## Negative controls: is the *value link* what protects performance?

### Research narrative

The core hypothesis is a chain from representation geometry to policy performance:

\[
\text{value-geometry collapse } (\mu \to 0)
\;\Rightarrow\;
\|Z^*(s) - Z(s)\| \le \tfrac{1}{\mu}\|\nabla_Z V(Z(s))\| \text{ blows up}
\;\Rightarrow\;
\text{KL trust region no longer protects } Z
\;\Rightarrow\;
\text{return degrades.}
\]

Here \(\mu\) is the strong-convexity constant of \(V\) in \(Z\), estimated online by the PL ratio
\(\mu_{PL} = \|\nabla_Z V\|^2 / (2\cdot\text{Bellman gap})\) ([`pl_ratio.py`](src/metrics/pl_ratio.py)),
and feature rank / participation ratio (PR) ([`feature_rank.py`](src/metrics/feature_rank.py)) is the
capacity-collapse proxy from Moalla et al. / Lyle et al.

A separate paper already documents *feature-rank collapse vs return*, so that is not the contribution
here. Our claim is sharper: it is the **constraint of the causal representation through value geometry**
(the PL coupling / non-vanishing \(\mu\)) that keeps performance high — not merely having a
causal/structured latent. The Minigrid 2×2 ablation supports this: PL-on cells (`PL_ONLY`, `FULL`)
raise both \(\mu_{PL}\) and return, `MICO_ONLY` does not, and PR ≈ 1 across all cells (rank collapse
does not separate good from bad policies). See `plots/ctro/ablation_final_bars.png` (Panel A),
`plots/ctro/mu_pl_vs_return.png` (Panel B), `plots/ctro/dual_axis_baseline_vs_pl.png` (Panel C).

### Two negative controls (dmcontrol_state and procgen_easy)

To isolate the value link on the performance suites, we add two controls that both **lack** the
CTRO value coupling but differ in representational machinery:

| Exp name | Stack | Losses | What it isolates |
|----------|-------|--------|------------------|
| `exp_baseline` | DMControl `ppo_mlp` (obs→MLP→V) / Procgen `ppo_impala` | plain PPO, \(\alpha=\beta=0\), no shared \(Z\) | No causal machinery at all. Critic-torso features are treated as \(Z\); \(\mu_{PL}\)/PR are expected to drift/stay unhealthy. |
| `exp_latent_nolink` | Same as CTRO: DMControl `ctro_mlp` / Procgen `ctro_cnn_vae` (shared \(Z\), policy-on-\(Z\), Procgen keeps `vae_coef=0.1`) | CTRO agent with \(\alpha=0, \beta=0\) (no MICo/PL) | A causally-informed / reconstructible representation **without** the value link. Shows a good latent alone is insufficient. |
| `exp_ctro_mlp` (DMControl) / `exp_full` (Procgen) | CTRO stacks | \(\alpha,\beta > 0\) | Positive control: value link on. |

Narrative across the three: (1) plain PPO has no shared-\(Z\)/value-link machinery → \(\mu_{PL}\)/PR
unprotected; (2) matched encoder + policy-on-\(Z\) **still fails** without PL/MICo; (3) only full CTRO
keeps \(\mu_{PL}\)/PR healthy and return with them.

Note on interpretation: for plain PPO the metrics are computed on **critic-torso (penultimate)
features**, not a causal latent the policy uses. That is the point — PPO lacks the machinery, so the
geometry is not protected. With a linear value head, \(\nabla_Z V\) is near-constant across samples, so
PPO's \(\mu_{PL}\) largely tracks the Bellman gap; it remains a valid drift diagnostic but is weaker
than CTRO's richer \(V(Z)\).

### Instrumentation

- [`FeedforwardCritic`](src/architectures/critics/feedforward.py) and
  [`ImpalaValueCritic`](src/architectures/critics/impala_value_critic.py) now expose
  `encode()` (torso → \(Z\)) and a separate `value_head`, so the same
  [`CTROMetricEvaluator`](src/utils/ctro_metric_evaluator.py) computes \(\mu_{PL}\)/PR for PPO too.
- [`performance_runner.py`](src/experiments/performance_runner.py) attaches the metric evaluator for
  **every** stack (not just CTRO), so all `metrics.csv` gain `mu_pl_*` and `feature_rank_*` columns.
- Because the critic layer names changed (`network.*` → `encoder.*` + `value_head.*`), old
  `exp_baseline` checkpoints will not reload. Retrain `exp_baseline` on both suites to populate the
  representation-metric columns.

### Throughput: per-task array + seed pool

Training uses the serial rollout (`num_envs=1`, config default) but parallelizes across the two axes
that don't change learning dynamics:

- **Across tasks/games** — each script is a SLURM **array** (`--array=0-3` DMControl, `--array=0-7`
  Procgen), so every task gets its own small allocation that schedules independently across the
  cluster.
- **Across seeds (within a task)** — inside each array task the 3 seeds run as a concurrency pool
  ([`_parallel_seeds.sh`](src/experiments/jobs/_parallel_seeds.sh)), i.e. 3 single-env processes
  (~1 core each, `MAX_PARALLEL=3`) sharing one GPU. This is where the wall-clock speedup comes from.

Per array task: `--cpus-per-task=4`, `--gres=gpu:1`, `--mem=32gb` (DMControl) / `96gb` (Procgen).
Tune with `MAX_PARALLEL` / `THREADS_PER_PROC`; add `--gres=gpu:2` to spread the 3 seeds over 2 GPUs.

**Optional vectorization.** A vectorized rollout is also available for a single-process speedup:
set `num_envs` in [`config.py`](src/experiments/config.py) (or `--num-envs`). DMControl uses
[`SubprocVectorEnv`](src/environments/vec_env.py) (one MuJoCo worker per core); Procgen uses native
`ProcgenGym3Env(num=N)` via [`ProcgenVectorEnv`](src/environments/vec_env.py). Semantics are
preserved — per-env GAE ([`compute_gae_vec`](src/experiments/performance_runner.py), equivalent to
`PPO.compute_gae` per env), shared-variance reward normalization, terminal-obs bootstrapping — but it
**changes the rollout regime**, so if you enable it, use the **same** `num_envs` for BASELINE,
LATENT_NOLINK, and CTRO (bump `--cpus-per-task`/`ENVS_PER_PROC` accordingly).

```bash
# Each sbatch fans out into an array (one allocation per task/game); the 3 seeds
# pool inside each. Positive control: CTRO
sbatch src/experiments/jobs/perf_train_dmcontrol_s.sh   # exp_ctro_mlp
sbatch src/experiments/jobs/perf_train_procgen_s.sh     # exp_full

# Negative control 1: plain PPO (now logs mu_PL / PR) — retrain required
sbatch src/experiments/jobs/perf_train_dmcontrol_baseline_s.sh   # exp_baseline
sbatch src/experiments/jobs/perf_train_procgen_baseline_s.sh     # exp_baseline

# Negative control 2: causal latent, no value link (alpha=beta=0)
sbatch src/experiments/jobs/perf_train_dmcontrol_latent_nolink_s.sh  # exp_latent_nolink
sbatch src/experiments/jobs/perf_train_procgen_latent_nolink_s.sh    # exp_latent_nolink

# Run a single game instead of the whole array, e.g. index 0 (coinrun)
sbatch --array=0 src/experiments/jobs/perf_train_procgen_baseline_s.sh

# Eval (dedicated nolink scripts; agg covers all three methods)
sbatch src/experiments/jobs/perf_eval_dmcontrol_latent_nolink_s.sh
sbatch src/experiments/jobs/perf_eval_procgen_latent_nolink_s.sh
sbatch src/experiments/jobs/perf_eval_agg_s.sh  # exp_full + exp_ctro_mlp + exp_baseline + exp_latent_nolink
```

### Panels

Per-task Panel A/B/C (BASELINE vs LATENT_NOLINK vs CTRO) via
[`plot_performance_panels.py`](src/experiments/plot_performance_panels.py):

```bash
python -m src.experiments.plot_performance_panels --suite dmcontrol_state --task cheetah-run
python -m src.experiments.plot_performance_panels --suite procgen_easy --task coinrun
```

Outputs `plots/{suite}/{task}/{final_bars,mu_pl_vs_return,dual_axis}.png` and `table.txt`.
Panel B/C use `mean_episode_return` (performance train CSVs log per-distribution eval returns, not a
single `eval_return_mean`).

---

## Running Experiments

### Submit all experiments:
```bash
# Experiment 1: Vanilla baselines
sbatch slurm/exp1_vanilla_ppo_impala_mlp.sh
sbatch slurm/exp1_vanilla_trpo_impala_mlp.sh

# Experiment 2: Representational baselines
sbatch slurm/exp2_repr_ppo_impala_vae.sh
sbatch slurm/exp2_repr_trpo_impala_vae.sh

# Experiment 3: RSTR
sbatch slurm/exp3_rstr_impala_icnn.sh

# Experiment 4: RRSTR
sbatch slurm/exp4_rstr_impala_vae_strict.sh
```

### Monitor jobs:
```bash
squeue -u $USER
```

### Check logs:
```bash
tail -f *.out  # View output files
```

---

## Key Differences

| Experiment | Policy | Critic | Algorithm | Key Feature |
|------------|--------|--------|-----------|-------------|
| 1.1 | IMPALA | MLP | PPO | Baseline |
| 1.2 | IMPALA | MLP | TRPO | Baseline |
| 2.1 | IMPALA | VAE | PPO | Causal representation |
| 2.2 | IMPALA | VAE | TRPO | Causal representation |
| 3 | IMPALA | ICNN | TRPO | Convexity enforcement |
| 4 | IMPALA | VAE | PPO (strict) | Strict clipping |
| DMControl PPO (`exp_baseline`) | MLP on obs | Feedforward MLP | PPO | Neg control 1: plain PPO, metrics on critic-torso features |
| DMControl latent-no-link (`exp_latent_nolink`) | MLP on \(Z\) | MLP encoder + value head | PPO, \(\alpha=\beta=0\) | Neg control 2: shared causal \(Z\), no value link |
| DMControl CTRO (`exp_ctro_mlp`) | MLP on \(Z\) | MLP encoder + value head | PPO + MICo + PL | Positive control: shared causal \(Z\), no VAE |

---

## Expected Outputs

All experiments will produce:
- Logs in `./logs/{environment}/{policy_type}_{critic_type}/`
- Metrics CSV: `{experiment_name}_metrics.csv`
- Config backup: `{experiment_name}_config.json`
- Model weights: `weights_latest.pt` and `weights_final.pt`

---

## Notes

1. **RSTR Implementation**: The full representation-space trust region algorithm (with gradient/Hessian thresholding) is not yet implemented. Experiment 3 currently uses TRPO with ICNN critic, which enforces convexity. The RSTR algorithm can be added to `src/algorithms/representation_trpo.py` later.

2. **Strict Clipping**: Experiment 4 uses `clip_epsilon=0.1` (half of standard 0.2) for stricter policy updates.

3. **Hessian Metrics**: Experiment 3 enables Hessian spectrum computation to analyze convexity properties of the ICNN critic.

4. **All experiments use**:
   - Buffer size: 2048 steps per epoch
   - Total epochs: 1000
   - Metric evaluation: Every 10 epochs
   - Policy evaluation: Every 10 epochs
   - Checkpointing: Every 100 epochs

