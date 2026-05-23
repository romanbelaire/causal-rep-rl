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

