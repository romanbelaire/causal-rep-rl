# DMControl PPO baseline: diagnosis & fix notes (P0.3)

## Symptom

Reported end-to-end PPO (`exp_baseline`) held-out returns were far below known PPO
levels on state DMControl:

| Task              | Prior baseline | Target (CleanRL-class) |
|-------------------|----------------|------------------------|
| cartpole-swingup  | ~25            | > 800                  |
| walker-walk       | ~29            | > 900                  |
| cheetah-run       | ~4             | > 400                  |
| hopper-hop        | ~0             | unsolved OK            |

## Likely causes (addressed in config/stack)

1. **Serial rollout only (`num_envs=1`)** — insufficient environment throughput vs
   CleanRL's parallel vector env. Default is now **`num_envs=8`** (buffer still
   2048 = 8 × 256).
2. **Minibatch too large (`batch_size=256`)** — fewer gradient steps per epoch.
   PPO baseline now uses **`batch_size=64`** (CleanRL default). CTRO keeps 256 for
   MICo/PL batch statistics.
3. **Baseline critic/policy widths** — both use arch `baseline_hidden=[64,64]`
   explicitly (CleanRL continuous PPO). Prior code pulled widths only from the
   policy block (same value), but the path is now explicit and documented.
4. **Old results may predate continuous-PPO bugfixes** (rank of actions/log-probs,
   truncate vs terminate, action clip, state-independent log_std) already in the
   code; retrain after this patch before quoting tables.

## What is *not* the architectural twin of CTRO

`exp_latent_nolink` (α=β=0 on the encoder stack) remains the architecture-matched
control. Repaired PPO-on-obs is the **external** CleanRL-level sanity bar.

## Acceptance (retrain, then check)

```bash
python -m src.experiments.run_performance_train \
  --suite dmcontrol_state --task cartpole-swingup --seed 42 \
  --exp-name exp_baseline_v3 --agent ppo --results-root results
```

Targets: cartpole-swingup >800, walker-walk >900, cheetah-run >400 at full 8M.
Hopper-hop may remain low — report as unsolved, not a separator method.

## Logging residual shortfall

If gates still fail after retrain, document env wrapper reward scale, horizon, and
seed returns here before using baseline in paper tables.
