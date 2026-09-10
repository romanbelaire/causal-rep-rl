# Moalla ALE protocol transcription

**Sources:** Moalla et al., *No Representation, No Trust* (NeurIPS 2024);
released configs in [CLAIRE-Labo/no-representation-no-trust](https://github.com/CLAIRE-Labo/no-representation-no-trust)
(`src/po_dynamics/configs/env/gym-atari.yaml`, `env_algo/gym-atari-ppo.yaml`, `algo/ppo.yaml`);
CleanRL Atari PPO defaults.

**Program overrides (locked):** seeds `{42,43,44}`; stress `num_epochs=16` (not 32);
Stage A total steps = Moalla released **100_000_000** (`collector.total_env_steps` in `gym-atari.yaml`).
Figures in the paper that extend to ~200M are longer diagnostic runs; our confirmatory
budget follows the released default config unless Stage A fails to reproduce collapse.

**Second-pass check:** verify this file against the paper PDF and the linked YAML files
before trusting Stage A as a literature reproduction.

---

## Environment

| Field | Value |
|---|---|
| IDs | `ALE/Phoenix-v5`, `ALE/NameThisGame-v5` |
| Sticky actions | `repeat_action_probability=0.25` |
| Frame skip | 3 in Moalla torchrl config; our gymnasium path uses `AtariPreprocessing(frame_skip=4)` (CleanRL/SB3 ALE-v5 convention) — **document any residual mismatch in run configs** |
| Obs | grayscale 84×84, frame stack 4 → shape `(4, 84, 84)` |
| Reward | sign transform |
| Truncation | 108_000 env steps / episode (30 min @ 60 fps); treat truncation as termination for GAE |
| Parallel envs | 8 |
| Eval envs / length | Moalla: 3 envs × 108_000; we use `eval_episodes` full episodes for return (same protocol for all methods) |

## PPO hyperparameters (Atari)

| Field | Value |
|---|---|
| Optimizer | Adam, lr `2.5e-4`, `max_grad_norm=0.5`, linear LR anneal (CleanRL) |
| Rollout | 8 envs × 128 steps = 1024 transitions; minibatch 256 → 4 minibatches/epoch |
| Standard epochs | **4** |
| Stress epochs | **16** (program lock) |
| Clip ε | 0.1 |
| Entropy coef | 0.01 |
| Value coef | 0.5 (L2; no value clipping in Moalla’s preferred setting) |
| γ | 0.99 |
| GAE λ | 0.95 |
| Advantage normalize | True |
| Total env steps | **100_000_000** |
| Actor / critic | **Separate** NatureCNN trunks (`share_features=False`) |

## Architecture (NatureCNN)

- Conv: 32@8×8/s4 → ReLU → 64@4×4/s2 → ReLU → 64@3×3/s1 → ReLU → Flatten
- Linear: 512 → ReLU  (**penultimate post-activation**)
- **Headline representation:** actor **pre-activation into the 512 Linear**
  (i.e. flattened conv features before the 512 affine), named `actor_preactivation`.
  Moalla PFO defaults to penultimate pre-activations; we lock this tensor name and never switch.
- Actor head: Linear(512, n_actions), orthogonal init std 0.01
- Critic head: Linear(512, 1), orthogonal init std 1.0
- Obs scaled by `/255.0` before the CNN

## PFO

\[
L_{\mathrm{PFO}}=\|\phi_{\mathrm{new}}(s)-\phi_{\mathrm{old}}(s)\|_2^2
\]

- \(\phi\) = `actor_preactivation`
- Old features from frozen pre-update actor copy; detached
- Coefficient **1.0** on ALE (Moalla: nearest power of 10 matching PPO clip loss scale)
- `feature_trust_use_preactivation=True`, penultimate only (`feature_trust_all_layers=False`)

## Primary functional metric

Freeze actor encoder (conv + flatten path). Fit fresh linear probes to **K fixed random
cumulant targets** with a fixed ridge/LS budget. Report held-out MSE, constant-predictor
MSE, NMSE = MSE/MSE_const, \(R^2=1-\mathrm{NMSE}\). Average over targets.
Assert encoder params receive no gradients during the fit.

## Geometric / capacity diagnostics (actor features)

On on-policy and fixed diagnostic observations: \(C_{0.01}\), dormant fraction,
pre/post-activation norms, stable rank, PCA/numerical rank, participation ratio,
mean pairwise cosine, duplicate-obs rate, policy entropy, approx KL, clip fraction.

## Version recording

Every run `config.json` must store: `ale_py` version, `gymnasium` version, env id,
wrapper stack name, protocol hash / path to this file, `freeze_id`.
