# Implementation Plan: Bounding Chain for Causal RL Representation

## Overview

This document provides a detailed implementation plan for the bounding chain experiments specified in `NEW_SPEC.md`. The plan is organized into phases with clear deliverables, dependencies, and technical requirements.

---

## Phase 0: Project Setup & Infrastructure

### 0.1 Project Structure
```
causal-rep/
├── src/
│   ├── __init__.py
│   ├── architectures/
│   │   ├── __init__.py
│   │   ├── critics/
│   │   │   ├── __init__.py
│   │   │   ├── icnn.py          # Input Convex Neural Network
│   │   │   ├── feedforward.py   # Standard MLP critic
│   │   │   └── vae_critic.py    # VAE-based critic
│   │   └── policies/
│   │       ├── __init__.py
│   │       ├── impala.py         # IMPALA policy
│   │       └── mlp_policy.py     # Standard MLP policy
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── procgen_wrapper.py
│   │   ├── mujoco_wrapper.py
│   │   ├── minigrid_wrapper.py
│   │   └── dmcontrol_wrapper.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── trpo.py               # TRPO baseline
│   │   ├── ppo.py                # PPO baseline
│   │   └── representation_trpo.py  # Representation-space trust region
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── hessian.py            # Hessian spectrum computation
│   │   ├── fisher.py             # Fisher information
│   │   ├── kl_divergence.py
│   │   ├── gradients.py          # Gradient magnitude & differences
│   │   ├── causal_error.py       # Causal prediction error
│   │   ├── regret.py             # Policy regret
│   │   └── occupancy.py          # Occupancy measure stability
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management
│   │   ├── logging.py            # Metric logging
│   │   └── visualization.py      # Plotting utilities
│   └── main.py                   # Main training loop
├── configs/
│   ├── base_config.yaml
│   ├── procgen_config.yaml
│   ├── mujoco_config.yaml
│   └── minigrid_config.yaml
├── scripts/
│   ├── train_baseline.py
│   ├── train_representation.py
│   ├── extract_metrics.py
│   └── visualize_results.py
├── tests/
│   ├── test_architectures.py
│   ├── test_metrics.py
│   └── test_environments.py
├── requirements.txt
├── setup.py
├── README.md
├── NEW_SPEC.md
├── IMPLEMENTATION_PLAN.md
└── slurm_template.sh
```

NOTE: use JSON for config format.

### 0.2 Dependencies
**TODO**: Specify exact versions for:
- PyTorch == as compatible with below
- Gym / Gymnasium == most recent
- MuJoCo (version) == most recent
- Procgen == low prio
- Minigrid == most recent
- NumPy, SciPy == as compatible with pytorch
- Matplotlib, Seaborn (visualization) == as needed
- ICNN implementation library (if external)

### 0.3 Development Environment
- Python version: **TODO** (3.10)
- CUDA version: **TODO** (from slurm_template.sh: CUDA 12.6.0)
- SLURM cluster configuration: Already configured in `slurm_template.sh`

---

## Phase 1: Core Architecture Implementations

### 1.1 Critic Architectures

#### 1.1.1 ICNN (Input Convex Neural Network)
**Status**: TODO
**Requirements**:
- Enforce local strong convexity for value function
- Configurable layer count, activation functions
- Explicit convexity constraints (e.g., positive weights in certain layers)
- **Dependencies**: ICNN library in @./convex-init/
- **Key parameters to document**: 
  - Layer architecture
  - Convexity enforcement method: see @convexity.md
  - Activation functions (must preserve convexity)

#### 1.1.2 Simple Feedforward NN
**Status**: TODO
**Requirements**:
- Standard multilayer perceptron
- Test for empirical convexity near optima
- Configurable depth, width, activation
- **Key parameters to document**:
  - Hidden layer sizes
  - Activation function (ReLU, Tanh, etc.)
  - Initialization scheme

#### 1.1.3 VAE-based Critic
**Status**: TODO
**Requirements**:
- Variational autoencoder for encoding causal features
- Value function over latent space
- **Key parameters to document**:
  - Encoder/decoder architecture
  - Latent dimension size
  - VAE loss weighting (reconstruction vs KL)
  - Value head architecture

### 1.2 Policy Architectures

#### 1.2.1 IMPALA
**Status**: TODO
**Requirements**:
- Scalable RL policy for multi-environment tests
- **Dependencies**: IMPALA implementation (custom or library)
- **Key parameters to document**:
  - Network architecture
  - Distribution type (categorical, Gaussian, etc.)

#### 1.2.2 Standard MLP Policy
**Status**: TODO
**Requirements**:
- Baseline MLP policy
- Support for discrete and continuous actions
- **Key parameters to document**:
  - Hidden layer sizes
  - Output distribution parameters

---

## Phase 2: Environment Integrations

### 2.1 Environment Wrappers
Each environment needs a standardized wrapper that:
- Provides consistent interface for state/action spaces
- Handles ground-truth causal representation extraction (where available)
- Supports metric collection hooks

#### 2.1.1 Procgen
**Status**: TODO, low prio
**Requirements**:
- **TODO**: Specify which Procgen games to use
- Ground-truth representation extraction method
- **Dependencies**: `procgen` package

#### 2.1.2 MuJoCo
**Status**: TODO, low prio
**Requirements**:
- Environments: Ant, Hopper, Walker2d, HalfCheetah (specified)
- **TODO**: Determine ground-truth representation (if any)
- **Dependencies**: `mujoco`, `gym`/`gymnasium`

#### 2.1.3 Minigrid
**Status**: TODO, high prio
**Requirements**:
- **TODO**: Specify which Minigrid tasks:  (MiniGrid-Unlock-v0, MiniGrid-DoorKey-5x5-v0)
- Ground-truth representation (grid positions, object states)
- **Dependencies**: `gym-minigrid`

---

## Phase 3: Training Algorithms

### 3.1 Baseline Algorithms

#### 3.1.1 TRPO (Trust Region Policy Optimization)
**Status**: TODO
**Requirements**:
- Standard TRPO with policy KL trust region
- **Key parameters**:
  - Trust region size (delta)
  - Conjugate gradient iterations
  - Line search parameters

#### 3.1.2 PPO (Proximal Policy Optimization)
**Status**: TODO
**Requirements**:
- Standard PPO implementation
- **Key parameters**:
  - Clipping epsilon
  - Value loss coefficient
  - Entropy coefficient

### 3.2 Representation-Space Trust Region
**Status**: TODO
**Requirements**:
- Modify trust region to use representation-space bounds
- Gradient/Hessian thresholding for critics
- **Key parameters**:
  - Representation trust region size
  - Hessian eigenvalue threshold
  - Gradient magnitude threshold
- **Dependencies**: Metrics from Phase 4

---

## Phase 4: Metrics Collection

### 4.1 Core Metrics

#### 4.1.1 Sampled Hessian Spectrum
**Status**: TODO
**Requirements**:
- Compute eigenvalues of $\nabla^2 V(z)$ on critic inputs
- Sample states near optimal states
- **Implementation notes**:
  - Use automatic differentiation for Hessian
  - Handle large Hessian matrices efficiently
  - **TODO**: Specify sampling strategy (how many states, which states)

#### 4.1.2 Fisher Information Index
**Status**: TODO
**Requirements**:
- Estimate policy Fisher information at sampled states
- **Implementation notes**:
  - Compute $F(\theta) = \mathbb{E}[\nabla \log \pi(a|s) \nabla \log \pi(a|s)^T]$
  - **TODO**: Specify sampling strategy

#### 4.1.3 KL Divergence
**Status**: TODO
**Requirements**:
- Aggregate policy KL between current and previous iterations
- **Implementation notes**:
  - Compute $D_{KL}(\pi_{\text{old}} || \pi_{\text{new}})$
  - Average over state distribution

#### 4.1.4 Gradient Magnitude
**Status**: TODO
**Requirements**:
- $||\nabla_\theta V^\pi(s)||$ per checkpoint
- **Implementation notes**:
  - Compute for all policy parameters
  - **TODO**: Specify checkpoint frequency

#### 4.1.5 Value Gradient Difference
**Status**: TODO
**Requirements**:
- $||\nabla V^{\pi'}(s) - \nabla V^{\pi}(s)||$ across policy changes
- **Implementation notes**:
  - Store previous policy's value gradients
  - Compare at same states

#### 4.1.6 Causal Prediction Error
**Status**: TODO
**Requirements**:
- $||Z^*(s) - Z(s)||$ with optimal representation $Z^*$
- **Implementation notes**:
  - **TODO**: Define how to obtain $Z^*$ for each environment
  - **TODO**: Define representation extraction method $Z(s)$ from learned models

#### 4.1.7 Final Policy Regret
**Status**: TODO
**Requirements**:
- Empirical difference to optimal policy return
- **Implementation notes**:
  - **TODO**: Define how to obtain optimal policy return (oracle, best baseline, etc.)

#### 4.1.8 Occupancy Measure Stability
**Status**: TODO
**Requirements**:
- Total variation and/or KL on state distributions
- **Implementation notes**:
  - Estimate state visitation distributions
  - Compare across iterations
  - **TODO**: Specify discretization method for continuous states

### 4.2.1 Logging Infrastructure
**Status**: TODO
**Requirements**:
- Per-iteration logging
- Final summary statistics
- **TODO**: Choose logging backend (custom CSV)
- Storage format: **TODO** (CSV)

### 4.2.2 Experiment Storage
- need to store experiment data: metrics, config, logs, model weights
- weights should be stored as: _latest (gets overwritten) and _final
- define directory structure by environmnet, then by network type. only need one instance per combination of networks (i.e., IMPALA PPO + ICNN is one combo). Existing combinations may be overwritten with new experiments for now.

---

## Phase 5: Experimental Protocols

### 5.1 Training Sequence
For each (environment, critic_arch, policy_arch) combination:

1. **Baseline Training**
   - Train with TRPO/PPO using standard policy KL trust region
   - **TODO**: Specify number of training steps/iterations
   - **TODO**: Specify convergence criteria

2. **Representation-Space Training**
   - Train with representation-space trust region
   - Use same hyperparameters where possible
   - **TODO**: Specify trust region parameters

3. **Metric Collection**
   - Collect all metrics at checkpoints during training
   - Collect final metrics after convergence
   - **TODO**: Specify checkpoint frequency

4. **Ablation Studies**
   - Switch critic architectures while holding policy fixed
   - Switch policy architectures while holding critic fixed
   - **TODO**: Specify which combinations to test

### 5.2 Hyperparameter Configuration
**Status**: TODO
**Requirements**:
- JSON-based configuration files
- Separate configs for each environment
- **TODO**: Define all hyperparameters:
  - Learning rates (policy, critic)
  - Batch sizes
  - Network architectures
  - Trust region sizes
  - Training duration
  - Evaluation frequency

---

## Phase 6: Visualization & Reporting

### 6.1 Visualization Scripts
**Status**: TODO
**Requirements**:
- Plot all metrics over training
- Compare baseline vs representation-space methods
- Compare across architectures
- **TODO**: Specify plot types:
  - Learning curves
  - Hessian eigenvalue distributions
  - Gradient magnitude distributions
  - Causal error vs training iteration

### 6.2 Summary Tables
**Status**: TODO
**Requirements**:
- Final metric values for all experiments
- Comparison tables (baseline vs representation-space)
- **TODO**: Define table format (Markdown)

### 6.3 Reproducibility
**Status**: TODO
**Requirements**:
- Version control all code, configs, logs
- README with:
  - Environment setup instructions
  - Installation steps
  - How to run experiments
  - How to reproduce results
- **TODO**: Specify version control strategy (Git)

---

## Phase 7: Testing & Validation
Note: tests should be as non-invasive as possible. This codebase is a research codebase, so only I will be accessing it.

### 7.1 Unit Tests
**Status**: TODO
**Requirements**:
- Test each architecture component
- Test metric computations on known inputs
- Test environment wrappers

### 7.2 Integration Tests
**Status**: TODO
**Requirements**:
- Test full training loop on small-scale problems
- Verify metric collection works end-to-end

### 7.3 Validation
**Status**: TODO
**Requirements**:
- **TODO**: Define validation criteria:
  - Do metrics match expected theoretical properties?
  - Do implementations match reference implementations (if any)?

---

## Implementation Timeline & Priorities

### Priority 1 (Core Functionality)
1. Basic project structure and dependencies
2. Simple Feedforward critic and MLP policy
3. One environment (start with Minigrid - simplest)
4. PPO baseline implementation
5. Basic metrics (KL, gradient magnitude)

### Priority 2 (Full Baseline)
1. TRPO implementation
2. All three environments (Procgen, MuJoCo, Minigrid)
3. All critic architectures (ICNN, Feedforward, VAE)
4. IMPALA policy
5. All metrics collection

### Priority 3 (Advanced Features)
1. Representation-space trust region algorithm
2. DMControl integration (optional)
3. Advanced visualization
4. Comprehensive ablation studies

---

## Open Questions & TODOs

### Technical Decisions Needed
1. **Deep Learning Framework**: PyTorch or TensorFlow? (Recommend PyTorch for research)
2. **ICNN Implementation**: Use existing library or implement from scratch?
3. **Ground-truth Representations**: How to extract $Z^*$ for each environment?
4. **VAE Architecture**: Specific VAE variant? (Standard VAE, $\beta$-VAE, etc.)
5. **Logging Backend**: WandB, TensorBoard, or custom?
6. **Hessian Computation**: Full Hessian or approximate (e.g., diagonal, top-k eigenvalues)?

### Hyperparameters to Define
1. Training duration (steps/iterations) per experiment
2. Checkpoint frequency
3. Evaluation frequency
4. Network architectures (layer sizes, activations)
5. Learning rates
6. Batch sizes
7. Trust region sizes
8. Convergence criteria

### Experimental Design
1. Which Procgen games to use?
2. Which Minigrid tasks to use?
3. Which DMControl tasks (if using)?
4. How many random seeds per experiment?
5. Which architecture combinations to test?
6. Sampling strategies for Hessian/Fisher computation


---

## References

As specified in NEW_SPEC.md:
1. Richens & Everitt. "Causal World Models". ICLR 2024.
2. Schulman et al. "Trust Region Policy Optimization". ICML 2015.
3. Nabati et al. "Representation-Driven RL". ICLR 2023.
4. Attached manuscripts: "Bounding Gradient Differences...", "Causal Bounds Formalization", "Local Convexity Theorem", "Summary Theorem & Literature Review".

