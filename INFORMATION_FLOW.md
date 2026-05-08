# Information Flow Documentation

This document describes the information flow through the training pipeline, from main entry point to episode processing and metric evaluation.

## High-Level Flow

```
Main Entry Point
    ↓
train() - Main training function
    ↓
Setup Phase (one-time)
    ↓
Epoch Loop (repeated)
    ├─→ collect_rollout_buffer() - Collect episodes
    ├─→ algorithm.update() - Train networks
    ├─→ metric_evaluator.evaluate_all() - Compute metrics (periodic)
    ├─→ Policy evaluation (periodic)
    └─→ logger.log_metrics() - Log results
```

## Detailed Flow

### 1. Main Entry Point (`src/main.py`)

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)  # → train()
```

**Input**: Config file path  
**Output**: None (side effects: training, logging, checkpointing)

---

### 2. Setup Phase (`train()` function)

**Steps**:
1. **Load Configuration** (`Config(config_path)`)
   - Loads JSON config file
   - Validates required keys
   - Provides dot-notation access

2. **Set Random Seeds** (`set_seed(seed)`)
   - Python random, NumPy, PyTorch (CPU/GPU)
   - Ensures reproducibility

3. **Create Environment** (`MinigridWrapper`)
   - Initialize gym environment
   - Extract observation/action dimensions
   - Set up ground-truth representation function

4. **Create Networks** (`create_policy()`, `create_critic()`)
   - Policy: MLP or IMPALA
   - Critic: Feedforward, ICNN, or VAE
   - Move to device (CPU/GPU)

5. **Create Algorithm** (`PPO` or `TRPO`)
   - Initialize optimizers
   - Set hyperparameters from config

6. **Create Logger** (`CSVLogger`)
   - Set up CSV logging
   - Save experiment config

7. **Create Metric Evaluator** (`MetricEvaluator`)
   - Configure which metrics to compute
   - Set up ground-truth representation function

---

### 3. Epoch Loop (`train()` function)

Each epoch consists of:

#### 3.1. Rollout Collection (`collect_rollout_buffer()`)

**Purpose**: Collect experience until buffer reaches target size (e.g., 2048 steps)

**Flow**:
```
collect_rollout_buffer()
    ↓
While buffer not full:
    ├─→ env.reset() - Reset environment
    ├─→ While episode not done:
    │   ├─→ policy.get_action(obs) - Sample action
    │   ├─→ critic(obs) - Estimate value
    │   ├─→ env.step(action) - Execute action
    │   └─→ Store (obs, action, reward, done, log_prob, value)
    └─→ Track episode returns/lengths
    ↓
Return buffer dict with tensors
```

**Input**:
- `env`: Environment wrapper
- `policy`: Policy network
- `critic`: Critic network
- `buffer_size`: Target buffer size (steps)

**Output**:
- `buffer`: Dictionary with:
  - `obs`: [buffer_size, obs_dim]
  - `actions`: [buffer_size]
  - `rewards`: [buffer_size]
  - `dones`: [buffer_size]
  - `log_probs`: [buffer_size]
  - `values`: [buffer_size]
  - `episode_returns`: List of episode returns
  - `episode_lengths`: List of episode lengths

**Key Points**:
- Episodes may vary in length
- Buffer fills until target size reached
- Episodes can span buffer boundaries

---

#### 3.2. Advantage Computation (`algorithm.compute_gae()`)

**Purpose**: Compute advantages and returns using Generalized Advantage Estimation

**Flow**:
```
compute_gae(rewards, values, dones, next_value)
    ↓
For each timestep (backward):
    ├─→ Compute TD error: δ = r + γV(s') - V(s)
    ├─→ Compute GAE: A = δ + (γλ)A_next
    └─→ Compute return: G = A + V(s)
    ↓
Return (advantages, returns)
```

**Input**:
- `rewards`: [buffer_size]
- `values`: [buffer_size]
- `dones`: [buffer_size]
- `next_value`: Value after buffer (usually 0)

**Output**:
- `advantages`: [buffer_size] - Advantage estimates
- `returns`: [buffer_size] - Return estimates

---

#### 3.3. Network Update (`algorithm.update()`)

**Purpose**: Update policy and critic networks on collected buffer

**Flow (PPO)**:
```
algorithm.update(obs, actions, old_log_probs, advantages, returns)
    ↓
Normalize advantages
    ↓
Create DataLoader from buffer
    ↓
For num_epochs:
    For each batch:
        ├─→ policy.evaluate_actions(obs, actions) - Get current log_probs
        ├─→ Compute policy loss (clipped surrogate)
        ├─→ critic(obs) - Get values
        ├─→ Compute value loss (MSE)
        ├─→ Compute entropy bonus
        ├─→ Backpropagate and update policy
        └─→ Backpropagate and update critic
    ↓
Return statistics (policy_loss, value_loss, entropy, kl)
```

**Flow (TRPO)**:
```
algorithm.update(obs, actions, old_log_probs, advantages, returns)
    ↓
Update critic (standard gradient descent)
    ↓
Compute policy gradient
    ↓
Compute Fisher-vector products (conjugate gradient)
    ↓
Natural gradient step with line search
    ↓
Return statistics
```

**Input**:
- `obs`: [buffer_size, obs_dim]
- `actions`: [buffer_size]
- `old_log_probs`: [buffer_size] - Log probs from rollout
- `advantages`: [buffer_size]
- `returns`: [buffer_size]

**Output**:
- Dictionary with:
  - `policy_loss`: Scalar
  - `value_loss`: Scalar
  - `entropy`: Scalar
  - `kl`: Scalar (KL divergence)

**Key Points**:
- One epoch = one pass through buffer
- Multiple epochs per buffer (configurable)
- Policy and critic updated separately

---

#### 3.4. Metric Evaluation (`metric_evaluator.evaluate_all()`)

**Purpose**: Compute expensive metrics periodically (e.g., every 10 epochs)

**Flow**:
```
metric_evaluator.evaluate_all(policy, critic, obs_buffer, ...)
    ↓
Sample subset of observations (e.g., 128)
    ↓
For each enabled metric:
    ├─→ Hessian spectrum (if enabled)
    │   └─→ compute_hessian_spectrum(critic, obs)
    ├─→ Fisher information (if enabled)
    │   └─→ compute_fisher_information_index(policy, obs)
    ├─→ Causal error (if enabled)
    │   └─→ compute_causal_prediction_error(critic, ground_truth_fn, obs)
    ├─→ Policy regret (if enabled)
    │   └─→ compute_policy_regret(episode_returns)
    ├─→ Occupancy stability (if enabled)
    │   └─→ compute_occupancy_measure(obs) + KL comparison
    ├─→ KL divergence (if old_policy available)
    └─→ Gradient metrics (if enabled)
    ↓
Return dictionary of metric values
```

**Input**:
- `policy`: Current policy
- `critic`: Current critic
- `obs_buffer`: Buffer of observations
- `old_policy`: Previous policy (for comparison)
- `old_critic`: Previous critic (for comparison)
- `episode_returns`: List of episode returns
- `old_occupancy`: Previous occupancy measure

**Output**:
- Dictionary of metric values (e.g., `{"hessian_min_eigenvalue": 0.5, ...}`)

**Key Points**:
- Expensive metrics computed periodically (not every epoch)
- Uses subset of observations for efficiency
- Compares current vs. previous models when available

---

#### 3.5. Policy Evaluation (`train()` function)

**Purpose**: Evaluate policy performance by running deterministic rollouts

**Flow**:
```
For eval_episodes:
    ├─→ env.reset()
    ├─→ While episode not done:
    │   ├─→ policy.get_action(obs, deterministic=True)
    │   ├─→ env.step(action)
    │   └─→ Accumulate reward
    └─→ Record episode return
    ↓
Compute mean/std of returns
    ↓
Log eval metrics
```

**Input**: None (uses current policy and environment)  
**Output**: Mean and std of evaluation returns

**Key Points**:
- Uses deterministic policy (no exploration)
- Separate from training rollouts
- Runs periodically (e.g., every 10 epochs)

---

#### 3.6. Logging (`logger.log_metrics()`)

**Purpose**: Write metrics to CSV file

**Flow**:
```
logger.log_metrics(step, metrics_dict)
    ↓
If first call:
    ├─→ Create CSV file
    └─→ Write header
    ↓
Write row: step + metrics_dict
    ↓
Flush to disk
```

**Input**:
- `step`: Current step/epoch number
- `metrics_dict`: Dictionary of metric values

**Output**: None (side effect: CSV file updated)

---

## Epoch Definition

**One Training Epoch**:
1. Collect rollout buffer (until `buffer_size` steps reached)
2. Compute advantages/returns
3. Update policy and critic networks (multiple passes through buffer)
4. Log basic metrics

**One Metric Evaluation Cycle** (every N epochs):
1. Evaluate all enabled metrics
2. Compare with previous models/measures
3. Log metric results

**One Policy Evaluation Cycle** (every M epochs):
1. Run deterministic rollouts
2. Compute mean/std returns
3. Log evaluation metrics

---

## Data Structures

### Buffer Structure
```python
buffer = {
    "obs": torch.Tensor,          # [buffer_size, obs_dim]
    "actions": torch.Tensor,       # [buffer_size]
    "rewards": torch.Tensor,       # [buffer_size]
    "dones": torch.Tensor,         # [buffer_size]
    "log_probs": torch.Tensor,    # [buffer_size]
    "values": torch.Tensor,        # [buffer_size]
    "episode_returns": List[float], # Variable length
    "episode_lengths": List[int],   # Variable length
}
```

### Metrics Dictionary
```python
metrics = {
    "epoch": int,
    "total_steps": int,
    "episodes": int,
    "mean_episode_return": float,
    "mean_episode_length": float,
    "policy_loss": float,
    "value_loss": float,
    "entropy": float,
    "kl": float,
    # ... additional metrics if evaluated
}
```

---

## Key Design Decisions

1. **Buffer-based epochs**: Episodes collected until buffer full, then training
2. **Periodic metric evaluation**: Expensive metrics computed every N epochs
3. **Separate evaluation**: Policy evaluation uses deterministic rollouts
4. **Model snapshots**: Old models saved for comparison metrics
5. **Efficient sampling**: Metrics use subset of observations

---

## Example Timeline

```
Epoch 1:  Collect buffer (2048 steps) → Update networks → Log basic metrics
Epoch 2:  Collect buffer → Update → Log basic metrics
...
Epoch 10: Collect buffer → Update → Evaluate metrics → Evaluate policy → Log all metrics
Epoch 11: Collect buffer → Update → Log basic metrics
...
Epoch 20: Collect buffer → Update → Evaluate metrics → Evaluate policy → Log all metrics
```

This design balances:
- **Training efficiency**: Frequent network updates
- **Metric accuracy**: Periodic comprehensive evaluation
- **Computational cost**: Expensive metrics computed sparingly

