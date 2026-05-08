# Causal RL Representation: Bounding Chain Experiments

This repository implements experiments for testing the bounding chain from policy KL to causal representation error via gradient and convexity measures in reinforcement learning.

## Documentation

- **[NEW_SPEC.md](NEW_SPEC.md)**: Experimental specification and requirements
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)**: Detailed implementation plan with phases, TODOs, and technical decisions

## Project Status

🚧 **In Planning Phase** - Implementation plan created, awaiting review and TODO completion.

## Quick Start

### Installation

1. Create a virtual environment (Python 3.10):
```bash
python3.10 -m venv rl-venv
source rl-venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install gym-minigrid:
```bash
pip install gym-minigrid
```

### Running Experiments

Train a baseline PPO agent on Minigrid:

```bash
python scripts/train_baseline.py --config configs/minigrid_config.json
```

Or use the main training script directly:

```bash
python -m src.main --config configs/minigrid_config.json
```

### Configuration

Experiments are configured via JSON files in `configs/`. See `configs/minigrid_config.json` for an example.

Key configuration sections:
- `experiment`: Experiment name, seed, device
- `environment`: Environment name and task
- `architecture`: Policy and critic network architectures
- `algorithm`: Training algorithm hyperparameters
- `training`: Training duration, checkpointing, evaluation
- `metrics`: Which metrics to collect
- `logging`: Logging directory and settings

### Experiment Storage

Experiments are stored in `{log_dir}/{environment}/{policy_type}_{critic_type}/`:
- `weights_latest.pt`: Latest checkpoint (overwritten)
- `weights_final.pt`: Final checkpoint
- `{experiment_name}_metrics.csv`: Training metrics
- `{experiment_name}_config.json`: Experiment configuration

## Project Structure

```
causal-rep/
├── src/                    # Main source code
│   ├── architectures/      # Critic and policy architectures
│   ├── environments/       # Environment wrappers
│   ├── algorithms/         # Training algorithms (TRPO, PPO, etc.)
│   ├── metrics/            # Metric collection implementations
│   └── utils/              # Utilities and helpers
├── configs/                # Configuration files (YAML)
├── scripts/                # Training and analysis scripts
├── tests/                  # Unit and integration tests
└── slurm_template.sh       # SLURM job template
```

## Requirements

**TODO**: See `requirements.txt` (to be created) and `IMPLEMENTATION_PLAN.md` Phase 0.2.

## Environments

- **Procgen**: Procedurally-generated tasks
- **MuJoCo**: Continuous control (Ant, Hopper, Walker2d, HalfCheetah)
- **Minigrid**: Gridworld navigation tasks
- **DMControl**: (Optional) Additional continuous control

## Architectures

### Critics
- ICNN (Input Convex Neural Network)
- Simple Feedforward NN
- VAE-based Critic

### Policies
- IMPALA
- Standard MLP Policy

## Metrics

All experiments collect:
- Sampled Hessian spectrum
- Fisher information index
- KL divergence
- Gradient magnitude
- Value gradient difference
- Causal prediction error
- Final policy regret
- Occupancy measure stability

## Contributing

This is a research codebase. Code must fail fast and loudly - always raise errors, never silently handle failures.

## References

See NEW_SPEC.md Section 6 for full references.

