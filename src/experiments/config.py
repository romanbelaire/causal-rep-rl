"""Shared hyperparameters for CTRO Minigrid experiments."""

BASE_ALGO_CONFIG = {
    "learning_rate": 1e-4,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "vae_coef": 0.1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "max_grad_norm": 0.2,
    "batch_size": 256,
    "num_epochs": 3,
    "mu_0": 0.1,
    "beta_mico": 0.1,
    "pl_eps": 1e-4,
    "mico_huber_delta": 1.0,
    "mico_target_update_tau": 0.005,
}

BASE_ARCH_CONFIG = {
    "critic": {
        "type": "vae",
        "latent_dim": 8,
        "encoder_hidden": [64, 64],
        "decoder_hidden": [64, 64],
        "value_hidden": [64, 64],
        "activation": "gelu",
        "beta": 1.0,
    },
    "policy": {
        "type": "impala",
        "hidden_sizes": [64, 64],
        "activation": "gelu",
        "num_residual_blocks": 2,
    },
}

BASE_TRAINING_CONFIG = {
    "buffer_size": 4096,
    "total_epochs": 1500,
    "checkpoint_frequency": 250,
    "eval_frequency": 100,
    "eval_episodes": 100,
    "eval_deterministic": True,
    "log_interval_steps": 10000,
    "reward_dispersion_threshold": 0.01,
    "reward_dispersion_warn_steps": 50000,
}

DEFAULT_SEEDS = [42, 43, 44]

FULL_SWEEP_ALPHA = [0.01, 0.1, 0.5]
FULL_SWEEP_BETA = [0.01, 0.1, 0.5]

PROCGEN_ARCH_CONFIG = {
    "critic": {
        "type": "vae",
        "latent_dim": 256,
        "encoder_hidden": [512, 512],
        "decoder_hidden": [512, 512],
        "value_hidden": [256, 256],
        "activation": "gelu",
        "beta": 1.0,
    },
    "policy": {
        "type": "impala",
        "hidden_sizes": [256, 256],
        "activation": "gelu",
        "num_residual_blocks": 2,
    },
}

DMCONTROL_ARCH_CONFIG = {
    "critic": {
        "type": "vae",
        "latent_dim": 64,
        "encoder_hidden": [256, 256],
        "decoder_hidden": [256, 256],
        "value_hidden": [128, 128],
        "activation": "gelu",
        "beta": 1.0,
    },
    "policy": {
        "type": "impala",
        "hidden_sizes": [256, 256],
        "activation": "gelu",
        "num_residual_blocks": 2,
    },
}

PROCGEN_TRAINING_CONFIG = {
    **BASE_TRAINING_CONFIG,
    "total_epochs": 25_000_000 // BASE_TRAINING_CONFIG["buffer_size"],
}

DMCONTROL_TRAINING_CONFIG = {
    **BASE_TRAINING_CONFIG,
    "total_epochs": 3_000_000 // BASE_TRAINING_CONFIG["buffer_size"],
}

PERFORMANCE_SUITE_CONFIG = {
    "procgen_easy": {
        "arch": PROCGEN_ARCH_CONFIG,
        "training": PROCGEN_TRAINING_CONFIG,
        "results_prefix": "procgen_easy",
    },
    "dmcontrol_state": {
        "arch": DMCONTROL_ARCH_CONFIG,
        "training": DMCONTROL_TRAINING_CONFIG,
        "results_prefix": "dmcontrol_state",
    },
}
