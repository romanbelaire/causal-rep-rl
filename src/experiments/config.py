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
    "mode": "cnn",
    "cnn": {"depths": [16, 32, 32], "emb_size": 256},
    "critic": {
        "type": "encoder",
        "latent_dim": 128,
        "encoder_hidden": [256, 256],
        "decoder_hidden": [256, 256],
        "value_hidden": [128, 128],
        "activation": "gelu",
        "beta": 1.0,
    },
    "policy": {
        "type": "impala",
        "hidden_sizes": [128, 128],
        "activation": "gelu",
        "num_residual_blocks": 2,
    },
}

DMCONTROL_ARCH_CONFIG = {
    "mode": "mlp",
    "critic": {
        "type": "mlp_encoder",
        "encoder_hidden": [256, 256],
        "activation": "tanh",
    },
    "policy": {
        "type": "mlp",
        "hidden_sizes": [64, 64],
        "activation": "tanh",
        "num_residual_blocks": 0,
    },
}

PROCGEN_PPO_ALGO_CONFIG = {
    "learning_rate": 5e-4,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "vae_coef": 0.0,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "max_grad_norm": 0.5,
    "batch_size": 256,
    "num_epochs": 3,
    "alpha": 0.0,
    "beta": 0.0,
}

PROCGEN_CTRO_ALGO_CONFIG = {
    **PROCGEN_PPO_ALGO_CONFIG,
    "vae_coef": 0.0,
    "alpha": 0.1,
    "beta": 0.5,
}

DMCONTROL_PPO_ALGO_CONFIG = {
    "learning_rate": 3e-4,
    "value_coef": 0.5,
    "entropy_coef": 0.0,
    "vae_coef": 0.0,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "max_grad_norm": 0.5,
    "batch_size": 256,
    "num_epochs": 10,
    "alpha": 0.0,
    "beta": 0.0,
}

DMCONTROL_CTRO_ALGO_CONFIG = {
    **DMCONTROL_PPO_ALGO_CONFIG,
    "vae_coef": 0.0,
    "entropy_coef": 0.0,
    "alpha": 0.01,
    "beta": 0.1,
}

PROCGEN_TRAINING_CONFIG = {
    **BASE_TRAINING_CONFIG,
    "n_envs": 64,
    "n_steps": 256,
    "buffer_size": 64 * 256,
    "total_epochs": 25_000_000 // (64 * 256),
    "checkpoint_frequency": 50,
    "eval_frequency": 500,
    "eval_episodes": 10,
    "metric_pl_max_samples": 64,
    "print_every_epochs": 10,
    # Serial rollout (num_envs=1). Set to 64 to vectorize: Procgen batches natively
    # in C, giving the canonical 64 envs x 256 steps = 16384-transition rollout.
    "num_envs": 1,
    "obs_norm": "running_mean_std",
    "obs_norm_clip": 10.0,
    "reward_norm": "return_var_scale",
}

DMCONTROL_TRAINING_CONFIG = {
    **BASE_TRAINING_CONFIG,
    "buffer_size": 2048,
    "total_epochs": 8_000_000 // 2048,
    "eval_frequency": 50,
    "eval_episodes": 10,
    "metric_pl_max_samples": 64,
    "print_every_epochs": 20,
    # Serial rollout (num_envs=1). Set to 8 to vectorize via subprocess MuJoCo
    # workers (one env per core): 8 envs x 256 steps = 2048-transition rollout.
    "num_envs": 1,
    # Obs: Welford running mean/std z-score with ±clip (not reward clipping).
    "obs_norm": "running_mean_std",
    "obs_norm_clip": 10.0,
    # Reward: CleanRL discounted-return variance scaling, no mean subtract, no clip.
    # ("Reward Normalization: False" in the SB3/recipe sense.)
    "reward_norm": "return_var_scale",
    # Early-stop / collapse prune (Optuna search + full runs).
    "collapse_min_steps": 200_000,
    "collapse_streak": 3,
}

# Per-task floor on rolling mean episode return for collapse pruning.
# None disables return-collapse (MedianPruner / NaN fail still apply in Optuna).
DMCONTROL_COLLAPSE_FLOORS = {
    "cartpole-swingup": 5.0,
    "cheetah-run": 1.0,
    "walker-walk": 1.0,
    # hopper-hop is sparse/unstable: train mean often returns to ~0 between rare hops.
    "hopper-hop": None,
}

# Truncated Optuna search budget (confirm winners at full 8M when search used 1M).
DMCONTROL_OPTUNA_SEARCH_STEPS = 1_000_000
# hopper needs a longer search horizon; confirm is the same 8M budget.
DMCONTROL_OPTUNA_SEARCH_STEPS_BY_TASK = {
    "hopper-hop": 8_000_000,
}
# Donor tasks whose best_trial.json is enqueued to warm-start hopper studies.
DMCONTROL_HOPPER_TRANSFER_TASKS = ("cheetah-run", "walker-walk")
# Fresh Optuna study folder for hopper (avoids TPE pollution from the all-pruned v1 study).
DMCONTROL_HOPPER_STUDY_KEY = "hopper-hop_v2"
PERFORMANCE_SUITE_CONFIG = {
    "procgen_easy": {
        "arch": PROCGEN_ARCH_CONFIG,
        "training": PROCGEN_TRAINING_CONFIG,
        "ppo_algo": PROCGEN_PPO_ALGO_CONFIG,
        "ctro_algo": PROCGEN_CTRO_ALGO_CONFIG,
        "results_prefix": "procgen_easy",
    },
    "dmcontrol_state": {
        "arch": DMCONTROL_ARCH_CONFIG,
        "training": DMCONTROL_TRAINING_CONFIG,
        "ppo_algo": DMCONTROL_PPO_ALGO_CONFIG,
        "ctro_algo": DMCONTROL_CTRO_ALGO_CONFIG,
        "results_prefix": "dmcontrol_state",
    },
    "dmcontrol_pixels": {
        "arch": PROCGEN_ARCH_CONFIG,
        "training": DMCONTROL_TRAINING_CONFIG,
        "ppo_algo": {
            **DMCONTROL_PPO_ALGO_CONFIG,
            "learning_rate": 1.0128e-4,
            "entropy_coef": 0.0408,
            "num_epochs": 20,
        },
        "ctro_algo": {
            **DMCONTROL_CTRO_ALGO_CONFIG,
            "learning_rate": 1.0128e-4,
            "entropy_coef": 0.0408,
            "num_epochs": 20,
            "vae_coef": 0.0,
            "alpha": 0.00205,
            "beta": 0.532,
            "alpha_warmup_epochs": 500,
            "beta_warmup_epochs": 500,
        },
        "results_prefix": "dmcontrol_pixels",
    },
}
