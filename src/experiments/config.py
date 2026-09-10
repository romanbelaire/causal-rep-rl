"""Shared hyperparameters for CTRO Minigrid experiments."""

BASE_ALGO_CONFIG = {
    "learning_rate": 1e-4,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "vae_coef": 0.1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "use_policy_clip": True,
    "max_grad_norm": 0.2,
    "batch_size": 256,
    "num_epochs": 3,
    "mu_0": 0.1,
    "beta_mico": 0.1,
    "pl_eps": 1e-4,
    "f_floor": 1e-3,
    "tau_ref": 0.01,
    "v_ref_quantile": 0.99,
    "mico_huber_delta": 1.0,
    "mico_target_update_tau": 0.005,
    # D_Z reference-scaled log-geometry trust region (off by default)
    "dz_enabled": False,
    "lambda_dz": 1.0,
    "eta_dz": 0.05,  # recalibrate via λ=0 run: sqrt(p25 of early D_Z)
    "dz_adapt": True,
    "dz_eta_adapt": False,
    "eta_dz_min": None,  # defaults to eta_dz when unset
    "eta_dz_c": 1.5,
    "dz_ema_tau": 0.05,
    "dz_ema_init": 0.5,  # formation-scale D_Z seed so first η is permissive
    "lambda_loc": 0.7,
    "dz_delta": 0.01,  # absolute δ on d_bar = dist / (sqrt(d) * s_ref)
    "dz_collapse_target_thresh": 0.1,
    "dz_n_pairs": 2048,
    "ref_buffer_size": 2048,
    "ref_refresh_factor": 2.0,
    "head_phasing": False,
    "enc_epochs": 4,
    "val_epochs": 4,
    # PL hinge plumbing (anti-aliased PPO vs legacy CTRO)
    "pl_scale_invariant": False,
    "pl_value_normalize": False,
    "pl_f_mode": "exclude",  # "exclude" (legacy) | "clamp" (live)
    "pl_on_ref_buffer": False,  # True: freeze N states once; hinge on that batch
    "pl_f_min": 1e-3,  # absolute clamp when pl_f_mode == "clamp"
    "value_spectral_norm": False,
    "actor_updates_encoder": True,
    "target_kl": None,
}

# Named recipes — see docs/METHOD_CHANGELOG.md
ALGO_PRESETS = {
    # Live method: PPO + scale-corrected one-sided PL hinge on a frozen ref batch.
    "anti_aliased_ppo": {
        "alpha": 0.0,
        "beta": 0.532,
        "alpha_warmup_epochs": 0,
        "beta_warmup_epochs": 500,
        "dz_enabled": False,
        "value_spectral_norm": False,
        "pl_scale_invariant": True,
        "pl_value_normalize": False,
        "pl_f_mode": "clamp",
        "pl_on_ref_buffer": True,
        "pl_f_min": 1e-3,
        "head_phasing": True,
        "enc_epochs": 4,
        "val_epochs": 4,
        "ref_buffer_size": 2048,
        "mu_0": 0.1,
        "f_floor": 1e-3,
        "tau_ref": 0.01,
        "v_ref_quantile": 0.99,
    },
    # Pre-rejection CTRO stack (MICo + PL on minibatch; D_Z still opt-in via flags).
    "ctro_full_legacy": {
        "alpha": 0.00205,
        "beta": 0.532,
        "alpha_warmup_epochs": 500,
        "beta_warmup_epochs": 500,
        "dz_enabled": False,
        "value_spectral_norm": False,
        "pl_scale_invariant": True,
        "pl_value_normalize": True,
        "pl_f_mode": "exclude",
        "pl_on_ref_buffer": False,
        "head_phasing": False,
        "mu_0": 0.1,
        "f_floor": 1e-3,
        "tau_ref": 0.01,
        "v_ref_quantile": 0.99,
    },
    "active_ctro": {
        "alpha": 0.0,
        "beta": 0.532,
        "alpha_warmup_epochs": 0,
        "beta_warmup_epochs": 500,
        "dz_enabled": False,
        "value_spectral_norm": False,
        "pl_scale_invariant": True,
        "pl_value_normalize": False,
        "pl_f_mode": "clamp",
        "pl_on_ref_buffer": True,
        "pl_f_min": 1e-3,
        "head_phasing": True,
        "enc_epochs": 4,
        "val_epochs": 4,
        "ref_buffer_size": 2048,
        "mu_0": 0.1,
        "f_floor": 1e-3,
        "tau_ref": 0.01,
        "v_ref_quantile": 0.99,
        "vae_coef": 0.0,
        "actor_updates_encoder": False,
    },
}


def active_ctro_block(
    *,
    enabled: bool,
    q_ensemble_on: bool = True,
    mico_on: bool = True,
    information_critic_on: bool = True,
    query_on: bool = True,
    separation_on: bool = False,
    relational_on: bool = True,
    relational_mode: str = "observe",
) -> dict:
    """Nested active-CTRO family. Do not overload legacy alpha (random-pair MICo)."""
    q_on = enabled and q_ensemble_on
    u_on = enabled and information_critic_on
    query = enabled and query_on
    mico_coef = 1.0 if (enabled and mico_on) else 0.0
    return {
        "enabled": enabled,
        "n_aux_updates": 4,
        "query_rollout_size": 1024,
        "replay": {
            "capacity": 200000,
            "batch_size": 64,
            "coverage_mix": 0.10,
            "priority_floor": 0.001,
            "importance_exponent": 0.4,
        },
        "q_ensemble": {
            "enabled": q_on,
            "members": 5,
            "hidden_sizes": [256, 256],
            "target_update": "hard",
            "target_interval_phases": 1,
            "td_huber_delta": 1.0,
        },
        "mico_ac": {
            "coefficient": mico_coef,
            "pair_metric": "angular_diffuse",
            "beta_mico": 0.1,
            "huber_delta": 1.0,
            "positive_ucb_threshold": 0.1,
            "negative_lcb_threshold": 0.3,
            "confidence_z": 1.0,
            "warmup_aux_steps": 10,
            "n_candidates": 8,
        },
        "separation": {
            "enabled": enabled and separation_on,
            "coefficient": 1.0 if (enabled and separation_on) else 0.0,
            "alpha_sep": 1.0,
            "cov_eig_floor": 1e-6,
        },
        "information_critic": {
            "enabled": u_on,
            "gamma_u": 0.95,
            "temperature": 0.25,
            "lambda_q": 1.0,
            "lambda_gap": 0.5,
            "lambda_pair": 1.0,
            "hidden_sizes": [256, 256],
        },
        "exploration": {
            "enabled": query,
            "query_mix_epsilon": 0.15 if query else 0.0,
            "uniform_floor_eta": 0.05 if query else 0.0,
        },
        "relational_tr": {
            "enabled": enabled and relational_on,
            "mode": relational_mode,
            "reference_size": 512,
            "min_old_distance": 1e-3,
            "epsilon_z": 0.10,
            "share_pl_ref_batch": True,
        },
        "priorities": {
            "lambda_ig": 1.0,
            "lambda_gap": 0.5,
            "lambda_pair": 1.0,
            "lambda_geom": 0.5,
        },
    }


BASE_ALGO_CONFIG["active"] = active_ctro_block(enabled=False)


def matrix_row_algo(row: int) -> dict:
    """Incremental MiniGrid matrix (Stage 5). Rows 1–7; stop on first failure."""
    base = {
        **BASE_ALGO_CONFIG,
        **ALGO_PRESETS["anti_aliased_ppo"],
    }
    if row == 1:
        return {
            **base,
            "actor_updates_encoder": True,
            "active": active_ctro_block(enabled=False),
        }
    if row == 2:
        return {
            **base,
            "actor_updates_encoder": True,
            "active": active_ctro_block(enabled=False),
        }
    if row == 3:
        return {
            **base,
            "alpha": 0.0,
            "vae_coef": 0.0,
            "actor_updates_encoder": False,
            "active": active_ctro_block(
                enabled=True,
                q_ensemble_on=True,
                mico_on=False,
                information_critic_on=False,
                query_on=False,
                separation_on=False,
                relational_on=False,
            ),
        }
    if row == 4:
        return {
            **base,
            "alpha": 0.0,
            "vae_coef": 0.0,
            "actor_updates_encoder": False,
            "active": active_ctro_block(
                enabled=True,
                q_ensemble_on=True,
                mico_on=True,
                information_critic_on=False,
                query_on=False,
                separation_on=False,
                relational_on=False,
            ),
        }
    if row == 5:
        return {
            **base,
            "alpha": 0.0,
            "vae_coef": 0.0,
            "actor_updates_encoder": False,
            "active": active_ctro_block(
                enabled=True,
                q_ensemble_on=True,
                mico_on=True,
                information_critic_on=False,
                query_on=False,
                separation_on=True,
                relational_on=False,
            ),
        }
    if row == 6:
        return {
            **base,
            "alpha": 0.0,
            "vae_coef": 0.0,
            "actor_updates_encoder": False,
            "active": active_ctro_block(
                enabled=True,
                q_ensemble_on=True,
                mico_on=True,
                information_critic_on=True,
                query_on=True,
                separation_on=True,
                relational_on=False,
            ),
        }
    if row == 7:
        return {
            **base,
            "alpha": 0.0,
            "vae_coef": 0.0,
            "actor_updates_encoder": False,
            "active": active_ctro_block(
                enabled=True,
                q_ensemble_on=True,
                mico_on=True,
                information_critic_on=True,
                query_on=True,
                separation_on=True,
                relational_on=True,
                relational_mode="observe",
            ),
        }
    raise RuntimeError(f"matrix row must be 1–7, got {row}")


E0_VANILLA_PPO = {
    **BASE_ALGO_CONFIG,
    "alpha": 0.0,
    "beta": 0.0,
    "actor_updates_encoder": True,
    "active": active_ctro_block(enabled=False),
}

E0_ANTI_ALIASED_PPO = {
    **BASE_ALGO_CONFIG,
    **ALGO_PRESETS["anti_aliased_ppo"],
    "actor_updates_encoder": True,
    "active": active_ctro_block(enabled=False),
}

ACTIVE_CTRO_MINIGRID = {
    **BASE_ALGO_CONFIG,
    **ALGO_PRESETS["anti_aliased_ppo"],
    "alpha": 0.0,
    "vae_coef": 0.0,
    "actor_updates_encoder": False,
    "active": active_ctro_block(enabled=True),
}


def apply_algo_preset(name: str) -> dict:
    if name not in ALGO_PRESETS:
        raise KeyError(
            f"Unknown algo preset {name!r}; choose from {sorted(ALGO_PRESETS)}"
        )
    preset = dict(ALGO_PRESETS[name])
    if name == "anti_aliased_ppo":
        if preset["alpha"] != 0.0:
            raise RuntimeError("anti_aliased_ppo requires alpha=0")
        if preset["dz_enabled"]:
            raise RuntimeError("anti_aliased_ppo requires dz_enabled=False")
        if preset["value_spectral_norm"]:
            raise RuntimeError("anti_aliased_ppo requires value_spectral_norm=False")
        if not preset["pl_scale_invariant"]:
            raise RuntimeError("anti_aliased_ppo requires pl_scale_invariant=True")
        if preset["pl_value_normalize"]:
            raise RuntimeError("anti_aliased_ppo requires pl_value_normalize=False")
        if preset["pl_f_mode"] != "clamp":
            raise RuntimeError("anti_aliased_ppo requires pl_f_mode='clamp'")
        if not preset["pl_on_ref_buffer"]:
            raise RuntimeError("anti_aliased_ppo requires pl_on_ref_buffer=True")
        if not preset["head_phasing"]:
            raise RuntimeError("anti_aliased_ppo requires head_phasing=True")
    if name == "active_ctro":
        if preset["alpha"] != 0.0:
            raise RuntimeError("active_ctro requires alpha=0 (legacy random-pair MICo off)")
        if preset["actor_updates_encoder"]:
            raise RuntimeError("active_ctro requires actor_updates_encoder=False")
        preset["active"] = active_ctro_block(enabled=True)
    return preset


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
    "shared_ref_path": "",
    "dump_ref_path": "",
    "diag_log_epochs": (),
}

E0_TRAINING_SMOKE = {
    **BASE_TRAINING_CONFIG,
    "buffer_size": 256,
    "total_epochs": 3,
    "checkpoint_frequency": 3,
    "eval_frequency": 3,
    "eval_episodes": 2,
    "log_interval_steps": 256,
}

ACTIVE_TRAINING_SMOKE = {
    **E0_TRAINING_SMOKE,
    "query_rollout_size": 128,
}

DEFAULT_SEEDS = [42, 43, 44]

# Active-CTRO MiniGrid pipeline (Unlock-v0). Full budget is BASE_TRAINING_CONFIG.
ACTIVE_CTRO_SEEDS = [42, 43, 44]
ACTIVE_CTRO_MINIGRID_ARMS = ("exp_e0_vanilla", "exp_e0_aa_ppo", "exp_active_ctro")
ACTIVE_CTRO_RESULTS_MINIGRID = "results/active_ctro/minigrid"
ACTIVE_CTRO_RESULTS_TOYS = "results/active_ctro/toys"
ACTIVE_CTRO_RESULTS_DIAG = "results/active_ctro/diagnostics"
ACTIVE_CTRO_RESULTS_MATRIX = "results/active_ctro/matrix"

MATRIX_TRAINING = {
    **BASE_TRAINING_CONFIG,
    "diag_log_epochs": (0, 50, 100, 200, 500),
}

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
    "metric_pl_max_samples": None,  # full buffer — floor-rate / μ_PL need the batch, not n=64
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
    "collapse_min_steps": 200_000,
    "collapse_streak": 3,
    # Save GPU when return plateaus after warm-up (pixels/state 8M ceiling).
    "early_stop_enabled": True,
    "early_stop_min_steps": 1_000_000,
    "early_stop_patience": 25,
    "early_stop_min_delta": 1.0,
    "early_stop_metric": "eval_full_return_mean",
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
    "ale": {
        "arch": {
            "mode": "nature_cnn",
            "cnn_family": "nature",
            "critic": {"type": "nature"},
            "policy": {"type": "nature"},
        },
        "training": {
            **BASE_TRAINING_CONFIG,
            "buffer_size": 8 * 128,
            "total_epochs": 100_000_000 // (8 * 128),
            "eval_frequency": 488,  # ~15 evals over 100M / 1024 ≈ 97656 updates; 97656/488≈200 — smoke uses override
            "eval_episodes": 10,
            "metric_pl_max_samples": 32,
            "print_every_epochs": 50,
            "num_envs": 8,
            "obs_norm": "none",
            "reward_norm": "none",
            "collapse_min_steps": None,
            "collapse_floor": None,
            "early_stop_enabled": False,
            "log_interval_steps": 50_000,
            "checkpoint_frequency": 2000,
        },
        "ppo_algo": {
            "learning_rate": 2.5e-4,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "vae_coef": 0.0,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.1,
            "use_policy_clip": True,
            "max_grad_norm": 0.5,
            "batch_size": 256,
            "num_epochs": 4,
            "alpha": 0.0,
            "beta": 0.0,
            "dz_n_pairs": 256,
            "anneal_lr": True,
            "norm_adv": True,
            "pfo_coef": 0.0,
        },
        "ctro_algo": {
            "learning_rate": 2.5e-4,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "vae_coef": 0.0,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.1,
            "use_policy_clip": True,
            "max_grad_norm": 0.5,
            "batch_size": 256,
            "num_epochs": 4,
            "alpha": 0.00205,
            "beta": 0.532,
            "alpha_warmup_epochs": 0,
            "beta_warmup_epochs": 0,
            "anneal_lr": True,
            "norm_adv": True,
            "pfo_coef": 0.0,
            "dz_n_pairs": 256,
        },
        "results_prefix": "ale",
    },
}
