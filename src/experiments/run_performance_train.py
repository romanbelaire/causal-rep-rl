"""Train a single Procgen or DMControl performance environment task."""

import argparse

from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.evaluation.suites import EVAL_SUITES
from src.experiments.config import apply_algo_preset
from src.experiments.performance_runner import run_performance_train


def _parse_hidden(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"empty hidden sizes: {s!r}")
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Train on a performance suite task")
    parser.add_argument("--suite", type=str, required=True, choices=list(EVAL_SUITES.keys()))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp-name", type=str, default="exp_full")
    parser.add_argument("--agent", type=str, default="ctro", choices=["ctro", "ppo", "pfo"])
    parser.add_argument(
        "--pfo-coef",
        type=float,
        default=None,
        help="PFO coefficient on actor_preactivation (ALE default 1.0 when --agent pfo).",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=None,
        help="Override training.eval_frequency (epochs between eval checkpoints).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override training.eval_episodes.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Override training.buffer_size (rollout length).",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from run_dir/weights_latest.pt (optimizer + step/epoch state).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Force fresh run even if weights_latest.pt exists (Slurm jobs default to auto-resume).",
    )
    parser.add_argument(
        "--load-freeze",
        type=str,
        default=None,
        help="Path to freeze JSON; sets FREEZE_ID and loads LTRO keys when dz_enabled.",
    )
    parser.add_argument(
        "--algo-preset",
        type=str,
        default=None,
        choices=["anti_aliased_ppo", "ctro_full_legacy"],
        help="Named recipe (see docs/METHOD_CHANGELOG.md). CLI flags override preset keys.",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--alpha-warmup-epochs",
        type=int,
        default=None,
        help="Linear ramp of alpha over this many training epochs (0 = no warmup).",
    )
    parser.add_argument(
        "--beta-warmup-epochs",
        type=int,
        default=None,
        help="Linear ramp of beta over this many training epochs (0 = no warmup).",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--policy-hidden",
        type=_parse_hidden,
        default=None,
        help="Comma-separated MLP hidden sizes, e.g. 64,64 or 256,256",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override train budget in env steps (epochs = total_steps // buffer_size).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override vectorized env count (default: suite config; 1 = serial rollout).",
    )
    # D_Z / constraint flags
    parser.add_argument("--dz-enabled", action="store_true", default=False)
    parser.add_argument("--eta-dz", type=float, default=None)
    parser.add_argument("--lambda-dz", type=float, default=None)
    parser.add_argument(
        "--no-dz-adapt",
        action="store_true",
        default=False,
        help="Freeze lambda_dz (use with --lambda-dz 0 to log D_Z without penalty).",
    )
    parser.add_argument(
        "--dz-eta-adapt",
        action="store_true",
        default=False,
        help="Adaptive radius η_t=max(η_min, c*EMA(D_Z)); floored at --eta-dz-min/--eta-dz.",
    )
    parser.add_argument(
        "--eta-dz-min",
        type=float,
        default=None,
        help="Floor for adaptive η (default: --eta-dz). Calibrated settled rate.",
    )
    parser.add_argument(
        "--eta-dz-c",
        type=float,
        default=None,
        help="Multiplier on pre-update D_Z EMA for adaptive η (default 1.5).",
    )
    parser.add_argument(
        "--dz-ema-tau",
        type=float,
        default=None,
        help="EMA timescale for realized D_Z (default 0.05; slow vs PPO updates).",
    )
    parser.add_argument(
        "--dz-ema-init",
        type=float,
        default=None,
        help="Initial D_Z EMA seed so first η is permissive (default 0.5).",
    )
    parser.add_argument(
        "--dz-delta",
        type=float,
        default=None,
        help="Absolute log-ratio delta on d_bar (default 0.01).",
    )
    parser.add_argument("--no-policy-clip", action="store_true", default=False)
    parser.add_argument("--head-phasing", action="store_true", default=False)
    parser.add_argument(
        "--pl-scale-invariant",
        action="store_true",
        default=False,
        help="PL hinge uses mu_PL * L^2 (T1.1); L = stopgrad median pairwise distance.",
    )
    parser.add_argument(
        "--pl-value-normalize",
        action="store_true",
        default=False,
        help="With --pl-scale-invariant, also divide by stopgrad(V range) (T1.2).",
    )
    parser.add_argument(
        "--pl-f-mode",
        type=str,
        default=None,
        choices=["exclude", "clamp"],
        help="PL denominator: exclude floored states (legacy) or clamp_min (live).",
    )
    parser.add_argument(
        "--pl-on-ref-buffer",
        action="store_true",
        default=False,
        help="Freeze N=ref_buffer_size states once; evaluate PL hinge on that batch.",
    )
    parser.add_argument(
        "--pl-f-min",
        type=float,
        default=None,
        help="Absolute f clamp when --pl-f-mode clamp (default 1e-3).",
    )
    parser.add_argument(
        "--value-spectral-norm",
        action="store_true",
        default=False,
        help="Spectral-normalize Linear layers in critic.value_head (E2 arm C).",
    )
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="Load policy/critic (and v_ref if present) from this .pt before training (E-A.5).",
    )
    parser.add_argument("--enc-epochs", type=int, default=None)
    parser.add_argument("--val-epochs", type=int, default=None)
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        default=False,
        help="Disable return-collapse prune (diagnostics / D_Z sweeps).",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        default=False,
        help="Disable plateau early-stop (always run full total_steps budget).",
    )
    parser.add_argument(
        "--early-stop-min-steps",
        type=int,
        default=None,
        help="Override plateau early-stop min env steps before stopping is allowed.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Override # of logged metrics without improvement before stop.",
    )
    args = parser.parse_args()

    if args.task not in EVAL_SUITES[args.suite].tasks:
        raise ValueError(f"Unknown task {args.task} for suite {args.suite}")

    if args.load_freeze is not None:
        import os
        from pathlib import Path

        freeze_file = Path(args.load_freeze)
        freeze_id = freeze_file.stem
        os.environ["FREEZE_ID"] = freeze_id

    agent_cls = CTRO if args.agent == "ctro" else PPO
    algo_overrides: dict = {}
    if args.algo_preset is not None:
        algo_overrides.update(apply_algo_preset(args.algo_preset))
        if args.algo_preset == "anti_aliased_ppo":
            agent_cls = CTRO
    if args.agent == "ctro":
        if args.alpha is not None:
            algo_overrides["alpha"] = args.alpha
        if args.beta is not None:
            algo_overrides["beta"] = args.beta
        if args.alpha_warmup_epochs is not None:
            algo_overrides["alpha_warmup_epochs"] = args.alpha_warmup_epochs
        if args.beta_warmup_epochs is not None:
            algo_overrides["beta_warmup_epochs"] = args.beta_warmup_epochs
    else:
        algo_overrides.update({"alpha": 0.0, "beta": 0.0, "vae_coef": 0.0})

    if args.agent == "pfo":
        agent_cls = PPO
        algo_overrides["pfo_coef"] = 1.0 if args.pfo_coef is None else float(args.pfo_coef)
    elif args.pfo_coef is not None:
        algo_overrides["pfo_coef"] = float(args.pfo_coef)

    if args.learning_rate is not None:
        algo_overrides["learning_rate"] = args.learning_rate
    if args.entropy_coef is not None:
        algo_overrides["entropy_coef"] = args.entropy_coef
    if args.num_epochs is not None:
        algo_overrides["num_epochs"] = args.num_epochs
    if args.clip_epsilon is not None:
        algo_overrides["clip_epsilon"] = args.clip_epsilon
    if args.dz_enabled:
        algo_overrides["dz_enabled"] = True
        agent_cls = CTRO
    if args.eta_dz is not None:
        algo_overrides["eta_dz"] = args.eta_dz
    if args.lambda_dz is not None:
        algo_overrides["lambda_dz"] = args.lambda_dz
    if args.no_dz_adapt:
        algo_overrides["dz_adapt"] = False
    if args.dz_eta_adapt:
        algo_overrides["dz_eta_adapt"] = True
    if args.eta_dz_min is not None:
        algo_overrides["eta_dz_min"] = args.eta_dz_min
    if args.eta_dz_c is not None:
        algo_overrides["eta_dz_c"] = args.eta_dz_c
    if args.dz_ema_tau is not None:
        algo_overrides["dz_ema_tau"] = args.dz_ema_tau
    if args.dz_ema_init is not None:
        algo_overrides["dz_ema_init"] = args.dz_ema_init
    if args.dz_delta is not None:
        algo_overrides["dz_delta"] = args.dz_delta
    if args.no_policy_clip:
        algo_overrides["use_policy_clip"] = False
    if args.head_phasing:
        algo_overrides["head_phasing"] = True
    if args.pl_scale_invariant:
        algo_overrides["pl_scale_invariant"] = True
    if args.pl_value_normalize:
        algo_overrides["pl_value_normalize"] = True
    if args.pl_f_mode is not None:
        algo_overrides["pl_f_mode"] = args.pl_f_mode
    if args.pl_on_ref_buffer:
        algo_overrides["pl_on_ref_buffer"] = True
    if args.pl_f_min is not None:
        algo_overrides["pl_f_min"] = args.pl_f_min
    if args.value_spectral_norm:
        algo_overrides["value_spectral_norm"] = True
    if args.enc_epochs is not None:
        algo_overrides["enc_epochs"] = args.enc_epochs
    if args.val_epochs is not None:
        algo_overrides["val_epochs"] = args.val_epochs

    arch_overrides = None
    if args.policy_hidden is not None:
        arch_overrides = {"policy_hidden": args.policy_hidden}

    train_overrides: dict = {}
    if args.total_steps is not None:
        train_overrides["total_steps"] = args.total_steps
    if args.eval_frequency is not None:
        train_overrides["eval_frequency"] = args.eval_frequency
    if args.eval_episodes is not None:
        train_overrides["eval_episodes"] = args.eval_episodes
    if args.buffer_size is not None:
        train_overrides["buffer_size"] = args.buffer_size
    if args.no_collapse:
        train_overrides["collapse_floor"] = None
    if args.no_early_stop:
        train_overrides["early_stop_enabled"] = False
    if args.early_stop_min_steps is not None:
        train_overrides["early_stop_min_steps"] = args.early_stop_min_steps
    if args.early_stop_patience is not None:
        train_overrides["early_stop_patience"] = args.early_stop_patience

    result = run_performance_train(
        suite_name=args.suite,
        task=args.task,
        seed=args.seed,
        exp_name=args.exp_name,
        agent_cls=agent_cls,
        algo_overrides=algo_overrides,
        arch_overrides=arch_overrides,
        train_overrides=train_overrides or None,
        results_root=args.results_root,
        device=args.device,
        num_envs=args.num_envs,
        init_weights=args.init_weights,
        resume=args.resume,
    )
    print(
        f"TrainResult status={result.status} steps={result.total_steps} "
        f"eval_full={result.eval_full_return_mean}",
        flush=True,
    )


if __name__ == "__main__":
    main()
