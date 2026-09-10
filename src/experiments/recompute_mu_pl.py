"""Offline recompute of new mu_PL on a finished checkpoint (CPU, deterministic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.experiments.performance_models import build_performance_stack_from_config
from src.metrics.pl_ratio import compute_mu_pl_bootstrap, update_v_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute mu_PL with value-suboptimality denom")
    parser.add_argument("--run-dir", type=str, required=True, help="seed_*/task run directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--f-floor", type=float, default=1e-3)
    parser.add_argument("--tau-ref", type=float, default=0.01)
    args = parser.parse_args()

    if args.device != "cpu":
        raise SystemExit("recompute_mu_pl must run on CPU (no GPU)")

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    weights_path = run_dir / "weights_final.pt"
    if not weights_path.exists():
        weights_path = run_dir / "weights_latest.pt"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Need config.json and weights in {run_dir}")

    with open(config_path) as f:
        config = json.load(f)

    device = args.device
    obs_shape = config.get("obs_shape")
    obs_dim = None
    if obs_shape is None:
        # state-based: reload from architecture — use first Linear weight
        ckpt_peek = torch.load(weights_path, map_location="cpu")
        # MLPEncoderCritic: encoder.0.weight is [h, obs_dim]
        w = ckpt_peek["critic"]["encoder.0.weight"]
        obs_dim = int(w.shape[1])
        action_dim = int(ckpt_peek["policy"]["action_mean.weight"].shape[0])
        action_space_type = "continuous"
        obs_shape_t = None
    else:
        obs_shape_t = tuple(obs_shape)
        obs_dim = 0
        action_dim = 0
        action_space_type = "discrete"
        ckpt_peek = torch.load(weights_path, map_location="cpu")
        if "action_mean.weight" in ckpt_peek["policy"]:
            action_dim = int(ckpt_peek["policy"]["action_mean.weight"].shape[0])
            action_space_type = "continuous"
        else:
            action_dim = int(ckpt_peek["policy"]["action_head.weight"].shape[0])

    stack = build_performance_stack_from_config(
        {
            "architecture": config["architecture"],
            "agent_class": config["agent_class"],
            "stack_type": config.get("stack_type"),
        },
        obs_dim=obs_dim if obs_shape is None else 0,
        action_dim=action_dim,
        action_space_type=action_space_type,
        obs_shape=obs_shape_t,
        device=device,
    )
    ckpt = torch.load(weights_path, map_location=device)
    stack.critic.load_state_dict(ckpt["critic"])
    stack.policy.load_state_dict(ckpt["policy"])
    stack.critic.eval()

    # Synthetic normalized states if no probe cache: random unit Gaussian.
    if not hasattr(stack.critic, "encode"):
        raise RuntimeError("Critic must support encode() for mu_PL recompute")

    obs_dim_use = stack.critic.encoder[0].in_features
    n = args.max_samples
    torch.manual_seed(0)
    obs = torch.randn(n, obs_dim_use, device=device)

    with torch.no_grad():
        z, _ = stack.critic.encode(obs)
        v = stack.critic.value_head(z).squeeze(-1)
        v_ref = update_v_ref(None, v, tau_ref=args.tau_ref)
        # one more EMA update for stability report
        v_ref = update_v_ref(v_ref, v, tau_ref=args.tau_ref)

    metrics = compute_mu_pl_bootstrap(
        stack.critic, z, v_ref=v_ref, f_floor=args.f_floor, max_samples=None
    )

    finite = all(
        isinstance(v, float) and (v == v) and abs(v) != float("inf")
        for k, v in metrics.items()
        if k.startswith("mu_pl")
    )
    if not finite:
        raise RuntimeError(f"Non-finite mu_PL metrics: {metrics}")
    if metrics["f_floor_rate"] >= 0.01:
        print(
            f"WARNING: f_floor mask rate {100*metrics['f_floor_rate']:.2f}% "
            f"(P0.1 acceptance target <1%; mu_PL uses unmasked samples only, "
            f"n_valid={int(metrics['mu_pl_n_valid'])}, "
            f"f_floor_eff={metrics.get('f_floor_eff', float('nan'))})"
        )
    if metrics["mu_pl_inf"] < 0:
        raise RuntimeError("mu_PL has negative values")

    out = {
        "run_dir": str(run_dir),
        "accept_finite_nonneg": True,
        **metrics,
    }
    out_path = run_dir / "mu_pl_recompute.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
