"""E-A shear / gauge experiment (CPU). Compensated reparameterization of Z.

Applies A = diag(1,c,1,...,1) to encoder outputs and A^{-1} to the first linear
layers of the policy and value heads. Reports E-A.1 behavioural invariance,
E-A.3 mu_t vs c, E-A.4 separation vs c, and E-A.6 orthogonal gauge.

Must terminate on CPU (no GPU).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.experiments.diagnose_latent_geometry import encode_probe, load_stack
from src.metrics.pl_ratio import (
    compute_mu_pl_bootstrap,
    median_pairwise_distance,
    update_v_ref,
)


DELTA_BEHAVIOURAL = 1e-5


def _diag_scale(dim: int, c: float, device: str) -> torch.Tensor:
    scale = torch.ones(dim, device=device)
    if dim < 2:
        raise RuntimeError(f"Need latent dim >= 2 for shear (got {dim})")
    scale[1] = c
    return scale


def _first_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, nn.Sequential):
        for m in module:
            if isinstance(m, nn.Linear):
                return m
    raise RuntimeError(f"Cannot find first Linear in {type(module)}")


def _policy_input_linear(policy: nn.Module) -> nn.Linear:
    if hasattr(policy, "input_layer") and isinstance(policy.input_layer, nn.Linear):
        return policy.input_layer
    if hasattr(policy, "feature_extractor"):
        return _first_linear(policy.feature_extractor)
    raise RuntimeError(f"Unsupported policy type for shear: {type(policy)}")


def _value_input_linear(critic: nn.Module) -> nn.Linear:
    return _first_linear(critic.value_head)


def _scale_encoder_output_inplace(critic: nn.Module, scale: torch.Tensor) -> None:
    """Left-multiply encoder output by diag(scale): z' = z * scale."""
    if hasattr(critic, "fc_z") and isinstance(critic.fc_z, nn.Linear):
        # CNN: z = fc_z(h); scale rows of W and bias
        with torch.no_grad():
            critic.fc_z.weight.mul_(scale.unsqueeze(1))
            critic.fc_z.bias.mul_(scale)
        return
    if hasattr(critic, "encoder") and isinstance(critic.encoder, nn.Sequential):
        # MLP: last Linear before final activation — scale that Linear's outputs
        linears = [m for m in critic.encoder if isinstance(m, nn.Linear)]
        last = linears[-1]
        with torch.no_grad():
            last.weight.mul_(scale.unsqueeze(1))
            last.bias.mul_(scale)
        return
    raise RuntimeError(f"Cannot scale encoder output for critic type {type(critic)}")


def _scale_head_input_inplace(linear: nn.Linear, inv_scale: torch.Tensor) -> None:
    """Right-multiply W by diag(inv_scale): columns of W scaled."""
    with torch.no_grad():
        linear.weight.mul_(inv_scale.unsqueeze(0))


def apply_compensated_shear(stack, c: float, device: str) -> None:
    z_dim = stack.critic.latent_dim
    scale = _diag_scale(z_dim, c, device)
    inv = 1.0 / scale
    _scale_encoder_output_inplace(stack.critic, scale)
    _scale_head_input_inplace(_value_input_linear(stack.critic), inv)
    _scale_head_input_inplace(_policy_input_linear(stack.policy), inv)


def apply_orthogonal_gauge(stack, device: str, seed: int = 0) -> None:
    z_dim = stack.critic.latent_dim
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(z_dim, z_dim, generator=g))
    q = q.to(device)
    # Encoder: z' = Q z. For Linear last/fc_z: W <- Q W, b <- Q b
    if hasattr(stack.critic, "fc_z") and isinstance(stack.critic.fc_z, nn.Linear):
        with torch.no_grad():
            stack.critic.fc_z.weight.copy_(q @ stack.critic.fc_z.weight)
            stack.critic.fc_z.bias.copy_(q @ stack.critic.fc_z.bias)
    else:
        linears = [m for m in stack.critic.encoder if isinstance(m, nn.Linear)]
        last = linears[-1]
        with torch.no_grad():
            last.weight.copy_(q @ last.weight)
            last.bias.copy_(q @ last.bias)
    # Heads: W <- W Q^{-1} = W Q^T
    qt = q.T
    for linear in (_value_input_linear(stack.critic), _policy_input_linear(stack.policy)):
        with torch.no_grad():
            linear.weight.copy_(linear.weight @ qt)


def _policy_outputs(policy: nn.Module, z: torch.Tensor) -> torch.Tensor:
    out = policy(z)
    if isinstance(out, tuple):
        return out[0]
    return out


def behavioural_max_rel_err(
    stack0, stack1, obs: torch.Tensor
) -> dict[str, float]:
    with torch.no_grad():
        z0, _ = stack0.critic.encode(obs)
        z1, _ = stack1.critic.encode(obs)
        v0 = stack0.critic.value_head(z0).squeeze(-1)
        v1 = stack1.critic.value_head(z1).squeeze(-1)
        p0 = _policy_outputs(stack0.policy, z0)
        p1 = _policy_outputs(stack1.policy, z1)
    denom_v = v0.abs().clamp_min(1e-8)
    denom_p = p0.abs().clamp_min(1e-8)
    return {
        "max_abs_value": float((v0 - v1).abs().max()),
        "max_rel_value": float(((v0 - v1).abs() / denom_v).max()),
        "max_abs_policy": float((p0 - p1).abs().max()),
        "max_rel_policy": float(((p0 - p1).abs() / denom_p).max()),
    }


def geometry_stats(critic, z: torch.Tensor, v_ref: float) -> dict[str, float]:
    with torch.no_grad():
        L = float(median_pairwise_distance(z).item())
        dmat = torch.cdist(z, z)
        iu = torch.triu_indices(z.shape[0], z.shape[0], offset=1)
        dists = dmat[iu[0], iu[1]]
        p05 = float(torch.quantile(dists, 0.05))
        p05_scaled = p05 / max(L, 1e-12)
    pl = compute_mu_pl_bootstrap(critic, z, v_ref=v_ref)
    return {
        "L": L,
        "pair_p05": p05,
        "pair_p05_over_L": p05_scaled,
        "mu_pl_tilde_q05": float(pl["mu_pl_tilde_q05"]),
        "mu_pl_q05": float(pl["mu_pl_q05"]),
    }


def fit_log_slope(cs: list[float], mus: list[float]) -> dict[str, float]:
    x = np.log(np.asarray(cs, dtype=float))
    y = np.log(np.asarray(mus, dtype=float))
    if np.any(~np.isfinite(y)):
        raise RuntimeError(f"Non-finite mu_t in fit: {mus}")
    slope, intercept = np.polyfit(x, y, 1)
    # Crude analytical SE for slope under homoscedastic OLS
    yhat = slope * x + intercept
    resid = y - yhat
    n = len(x)
    s2 = float(np.sum(resid**2) / max(n - 2, 1))
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "se": se,
        "ci95_lo": float(slope - 1.96 * se),
        "ci95_hi": float(slope + 1.96 * se),
    }


def clone_stack(stack):
    return copy.deepcopy(stack)


def main() -> None:
    parser = argparse.ArgumentParser(description="E-A shear / gauge (CPU)")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--c-values",
        type=str,
        default="1,2,5,10,30",
        help="Comma-separated shear factors",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--write-sheared-ckpts",
        action="store_true",
        default=False,
        help="Write weights_final.pt for each c under output-dir/c_{c}/ for E-A.5 resume.",
    )
    args = parser.parse_args()

    if args.device != "cpu":
        raise SystemExit("shear_gauge_experiment must run on CPU (no GPU)")

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "shear_ea"
    out_dir.mkdir(parents=True, exist_ok=True)

    base, config, ckpt = load_stack(run_dir, device=args.device)
    obs = None
    torch.manual_seed(0)
    obs_shape = config.get("obs_shape")
    if obs_shape is not None:
        c, h, w = int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2])
        obs = torch.rand(args.n, c, h, w, device=args.device)
    else:
        first = base.critic.encoder[0]
        obs = torch.randn(args.n, first.in_features, device=args.device)

    with torch.no_grad():
        z0, _ = base.critic.encode(obs)
        v0 = base.critic.value_head(z0).squeeze(-1)
    v_ref = update_v_ref(None, v0)

    cs = [float(x) for x in args.c_values.split(",") if x.strip()]
    shear_rows = []
    for c in cs:
        sheared = clone_stack(base)
        apply_compensated_shear(sheared, c, args.device)
        beh = behavioural_max_rel_err(base, sheared, obs)
        with torch.no_grad():
            z_s, _ = sheared.critic.encode(obs)
        geo = geometry_stats(sheared.critic, z_s, v_ref)
        row = {"c": c, **beh, **geo}
        shear_rows.append(row)
        print(
            f"c={c}: rel_v={beh['max_rel_value']:.3e} rel_pi={beh['max_rel_policy']:.3e} "
            f"mu_tilde_q05={geo['mu_pl_tilde_q05']:.6g} p05/L={geo['pair_p05_over_L']:.6g}",
            flush=True,
        )
        if args.write_sheared_ckpts:
            c_dir = out_dir / f"c_{c:g}"
            c_dir.mkdir(parents=True, exist_ok=True)
            out_ckpt = {
                "policy": sheared.policy.state_dict(),
                "critic": sheared.critic.state_dict(),
                "v_ref": ckpt.get("v_ref"),
                "shear_c": c,
                "source_run_dir": str(run_dir),
            }
            torch.save(out_ckpt, c_dir / "weights_final.pt")
            print(f"  wrote {c_dir / 'weights_final.pt'}", flush=True)

    # E-A.1 gate
    worst_rel = max(
        max(r["max_rel_value"], r["max_rel_policy"]) for r in shear_rows
    )
    ea1_pass = worst_rel <= DELTA_BEHAVIOURAL

    # E-A.3 slope of log mu vs log c (exclude c=1 if only one point)
    mus = [r["mu_pl_tilde_q05"] for r in shear_rows]
    slope_fit = fit_log_slope(cs, mus) if len(cs) >= 2 else {}
    ea3_pass = bool(
        slope_fit
        and slope_fit["ci95_lo"] <= -2 <= slope_fit["ci95_hi"]
        and not (slope_fit["ci95_lo"] <= 0 <= slope_fit["ci95_hi"])
    )

    # E-A.4: p05/L at c=30 vs c=1
    by_c = {r["c"]: r for r in shear_rows}
    ea4 = {}
    if 1.0 in by_c and 30.0 in by_c:
        diff = by_c[30.0]["pair_p05_over_L"] - by_c[1.0]["pair_p05_over_L"]
        ea4 = {
            "p05_over_L_c1": by_c[1.0]["pair_p05_over_L"],
            "p05_over_L_c30": by_c[30.0]["pair_p05_over_L"],
            "diff_c30_minus_c1": diff,
            "h1_decreases": diff < 0,
        }

    # E-A.6 orthogonal
    orth = clone_stack(base)
    apply_orthogonal_gauge(orth, args.device, seed=0)
    beh_o = behavioural_max_rel_err(base, orth, obs)
    with torch.no_grad():
        z_o, _ = orth.critic.encode(obs)
    geo0 = geometry_stats(base.critic, z0, v_ref)
    geo_o = geometry_stats(orth.critic, z_o, v_ref)
    ea6 = {
        "behavioural": beh_o,
        "mu_pl_tilde_q05_base": geo0["mu_pl_tilde_q05"],
        "mu_pl_tilde_q05_orth": geo_o["mu_pl_tilde_q05"],
        "pair_p05_over_L_base": geo0["pair_p05_over_L"],
        "pair_p05_over_L_orth": geo_o["pair_p05_over_L"],
        "behavioural_ok": max(beh_o["max_rel_value"], beh_o["max_rel_policy"])
        <= DELTA_BEHAVIOURAL,
    }

    report = {
        "run_dir": str(run_dir),
        "delta_behavioural": DELTA_BEHAVIOURAL,
        "ea1_pass": ea1_pass,
        "worst_rel_err": worst_rel,
        "shear_rows": shear_rows,
        "ea3_log_slope": slope_fit,
        "ea3_pass_ci_contains_neg2_excludes_0": ea3_pass,
        "ea4": ea4,
        "ea6_orthogonal": ea6,
        "note_ea5": "Resume-training harm is a separate GPU job (ea5_resume_shear_s.sh)",
    }
    out_path = out_dir / "shear_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in report if k != "shear_rows"}, indent=2))
    print(f"Wrote {out_path}")
    if not ea1_pass:
        raise SystemExit(
            f"E-A.1 FAIL: max relative behavioural error {worst_rel:.3e} > {DELTA_BEHAVIOURAL}"
        )


if __name__ == "__main__":
    main()
