"""Seed-matched expert Z_ref paths for theory validation v2."""

from __future__ import annotations

from pathlib import Path

TV2_LOG = Path("outputs/theory_validation_v2")
TV4_LOG = Path("outputs/theory_validation_v4")

EXPERT_BASE_CONFIG = {
    "rstr": Path("configs/theory_validation/expert_rstr_minigrid.json"),
    "vae": Path("configs/theory_validation/expert_vae_minigrid.json"),
    "mico": Path("configs/theory_validation_v4/expert_mico_minigrid.json"),
    "dbc": Path("configs/theory_validation_v4/expert_dbc_minigrid.json"),
}

EXPERT_GENERATED_CONFIG = {
    "rstr": TV2_LOG / "generated_configs/expert_rstr_minigrid_stochastic_3000.json",
    "vae": TV2_LOG / "generated_configs/expert_vae_minigrid_stochastic_3000.json",
    "mico": TV4_LOG / "generated_configs/expert_mico_minigrid_stochastic_3000.json",
    "dbc": TV4_LOG / "generated_configs/expert_dbc_minigrid_stochastic_3000.json",
}


def _expert_run_dir(weights_or_dir: Path) -> Path:
    """Directory containing weights_expert.pt / weights_final.pt."""
    p = Path(weights_or_dir)
    return p if p.is_dir() else p.parent


def resolve_expert_weights_file(
    weights_or_dir: str | Path,
    use_weights_final: bool = False,
) -> tuple[Path, str]:
    """
    Pick checkpoint file under an expert run directory.

    Args:
        weights_or_dir: Path to weights_*.pt or the expert run directory.
        use_weights_final: If True, use weights_final.pt (e.g. 92% run without 95% gate save).

    Returns:
        (weights_path, checkpoint_kind) with checkpoint_kind in weights_expert | weights_final.
    """
    expert_dir = _expert_run_dir(Path(weights_or_dir))
    expert_pt = expert_dir / "weights_expert.pt"
    final_pt = expert_dir / "weights_final.pt"

    if use_weights_final:
        if not final_pt.exists():
            raise FileNotFoundError(
                f"weights_final.pt not found under {expert_dir} (--use-weights-final)."
            )
        if expert_pt.exists():
            print(
                f"WARNING [build_z_ref]: {expert_dir.name} — "
                "--use-weights-final set; using weights_final.pt "
                "(weights_expert.pt also exists but is ignored).",
                flush=True,
            )
        else:
            print(
                f"WARNING [build_z_ref]: {expert_dir.name} — "
                "using weights_final.pt (weights_expert.pt missing).",
                flush=True,
            )
        return final_pt, "weights_final"

    if expert_pt.exists():
        return expert_pt, "weights_expert"

    if final_pt.exists():
        print(
            f"WARNING [build_z_ref]: {expert_dir.name} — "
            "weights_expert.pt missing; using weights_final.pt. "
            "Pass --use-weights-final to silence this auto-fallback.",
            flush=True,
        )
        return final_pt, "weights_final"

    raise FileNotFoundError(
        f"No weights_expert.pt or weights_final.pt under {expert_dir}."
    )


def resolve_z_ref_expert(family: str, seed: int) -> tuple[str, str, str]:
    """
    Return (expert_config_path, weights_path, checkpoint_kind) for live Z*(s) encoding.

    Prefers weights_expert.pt (95% success gate). Falls back to weights_final.pt with a
    printed warning when the expert checkpoint was never saved.

    checkpoint_kind is "weights_expert" or "weights_final".
    """
    if family not in EXPERT_BASE_CONFIG:
        raise ValueError(
            f"Unknown z_ref expert family {family!r}; expected rstr, vae, mico, or dbc"
        )

    generated = EXPERT_GENERATED_CONFIG[family]
    base = EXPERT_BASE_CONFIG[family]
    config_path = generated if generated.exists() else base
    if not config_path.exists():
        raise FileNotFoundError(f"Expert config not found: {config_path}")

    if family in ("mico", "dbc"):
        expert_dir = TV4_LOG / f"expert_{family}_minigrid_seed{seed}"
    else:
        expert_dir = TV2_LOG / f"expert_{family}_minigrid_seed{seed}"
    weights_path, ckpt_kind = resolve_expert_weights_file(expert_dir, use_weights_final=False)
    if ckpt_kind == "weights_final":
        print(
            f"WARNING [z_ref expert]: expert_{family}_minigrid_seed{seed} — "
            "weights_expert.pt missing (eval did not reach min_success_rate); "
            "using weights_final.pt for Z* encoding.",
            flush=True,
        )
    return str(config_path), str(weights_path), ckpt_kind
