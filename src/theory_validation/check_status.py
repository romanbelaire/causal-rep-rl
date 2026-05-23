"""
Print theory_validation_v2 run completeness (CPU-only).

Naming on disk:
  outputs/theory_validation_v2/{experiment_name}_seed{N}_metrics.csv
  outputs/theory_validation_v2/{experiment_name}_seed{N}/weights_final.pt
  outputs/theory_validation_v2/{experiment_name}_seed{N}/weights_expert.pt  (exp0 only)
  artifacts/theory_validation/z_ref/{rstr|vae}_seed{N}.pt
"""

import argparse
from pathlib import Path

import pandas as pd

LOG = Path("outputs/theory_validation_v2")
ZREF = Path("artifacts/theory_validation/z_ref")
SEEDS = [42, 43, 44]
FULL_STEP = 1500 * 4096


def _status(log_dir: Path, name: str, seed: int, need_expert: bool = False) -> str:
    base = f"{name}_seed{seed}"
    csv = log_dir / f"{base}_metrics.csv"
    wdir = log_dir / base
    if not csv.exists():
        return "NOT_STARTED"
    df = pd.read_csv(csv)
    step = int(df["step"].max()) if "step" in df.columns and df["step"].notna().any() else 0
    has_final = (wdir / "weights_final.pt").exists()
    has_expert = (wdir / "weights_expert.pt").exists()
    if need_expert:
        if has_expert:
            return "OK (95% expert ckpt)"
        if step >= FULL_STEP and has_final:
            return "DONE (no 95% ckpt; used for zref if built from final)"
        if step >= FULL_STEP:
            return "METRICS_OK (no expert/final weights)"
        if len(df) <= 3:
            return "STARTED"
        return f"PARTIAL (step={step})"
    if step >= FULL_STEP and has_final:
        return "OK"
    if step >= FULL_STEP:
        return "METRICS_OK (no weights_final)"
    if len(df) <= 3:
        return "STARTED"
    return f"PARTIAL (step={step})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=LOG)
    args = parser.parse_args()
    log_dir = args.log_dir

    print(f"Full training: max_step >= {FULL_STEP}\n")

    blocks = [
        ("Exp0 expert RSTR", "expert_rstr_minigrid", SEEDS, True),
        ("Exp0 expert VAE", "expert_vae_minigrid", SEEDS, True),
        ("Exp1–2 RSTR (μ vs rank)", "theory_rstr_lconv_on_v2", SEEDS, False),
        ("Exp1–2 VAE (μ vs rank)", "theory_vae_ppo_no_repr_loss_v2", SEEDS, False),
        ("Exp3 repr ON", "exp3_rstr_repr_on", SEEDS, False),
        ("Exp3 repr OFF", "exp3_rstr_repr_off", SEEDS, False),
        ("Exp4 RSTR chain", "exp4_rstr_chain", [42], False),
        ("Exp4 VAE chain", "exp4_vae_chain", [42], False),
        ("Exp5 RSTR transfer", "exp5_rstr_transfer", SEEDS, False),
        ("Exp5 VAE transfer", "exp5_vae_transfer", SEEDS, False),
        ("Exp5 vanilla transfer", "exp5_vanilla_transfer", SEEDS, False),
    ]

    todo = []
    for title, name, seeds, expert in blocks:
        print(title)
        for s in seeds:
            st = _status(log_dir, name, s, expert)
            print(f"  seed {s}: {st}")
            if st not in ("OK", "METRICS_OK (no weights_final)"):
                todo.append(f"{name} seed {s} — {st}")
        print()

    print("Exp0 Z_ref")
    for kind in ("rstr", "vae"):
        for s in SEEDS:
            p = ZREF / f"{kind}_seed{s}.pt"
            print(f"  {kind} seed{s}: {'OK' if p.exists() else 'MISSING'}")
    print()

    if todo:
        print("=== Still need runs (for plots or checkpoints) ===")
        for line in todo:
            print(f"  {line}")
    else:
        print("All planned runs have at least full metrics.")


if __name__ == "__main__":
    main()
