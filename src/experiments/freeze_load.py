"""Load Phase-0 freeze files as real algorithm overrides (not stamp-only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Keys that define LTRO-fixed under ltro_phase0_v1. Architecture/training may differ
# on ALE; these algorithm fields must match the freeze after load.
LTRO_FREEZE_ALGO_KEYS: tuple[str, ...] = (
    "dz_enabled",
    "lambda_dz",
    "eta_dz",
    "dz_adapt",
    "dz_eta_adapt",
    "eta_dz_min",
    "eta_dz_c",
    "dz_ema_tau",
    "dz_ema_init",
    "lambda_loc",
    "dz_delta",
    "dz_collapse_target_thresh",
    "dz_n_pairs",
    "ref_buffer_size",
    "ref_refresh_factor",
)


def freeze_path(freeze_id: str) -> Path:
    return Path("configs") / "frozen" / f"{freeze_id}.json"


def load_freeze_file(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"freeze file not found: {p}")
    return json.loads(p.read_text())


def protocol_hash(protocol_path: str | Path = "docs/moalla_ale_protocol.md") -> str:
    data = Path(protocol_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def apply_ltro_freeze_overrides(algo_cfg: dict, freeze: dict) -> dict:
    """Merge LTRO geometry keys from freeze into algo_cfg; verify exact match."""
    frozen_algo = freeze["algorithm"]
    out = {**algo_cfg}
    for key in LTRO_FREEZE_ALGO_KEYS:
        if key not in frozen_algo:
            raise KeyError(f"freeze algorithm missing key {key}")
        out[key] = frozen_algo[key]
    # Fail if LTRO keys differ from freeze (s_ref is runtime-measured, not in freeze).
    for key in LTRO_FREEZE_ALGO_KEYS:
        if out[key] != frozen_algo[key]:
            raise RuntimeError(
                f"algo[{key}]={out[key]!r} differs from freeze {frozen_algo[key]!r}"
            )
    out["dz_enabled"] = True
    if out["dz_adapt"] is not False:
        raise RuntimeError("ltro_phase0_v1 requires dz_adapt=false (LTRO-fixed)")
    if float(out["lambda_dz"]) != 1.0:
        raise RuntimeError("ltro_phase0_v1 requires lambda_dz=1.0")
    return out
