"""
Z_ref lookup tables keyed by ground-truth Minigrid state (gt_repr).

Built from frozen near-optimal expert rollouts (Exp 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def gt_repr_key(gt: torch.Tensor, decimals: int = 4) -> tuple:
    """Hashable key from ground-truth representation vector."""
    return tuple(round(float(x), decimals) for x in gt.tolist())


class ZRefStore:
    """gt_repr key -> expert latent z_ref (mean if duplicate keys)."""

    def __init__(self, table: dict[tuple, torch.Tensor]):
        self.table = table

    @classmethod
    def load(cls, path: str | Path) -> "ZRefStore":
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw = payload["table"] if isinstance(payload, dict) and "table" in payload else payload
        table = {}
        for k, v in raw.items():
            key = k if isinstance(k, tuple) else tuple(k)
            table[key] = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        return cls(table)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v.cpu() for k, v in self.table.items()}
        torch.save({"table": serializable, "metadata": metadata or {}}, path)
        if metadata is not None:
            meta_path = path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

    def lookup_batch(self, gt_batch: torch.Tensor, decimals: int = 4) -> torch.Tensor:
        """Return z_ref [N, d] for each row in gt_batch."""
        refs = []
        for i in range(gt_batch.shape[0]):
            key = gt_repr_key(gt_batch[i], decimals=decimals)
            if key not in self.table:
                raise KeyError(
                    f"No Z_ref for gt_repr key {key} (sample {i}). "
                    "Roll more expert episodes or check gt_repr alignment."
                )
            refs.append(self.table[key])
        return torch.stack(refs, dim=0)

    def __len__(self) -> int:
        return len(self.table)


def build_table_from_rollout(
    gt_list: list[torch.Tensor],
    z_list: list[torch.Tensor],
    decimals: int = 4,
) -> dict[tuple, torch.Tensor]:
    """Build table from parallel lists of gt_repr and expert latents."""
    table: dict[tuple, torch.Tensor] = {}
    counts: dict[tuple, int] = {}
    for gt, z in zip(gt_list, z_list):
        key = gt_repr_key(gt, decimals=decimals)
        z = z.detach().cpu().float()
        if key not in table:
            table[key] = z.clone()
            counts[key] = 1
        else:
            n = counts[key]
            table[key] = (table[key] * n + z) / (n + 1)
            counts[key] = n + 1
    return table
