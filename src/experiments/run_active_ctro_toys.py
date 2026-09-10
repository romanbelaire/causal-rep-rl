"""CPU toy pipeline for active CTRO (E1–E5 + invariants). No GPU. Deterministic exit."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from src.experiments.config import ACTIVE_CTRO_RESULTS_TOYS
from src.experiments.exp_e1_action_alias import main as e1_main
from src.experiments.exp_e2_ensemble_calibration import main as e2_main
from src.experiments.exp_e3_query_coverage import main as e3_main
from src.experiments.exp_e4_nuisance import main as e4_main
from src.experiments.exp_e5_kl_null import main as e5_main
from src.experiments.exp_t1_pl_scale_collapse import main as t1_main
from src.experiments.exp_t2_equal_v_diff_q import main as t2_main
from src.experiments.exp_t3_nuisance_quotient import main as t3_main
from src.experiments.exp_t4_empirical_pseudometric import main as t4_main
from src.experiments.exp_t5_relational_preservation import main as t5_main
from src.experiments.validate_invariants import main as invariants_main

TOYS = (
    ("invariants", invariants_main),
    ("e1_action_alias", e1_main),
    ("e2_ensemble_calibration", e2_main),
    ("e3_query_coverage", e3_main),
    ("e4_nuisance", e4_main),
    ("e5_kl_null", e5_main),
    ("t1_pl_scale_collapse", t1_main),
    ("t2_equal_v_diff_q", t2_main),
    ("t3_nuisance_quotient", t3_main),
    ("t4_empirical_pseudometric", t4_main),
    ("t5_relational_preservation", t5_main),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active-CTRO CPU toys and write a summary")
    parser.add_argument("--results-root", type=str, default=ACTIVE_CTRO_RESULTS_TOYS)
    args = parser.parse_args()
    out_dir = Path(args.results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failed: str | None = None
    for name, fn in TOYS:
        print(f"=== {name} ===", flush=True)
        try:
            fn()
            results.append({"name": name, "status": "ok"})
        except Exception as exc:
            traceback.print_exc()
            results.append({"name": name, "status": "fail", "error": str(exc)})
            failed = name
            break

    summary = {"cells": results, "ok": failed is None}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {path}", flush=True)
    if failed is not None:
        raise RuntimeError(f"active-CTRO toy pipeline failed at {failed}")


if __name__ == "__main__":
    main()
