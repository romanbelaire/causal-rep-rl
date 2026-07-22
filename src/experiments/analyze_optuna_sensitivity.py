"""Sensitivity analysis + confirm-command export for DMControl Optuna studies.

Example:
  python -m src.experiments.analyze_optuna_sensitivity \\
    --optuna-root results/optuna --agent exp_ctro_mlp --task cartpole-swingup

  python -m src.experiments.analyze_optuna_sensitivity --optuna-root results/optuna --all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import optuna

from src.experiments.optuna_dmcontrol import (
    AGENT_SPECS,
    DMCONTROL_TASKS,
    HIDDEN_CHOICES,
    storage_url,
    study_dir,
    study_name,
)


def _load_study(optuna_root: Path, agent_key: str, task: str) -> optuna.Study:
    storage = storage_url(optuna_root, agent_key, task)
    return optuna.load_study(study_name=study_name(agent_key, task), storage=storage)


def analyze_one(optuna_root: Path, agent_key: str, task: str, top_k: int) -> Path:
    study = _load_study(optuna_root, agent_key, task)
    out_dir = study_dir(optuna_root, agent_key, task)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = study.get_trials(deepcopy=False)
    n_complete = sum(1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED)
    n_fail = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)
    n_total = len(trials)

    rows = []
    for t in trials:
        row = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value if t.value is not None else "",
            **{k: t.params.get(k, "") for k in sorted({p for tr in trials for p in tr.params})},
        }
        rows.append(row)

    csv_path = out_dir / "trials.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    importance = {}
    if n_complete >= 2:
        importance = optuna.importance.get_param_importances(study)

    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    top = completed[:top_k]

    winners = []
    for t in top:
        params = dict(t.params)
        hidden_key = params.get("policy_hidden_key", "64_64")
        winners.append(
            {
                "number": t.number,
                "value": t.value,
                "params": params,
                "policy_hidden": HIDDEN_CHOICES[hidden_key],
                "confirm_extra_args": _confirm_extra_args(agent_key, params),
            }
        )

    report = {
        "agent": agent_key,
        "task": task,
        "n_total": n_total,
        "n_complete": n_complete,
        "n_pruned": n_pruned,
        "n_fail": n_fail,
        "prune_fail_rate": (n_pruned + n_fail) / n_total if n_total else 0.0,
        "param_importances": importance,
        "best_trial": winners[0] if winners else None,
        "top_k": winners,
    }
    report_path = out_dir / "sensitivity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    txt_path = out_dir / "sensitivity.txt"
    lines = [
        f"Study: {study_name(agent_key, task)}",
        f"trials={n_total} complete={n_complete} pruned={n_pruned} fail={n_fail} "
        f"prune_fail_rate={report['prune_fail_rate']:.3f}",
        "",
        "Param importances:",
    ]
    if importance:
        for k, v in sorted(importance.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k}: {v:.4f}")
    else:
        lines.append("  (need >=2 complete trials)")
    lines.append("")
    lines.append(f"Top-{top_k} trials:")
    for w in winners:
        lines.append(f"  #{w['number']} value={w['value']:.4f} params={w['params']}")
        lines.append(f"    EXTRA_ARGS: {' '.join(w['confirm_extra_args'])}")
    txt_path.write_text("\n".join(lines) + "\n")

    confirm_path = out_dir / "confirm_commands.sh"
    confirm_lines = [
        "#!/bin/bash",
        f"# Full-budget confirm for {agent_key}/{task} top-{top_k}",
        "set -euo pipefail",
        "cd /common/home/users/r/rbelaire.2021/causal-rep",
        "source rl-venv/bin/activate",
        "",
    ]
    for i, w in enumerate(winners):
        exp = f"exp_optuna_confirm_{agent_key}_t{w['number']}"
        args = " ".join(w["confirm_extra_args"])
        confirm_lines.append(f"# rank {i} trial {w['number']} value={w['value']}")
        confirm_lines.append(
            f"EXP_NAME={exp} EXTRA_ARGS_STR={args!r} "
            f"TASK={task} sbatch src/experiments/jobs/optuna_confirm_dmcontrol_s.sh"
        )
        confirm_lines.append("")
    confirm_path.write_text("\n".join(confirm_lines) + "\n")
    confirm_path.chmod(0o755)

    print(txt_path.read_text(), flush=True)
    print(f"Wrote {report_path}, {csv_path}, {confirm_path}", flush=True)
    return report_path


def _confirm_extra_args(agent_key: str, params: dict) -> list[str]:
    spec = AGENT_SPECS[agent_key]
    hidden = HIDDEN_CHOICES[params["policy_hidden_key"]]
    hidden_str = ",".join(str(h) for h in hidden)
    args = [
        f"--agent {spec['agent']}",
        f"--learning-rate {params['learning_rate']}",
        f"--entropy-coef {params['entropy_coef']}",
        f"--num-epochs {params['num_epochs']}",
        f"--policy-hidden {hidden_str}",
    ]
    if agent_key == "exp_ctro_mlp":
        args.append(f"--alpha {params['alpha']}")
        args.append(f"--beta {params['beta']}")
    return args


def main():
    parser = argparse.ArgumentParser(description="Optuna sensitivity for DMControl")
    parser.add_argument("--optuna-root", type=str, default="results/optuna")
    parser.add_argument("--agent", choices=list(AGENT_SPECS.keys()), default=None)
    parser.add_argument("--task", choices=list(DMCONTROL_TASKS), default=None)
    parser.add_argument("--all", action="store_true", help="Analyze all agent/task studies")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.optuna_root)
    if args.all:
        pairs = [(a, t) for a in AGENT_SPECS for t in DMCONTROL_TASKS]
    else:
        if args.agent is None or args.task is None:
            raise SystemExit("Provide --agent and --task, or --all")
        pairs = [(args.agent, args.task)]

    for agent_key, task in pairs:
        db = study_dir(root, agent_key, task) / "study.db"
        if not db.exists():
            print(f"Skipping {agent_key}/{task} — no study.db at {db}", flush=True)
            continue
        analyze_one(root, agent_key, task, top_k=args.top_k)


if __name__ == "__main__":
    main()
