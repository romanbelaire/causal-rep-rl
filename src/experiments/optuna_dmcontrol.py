"""Optuna hyperparameter search for DMControl performance suite.

One study per (agent, task). Search budget is truncated (1M steps); confirm
winners at full 8M via confirm_optuna_winners / train jobs.

Example:
  python -m src.experiments.optuna_dmcontrol \\
    --agent exp_ctro_mlp --task cartpole-swingup --n-trials 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.experiments.config import DMCONTROL_OPTUNA_SEARCH_STEPS
from src.experiments.performance_runner import RunAborted, run_performance_train

AGENT_SPECS = {
    "exp_baseline": {"agent": "ppo", "agent_cls": PPO, "exp_name": "exp_baseline"},
    "exp_ctro_mlp": {"agent": "ctro", "agent_cls": CTRO, "exp_name": "exp_ctro_mlp"},
}

DMCONTROL_TASKS = (
    "cartpole-swingup",
    "cheetah-run",
    "hopper-hop",
    "walker-walk",
)

HIDDEN_CHOICES = {
    "64_64": [64, 64],
    "256_256": [256, 256],
}


def study_name(agent_key: str, task: str) -> str:
    return f"dmcontrol_{agent_key}_{task}"


def study_dir(root: Path, agent_key: str, task: str) -> Path:
    return root / "dmcontrol_state" / agent_key / task


def storage_url(root: Path, agent_key: str, task: str) -> str:
    d = study_dir(root, agent_key, task)
    d.mkdir(parents=True, exist_ok=True)
    db = d / "study.db"
    return f"sqlite:///{db.resolve()}"


def suggest_params(trial: optuna.Trial, agent_key: str) -> dict:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "entropy_coef": trial.suggest_float("entropy_coef", 0.0, 0.05),
        "num_epochs": trial.suggest_categorical("num_epochs", [5, 10, 20]),
        "policy_hidden_key": trial.suggest_categorical(
            "policy_hidden_key", list(HIDDEN_CHOICES.keys())
        ),
    }
    if agent_key == "exp_ctro_mlp":
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 0.5, log=True)
        params["beta"] = trial.suggest_float("beta", 1e-2, 1.0, log=True)
    return params


def make_objective(
    agent_key: str,
    task: str,
    seed: int,
    results_root: Path,
    optuna_root: Path,
    device: str | None,
    search_steps: int,
):
    spec = AGENT_SPECS[agent_key]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, agent_key)
        hidden = HIDDEN_CHOICES[params["policy_hidden_key"]]
        algo_overrides = {
            "learning_rate": params["learning_rate"],
            "entropy_coef": params["entropy_coef"],
            "num_epochs": params["num_epochs"],
        }
        if agent_key == "exp_ctro_mlp":
            algo_overrides["alpha"] = params["alpha"]
            algo_overrides["beta"] = params["beta"]
        else:
            algo_overrides["alpha"] = 0.0
            algo_overrides["beta"] = 0.0
            algo_overrides["vae_coef"] = 0.0

        exp_name = f"optuna/{agent_key}/{task}/trial_{trial.number}"
        trial_meta = {
            "trial_number": trial.number,
            "agent": agent_key,
            "task": task,
            "params": {**params, "policy_hidden": hidden},
        }
        trial_dir = study_dir(optuna_root, agent_key, task)
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / f"trial_{trial.number}_params.json").write_text(
            json.dumps(trial_meta, indent=2) + "\n"
        )

        def report_callback(total_steps: int, metrics: dict) -> bool:
            value = metrics.get("eval_full_return_mean")
            if value is None:
                return False
            # Use epoch as the Optuna step so MedianPruner compares aligned checkpoints.
            step = int(metrics["epoch"])
            trial.report(float(value), step=step)
            return bool(trial.should_prune())

        try:
            result = run_performance_train(
                suite_name="dmcontrol_state",
                task=task,
                seed=seed,
                exp_name=exp_name,
                agent_cls=spec["agent_cls"],
                algo_overrides=algo_overrides,
                arch_overrides={"policy_hidden": hidden},
                train_overrides={"total_steps": search_steps},
                results_root=results_root,
                device=device,
                report_callback=report_callback,
            )
        except RunAborted as exc:
            print(
                f"Trial {trial.number} aborted: {exc.status} — {exc.reason}",
                flush=True,
            )
            raise optuna.TrialPruned() from exc

        if result.eval_full_return_mean is None:
            raise RuntimeError(
                f"Trial {trial.number} finished without eval_full_return_mean"
            )
        return float(result.eval_full_return_mean)

    return objective


def run_study(
    agent_key: str,
    task: str,
    n_trials: int,
    seed: int,
    results_root: Path,
    optuna_root: Path,
    device: str | None,
    search_steps: int,
) -> optuna.Study:
    if agent_key not in AGENT_SPECS:
        raise ValueError(f"Unknown agent {agent_key}; allowed={sorted(AGENT_SPECS)}")
    if task not in DMCONTROL_TASKS:
        raise ValueError(f"Unknown task {task}; allowed={list(DMCONTROL_TASKS)}")

    storage = storage_url(optuna_root, agent_key, task)
    study = optuna.create_study(
        study_name=study_name(agent_key, task),
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    objective = make_objective(
        agent_key=agent_key,
        task=task,
        seed=seed,
        results_root=results_root,
        optuna_root=optuna_root,
        device=device,
        search_steps=search_steps,
    )
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
    best_path = study_dir(optuna_root, agent_key, task) / "best_trial.json"
    if study.best_trial is not None:
        best = {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        }
        best_path.write_text(json.dumps(best, indent=2) + "\n")
        print(f"Best trial {best['number']} value={best['value']} -> {best_path}", flush=True)
    return study


def main():
    parser = argparse.ArgumentParser(description="Optuna DMControl HP search")
    parser.add_argument("--agent", required=True, choices=list(AGENT_SPECS.keys()))
    parser.add_argument("--task", required=True, choices=list(DMCONTROL_TASKS))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--optuna-root", type=str, default="results/optuna")
    parser.add_argument(
        "--search-steps",
        type=int,
        default=DMCONTROL_OPTUNA_SEARCH_STEPS,
        help="Truncated env-step budget per trial",
    )
    args = parser.parse_args()

    run_study(
        agent_key=args.agent,
        task=args.task,
        n_trials=args.n_trials,
        seed=args.seed,
        results_root=Path(args.results_root),
        optuna_root=Path(args.optuna_root),
        device=args.device,
        search_steps=args.search_steps,
    )


if __name__ == "__main__":
    main()
