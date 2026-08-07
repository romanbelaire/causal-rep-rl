"""
Performance evaluation suite definitions.

Each suite specifies environments and two evaluation distributions:
  - full:  in-distribution / training-set performance
  - test:  held-out / generalization performance

Procgen (easy, 25M steps):
  Training uses 200 levels (start_level=0). Full eval replays those levels;
  test eval uses held-out levels (start_level=200) or the unlimited distribution
  per train-procgen convention (num_levels=0).

DMControl (state observations):
  Full eval uses random seeds [0, n); test eval uses seeds [seed_offset, seed_offset + n).

DMControl (pixel observations):
  Same tasks and seed offsets as state; obs are 84x84 RGB renders.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionSpec:
    """Parameters that define an evaluation distribution within a suite."""

    name: str
    procgen_num_levels: int | None = None
    procgen_start_level: int | None = None
    dmcontrol_seed_offset: int | None = None


@dataclass(frozen=True)
class EvalSuite:
    name: str
    env_type: str
    tasks: tuple[str, ...]
    distributions: tuple[DistributionSpec, ...]
    distribution_mode: str | None = None
    training_steps: int | None = None
    train_num_levels: int | None = None
    eval_episodes: int = 100
    eval_deterministic: bool = True


PROCGEN_EASY_GAMES: tuple[str, ...] = (
    "coinrun",
    "starpilot",
    "caveflyer",
    "fruitbot",
    "chaser",
    "leaper",
    "maze",
    "miner",
)

PROCGEN_EASY_SUITE = EvalSuite(
    name="procgen_easy",
    env_type="procgen",
    tasks=PROCGEN_EASY_GAMES,
    distribution_mode="easy",
    training_steps=25_000_000,
    train_num_levels=200,
    distributions=(
        DistributionSpec(
            name="full",
            procgen_num_levels=200,
            procgen_start_level=0,
        ),
        DistributionSpec(
            name="test",
            procgen_num_levels=0,
            procgen_start_level=0,
        ),
    ),
)

DMCONTROL_STATE_TASKS: tuple[str, ...] = (
    "cheetah-run",
    "walker-walk",
    "hopper-hop",
    "cartpole-swingup",
)

DMCONTROL_STATE_SUITE = EvalSuite(
    name="dmcontrol_state",
    env_type="dmcontrol",
    tasks=DMCONTROL_STATE_TASKS,
    distributions=(
        DistributionSpec(
            name="full",
            dmcontrol_seed_offset=0,
        ),
        DistributionSpec(
            name="test",
            dmcontrol_seed_offset=10_000,
        ),
    ),
)

DMCONTROL_PIXELS_SUITE = EvalSuite(
    name="dmcontrol_pixels",
    env_type="dmcontrol_pixels",
    tasks=DMCONTROL_STATE_TASKS,
    distributions=(
        DistributionSpec(
            name="full",
            dmcontrol_seed_offset=0,
        ),
        DistributionSpec(
            name="test",
            dmcontrol_seed_offset=10_000,
        ),
    ),
)

EVAL_SUITES: dict[str, EvalSuite] = {
    PROCGEN_EASY_SUITE.name: PROCGEN_EASY_SUITE,
    DMCONTROL_STATE_SUITE.name: DMCONTROL_STATE_SUITE,
    DMCONTROL_PIXELS_SUITE.name: DMCONTROL_PIXELS_SUITE,
}


def parse_dmcontrol_task(task_id: str) -> tuple[str, str]:
    """Split 'cheetah-run' into ('cheetah', 'run')."""
    domain, task = task_id.split("-", 1)
    return domain, task
