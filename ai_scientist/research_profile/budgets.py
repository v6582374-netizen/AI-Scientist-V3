"""Static budget profiles and config mutation helpers."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BudgetProfile:
    id: str
    num_workers: int | None
    num_seeds: int | None
    timeout_per_trial_minutes: int | None
    max_stage_iterations: int | None
    max_epochs_hint: int | None
    max_dataset_samples_hint: int | None


BUDGET_PROFILES = {
    "tiny": BudgetProfile(
        id="tiny",
        num_workers=1,
        num_seeds=1,
        timeout_per_trial_minutes=10,
        max_stage_iterations=2,
        max_epochs_hint=1,
        max_dataset_samples_hint=1000,
    ),
    "small": BudgetProfile(
        id="small",
        num_workers=1,
        num_seeds=1,
        timeout_per_trial_minutes=30,
        max_stage_iterations=4,
        max_epochs_hint=3,
        max_dataset_samples_hint=5000,
    ),
    "medium": BudgetProfile(
        id="medium",
        num_workers=2,
        num_seeds=2,
        timeout_per_trial_minutes=60,
        max_stage_iterations=8,
        max_epochs_hint=5,
        max_dataset_samples_hint=20000,
    ),
    "full": BudgetProfile(
        id="full",
        num_workers=None,
        num_seeds=3,
        timeout_per_trial_minutes=None,
        max_stage_iterations=None,
        max_epochs_hint=None,
        max_dataset_samples_hint=None,
    ),
}


def get_budget_profile(profile_id: str) -> BudgetProfile:
    try:
        return BUDGET_PROFILES[profile_id]
    except KeyError as exc:
        valid = ", ".join(sorted(BUDGET_PROFILES))
        raise ValueError(
            f"Unknown budget profile '{profile_id}'. Expected one of: {valid}"
        ) from exc


def valid_budget_profile_ids() -> tuple[str, ...]:
    return tuple(sorted(BUDGET_PROFILES))


def _ensure_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.setdefault(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected config['{key}'] to be a mapping.")
    return value


def _cap_int(existing: Any, cap: int) -> int:
    try:
        existing_int = int(existing)
    except (TypeError, ValueError):
        return cap
    return min(existing_int, cap)


def apply_budget_profile_to_config(
    config: dict[str, Any], budget_profile_id: str
) -> dict[str, Any]:
    """Apply static hard limits to a loaded BFTS config dictionary."""
    profile = get_budget_profile(budget_profile_id)
    agent = _ensure_mapping(config, "agent")
    exec_cfg = _ensure_mapping(config, "exec")

    if profile.num_workers is not None:
        agent["num_workers"] = profile.num_workers

    if profile.num_seeds is not None:
        multi_seed_eval = agent.setdefault("multi_seed_eval", {})
        if not isinstance(multi_seed_eval, dict):
            raise ValueError("Expected agent.multi_seed_eval to be a mapping.")
        multi_seed_eval["num_seeds"] = profile.num_seeds

    if profile.timeout_per_trial_minutes is not None:
        timeout_seconds = profile.timeout_per_trial_minutes * 60
        exec_cfg["timeout"] = _cap_int(exec_cfg.get("timeout"), timeout_seconds)

    if profile.max_stage_iterations is not None:
        stages = agent.setdefault("stages", {})
        if not isinstance(stages, dict):
            raise ValueError("Expected agent.stages to be a mapping.")
        for stage_key in list(stages.keys()):
            if stage_key.endswith("_max_iters"):
                stages[stage_key] = _cap_int(
                    stages[stage_key], profile.max_stage_iterations
                )
        agent["steps"] = _cap_int(agent.get("steps"), profile.max_stage_iterations)

    return config


def budget_prompt_guidance(budget_profile_id: str, *, domain_id: str) -> list[str]:
    profile = get_budget_profile(budget_profile_id)
    lines = [
        f"Respect the selected budget profile: {profile.id}.",
    ]
    if profile.timeout_per_trial_minutes is not None:
        lines.append(
            f"Design each generated program to finish within about {profile.timeout_per_trial_minutes} minutes."
        )
    if profile.max_dataset_samples_hint is not None:
        lines.append(
            f"Keep data volume modest; use at most about {profile.max_dataset_samples_hint} rows/items unless the task explicitly requires less."
        )
    if domain_id == "machine_learning" and profile.max_epochs_hint is not None:
        lines.append(
            f"Keep training short; use at most about {profile.max_epochs_hint} epochs unless the idea requires fewer."
        )
    return lines
