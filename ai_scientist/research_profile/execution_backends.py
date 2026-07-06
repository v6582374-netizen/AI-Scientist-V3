"""Local execution backend metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionBackend:
    id: str
    display_name: str
    runs_code: bool
    guidance: str


EXECUTION_BACKENDS = {
    "dry_run": ExecutionBackend(
        id="dry_run",
        display_name="Dry Run",
        runs_code=False,
        guidance=(
            "Do not claim measured empirical results. Produce plans, expected "
            "validation steps, and feasibility reasoning only."
        ),
    ),
    "smoke": ExecutionBackend(
        id="smoke",
        display_name="Smoke",
        runs_code=True,
        guidance=(
            "Run only minimal code paths that check feasibility and basic sanity. "
            "Do not draw performance or robustness conclusions."
        ),
    ),
    "local_cpu_limited": ExecutionBackend(
        id="local_cpu_limited",
        display_name="Local CPU Limited",
        runs_code=True,
        guidance=(
            "Use CPU-friendly code and keep the workload bounded by the selected "
            "budget. Do not require accelerator hardware."
        ),
    ),
    "local_gpu_cuda_limited": ExecutionBackend(
        id="local_gpu_cuda_limited",
        display_name="Local CUDA GPU Limited",
        runs_code=True,
        guidance=(
            "CUDA may be used, but the workload must still respect the selected "
            "budget profile."
        ),
    ),
}


def valid_execution_backend_ids() -> tuple[str, ...]:
    return tuple(sorted(EXECUTION_BACKENDS))


def get_execution_backend(backend_id: str) -> ExecutionBackend:
    try:
        return EXECUTION_BACKENDS[backend_id]
    except KeyError as exc:
        valid = ", ".join(sorted(EXECUTION_BACKENDS))
        raise ValueError(
            f"Unknown execution backend '{backend_id}'. Expected one of: {valid}"
        ) from exc


def detect_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def choose_execution_backend(
    domain_id: str,
    requested_backend: str = "auto",
    *,
    cuda_available: bool | None = None,
) -> str:
    if requested_backend != "auto":
        backend = get_execution_backend(requested_backend)
        if backend.id == "local_gpu_cuda_limited" and cuda_available is False:
            raise ValueError("local_gpu_cuda_limited was requested, but CUDA is not available.")
        return backend.id

    if cuda_available is None:
        cuda_available = detect_cuda_available()

    if domain_id == "machine_learning" and cuda_available:
        return "local_gpu_cuda_limited"
    return "local_cpu_limited"


def evidence_level_for_backend(backend_id: str, budget_profile_id: str) -> str:
    if backend_id == "dry_run":
        return "dry_run"
    if backend_id == "smoke":
        return "smoke"
    if budget_profile_id == "full":
        return "full_empirical"
    return "limited_empirical"


def claim_policy_for_evidence(evidence_level: str) -> dict[str, list[str]]:
    policies = {
        "dry_run": {
            "allowed": [
                "proposed method",
                "expected validation plan",
                "feasibility reasoning",
            ],
            "forbidden": [
                "empirical improvement",
                "measured performance",
                "validated conclusion",
            ],
        },
        "smoke": {
            "allowed": [
                "implementation feasibility",
                "basic sanity check",
                "smoke-test observation",
            ],
            "forbidden": [
                "performance conclusion",
                "robustness claim",
                "full validation",
            ],
        },
        "limited_empirical": {
            "allowed": [
                "preliminary empirical evidence",
                "feasibility observations",
                "resource-limited analysis",
            ],
            "forbidden": [
                "full empirical validation",
                "state-of-the-art performance claim",
                "strong causal conclusion",
            ],
        },
        "full_empirical": {
            "allowed": [
                "complete empirical conclusion within the measured experiment scope",
                "claims directly supported by measured evidence",
            ],
            "forbidden": [
                "claims beyond measured evidence",
                "unmeasured generalization claim",
            ],
        },
    }
    try:
        return policies[evidence_level]
    except KeyError as exc:
        valid = ", ".join(sorted(policies))
        raise ValueError(
            f"Unknown evidence level '{evidence_level}'. Expected one of: {valid}"
        ) from exc


def valid_evidence_levels() -> tuple[str, ...]:
    return ("dry_run", "full_empirical", "limited_empirical", "smoke")
