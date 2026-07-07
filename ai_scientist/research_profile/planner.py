"""Research Profile Planner.

The first version is deterministic and auditable. It can be replaced by an LLM
selector later without changing the downstream Research Profile contract.
"""

from __future__ import annotations

from .domains import get_domain_pack, valid_domain_ids
from .execution_backends import (
    choose_execution_backend,
    claim_policy_for_evidence,
    detect_cuda_available,
    evidence_level_for_backend,
)
from .schema import AUTO, validate_research_profile


ML_KEYWORDS = (
    "ablation",
    "accuracy",
    "benchmark",
    "classification",
    "cuda",
    "dataloader",
    "deep learning",
    "fine-tune",
    "gpu",
    "hyperparameter",
    "loss",
    "machine learning",
    "neural network",
    "pytorch",
    "reinforcement learning",
    "supervised learning",
    "torch",
    "train a transformer",
    "transformer model",
)


def _choose_domain(topic_text: str) -> tuple[str, float, str]:
    normalized = topic_text.lower()
    matches = [keyword for keyword in ML_KEYWORDS if keyword in normalized]
    if matches:
        return (
            "machine_learning",
            min(0.95, 0.72 + 0.04 * len(matches)),
            "The research input contains ML-specific terms: " + ", ".join(matches[:6]),
        )
    return (
        "general",
        0.78,
        "The research input does not require ML model training or ML-specific evaluation.",
    )


def _choose_budget(
    requested_budget: str,
    *,
    domain_id: str,
    backend_id: str,
) -> tuple[str, str]:
    if requested_budget != AUTO:
        return requested_budget, "Budget profile was explicitly requested."

    if backend_id == "dry_run":
        return "tiny", "Dry-run execution only needs a tiny budget."
    if backend_id == "smoke":
        return "tiny", "Smoke execution should stay minimal."
    if domain_id == "machine_learning" and backend_id == "local_gpu_cuda_limited":
        return "medium", "CUDA is available for an ML task, so a medium local budget is allowed."
    return "small", "Defaulting to a small local budget suitable for ordinary computers."


def plan_research_profile(
    topic_text: str,
    *,
    domain: str = AUTO,
    execution_backend: str = AUTO,
    budget_profile: str = AUTO,
    cuda_available: bool | None = None,
) -> dict:
    if not isinstance(topic_text, str) or not topic_text.strip():
        raise ValueError("Research input must be a non-empty string.")

    if domain == AUTO:
        domain_id, confidence, domain_rationale = _choose_domain(topic_text)
    else:
        if domain not in valid_domain_ids():
            valid = ", ".join(valid_domain_ids())
            raise ValueError(f"Invalid domain '{domain}'. Expected one of: auto, {valid}")
        domain_id = domain
        confidence = 1.0
        domain_rationale = "Domain was explicitly requested."

    if cuda_available is None:
        cuda_available = detect_cuda_available()

    backend_id = choose_execution_backend(
        domain_id,
        execution_backend,
        cuda_available=cuda_available,
    )
    budget_id, budget_rationale = _choose_budget(
        budget_profile,
        domain_id=domain_id,
        backend_id=backend_id,
    )
    evidence_level = evidence_level_for_backend(backend_id, budget_id)
    domain_pack = get_domain_pack(domain_id)

    profile = {
        "schema_version": 1,
        "domain": {
            "id": domain_pack.id,
            "confidence": confidence,
            "rationale": domain_rationale,
        },
        "execution": {
            "backend": backend_id,
            "budget_profile": budget_id,
            "evidence_level": evidence_level,
            "rationale": budget_rationale,
        },
        "claim_policy": claim_policy_for_evidence(evidence_level),
        "risk_flags": [
            "automatic domain selection may be wrong",
            "resource-limited execution may understate or overstate effects",
        ],
    }
    return validate_research_profile(profile)
