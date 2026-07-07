"""Validation helpers for Research Profiles and schema v2 idea files."""

from __future__ import annotations

import copy
from typing import Any

from .budgets import valid_budget_profile_ids
from .domains import valid_domain_ids
from .execution_backends import (
    claim_policy_for_evidence,
    evidence_level_for_backend,
    valid_evidence_levels,
    valid_execution_backend_ids,
)


AUTO = "auto"
IDEA_SCHEMA_VERSION = 2
RESEARCH_PROFILE_SCHEMA_VERSION = 1


def _validate_choice(value: str, *, valid: tuple[str, ...], field_name: str) -> str:
    if value not in valid:
        expected = ", ".join(valid)
        raise ValueError(f"Invalid {field_name} '{value}'. Expected one of: {expected}")
    return value


def validate_research_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("research_profile must be a JSON object.")

    profile = copy.deepcopy(profile)
    schema_version = profile.get("schema_version", RESEARCH_PROFILE_SCHEMA_VERSION)
    if schema_version != RESEARCH_PROFILE_SCHEMA_VERSION:
        raise ValueError("research_profile.schema_version must be 1.")
    profile["schema_version"] = schema_version

    domain = profile.get("domain")
    if not isinstance(domain, dict):
        raise ValueError("research_profile.domain must be a JSON object.")
    domain_id = _validate_choice(
        domain.get("id"),
        valid=valid_domain_ids(),
        field_name="domain",
    )
    domain.setdefault("confidence", 1.0)
    domain.setdefault("rationale", "Domain was provided by configuration.")
    domain["id"] = domain_id

    execution = profile.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("research_profile.execution must be a JSON object.")
    backend = _validate_choice(
        execution.get("backend"),
        valid=valid_execution_backend_ids(),
        field_name="execution backend",
    )
    budget = _validate_choice(
        execution.get("budget_profile"),
        valid=valid_budget_profile_ids(),
        field_name="budget profile",
    )
    evidence_level = execution.get("evidence_level") or evidence_level_for_backend(
        backend,
        budget,
    )
    execution["evidence_level"] = _validate_choice(
        evidence_level,
        valid=valid_evidence_levels(),
        field_name="evidence level",
    )
    execution.setdefault("rationale", "Execution settings were selected automatically.")

    claim_policy = profile.get("claim_policy")
    if claim_policy is None:
        claim_policy = claim_policy_for_evidence(execution["evidence_level"])
        profile["claim_policy"] = claim_policy
    if not isinstance(claim_policy, dict):
        raise ValueError("research_profile.claim_policy must be a JSON object.")
    for key in ("allowed", "forbidden"):
        value = claim_policy.get(key)
        if value is None:
            claim_policy[key] = []
        elif not isinstance(value, list):
            raise ValueError(f"research_profile.claim_policy.{key} must be a list.")

    risk_flags = profile.get("risk_flags")
    if risk_flags is None:
        profile["risk_flags"] = []
    elif not isinstance(risk_flags, list):
        raise ValueError("research_profile.risk_flags must be a list.")

    return profile


def make_idea_envelope(
    research_profile: dict[str, Any],
    ideas: list[dict[str, Any]],
) -> dict[str, Any]:
    envelope = {
        "schema_version": IDEA_SCHEMA_VERSION,
        "research_profile": validate_research_profile(research_profile),
        "ideas": copy.deepcopy(ideas),
    }
    return validate_idea_envelope(envelope)


def validate_idea_envelope(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        raise ValueError(
            "Generalized AI Scientist requires idea schema version 2; "
            "legacy top-level idea arrays are not supported."
        )
    if not isinstance(payload, dict):
        raise ValueError("Idea file must be a JSON object using schema version 2.")
    if payload.get("schema_version") != IDEA_SCHEMA_VERSION:
        raise ValueError("Idea file must declare schema version 2.")

    profile = validate_research_profile(payload.get("research_profile"))
    ideas = payload.get("ideas")
    if not isinstance(ideas, list) or not ideas:
        raise ValueError("Idea file schema version 2 requires a non-empty ideas list.")

    validated_ideas = []
    for index, idea in enumerate(ideas):
        if not isinstance(idea, dict):
            raise ValueError(f"ideas[{index}] must be a JSON object.")
        if not idea.get("Name"):
            raise ValueError(f"ideas[{index}] must include a non-empty Name field.")
        validated_ideas.append(copy.deepcopy(idea))

    return {
        "schema_version": IDEA_SCHEMA_VERSION,
        "research_profile": profile,
        "ideas": validated_ideas,
    }


def apply_profile_overrides(
    profile: dict[str, Any],
    *,
    domain: str = AUTO,
    execution_backend: str = AUTO,
    budget_profile: str = AUTO,
) -> dict[str, Any]:
    updated = validate_research_profile(profile)

    if domain != AUTO:
        updated["domain"]["id"] = _validate_choice(
            domain,
            valid=valid_domain_ids(),
            field_name="domain",
        )
        updated["domain"]["confidence"] = 1.0
        updated["domain"]["rationale"] = "Domain was explicitly overridden by CLI."

    if execution_backend != AUTO:
        updated["execution"]["backend"] = _validate_choice(
            execution_backend,
            valid=valid_execution_backend_ids(),
            field_name="execution backend",
        )
        updated["execution"]["rationale"] = (
            "Execution backend was explicitly overridden by CLI."
        )

    if budget_profile != AUTO:
        updated["execution"]["budget_profile"] = _validate_choice(
            budget_profile,
            valid=valid_budget_profile_ids(),
            field_name="budget profile",
        )
        updated["execution"]["rationale"] = "Budget profile was explicitly overridden by CLI."

    evidence_level = evidence_level_for_backend(
        updated["execution"]["backend"],
        updated["execution"]["budget_profile"],
    )
    updated["execution"]["evidence_level"] = evidence_level
    updated["claim_policy"] = claim_policy_for_evidence(evidence_level)
    return validate_research_profile(updated)


def select_idea(envelope: dict[str, Any], idea_idx: int) -> dict[str, Any]:
    validated = validate_idea_envelope(envelope)
    ideas = validated["ideas"]
    if idea_idx < 0 or idea_idx >= len(ideas):
        raise ValueError(
            f"idea_idx {idea_idx} is out of range for {len(ideas)} generated ideas."
        )
    return copy.deepcopy(ideas[idea_idx])
