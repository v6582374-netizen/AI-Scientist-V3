"""Static built-in domain packs.

Domain packs provide prompt guidance only. They do not own tools, artifact
layout, or tree-search control flow.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainPack:
    id: str
    display_name: str
    idea_generation_guidance: str
    experiment_guidance: str
    writeup_guidance: str
    review_guidance: str
    default_execution_backend_preference: str
    default_budget_profile_preference: str


GENERAL_DOMAIN = DomainPack(
    id="general",
    display_name="General Research",
    idea_generation_guidance=(
        "Frame proposals around a clear research question, hypothesis, method, "
        "evidence plan, limitations, and validation strategy. Code may be used "
        "for simulation, statistical analysis, numerical checks, data processing, "
        "or reproducibility support when useful. Do not assume model training, "
        "benchmarks, datasets, or field-specific metrics unless the topic itself "
        "requires them."
    ),
    experiment_guidance=(
        "Use executable analysis only when it helps test the hypothesis. Suitable "
        "methods include simulation, statistics, numerical analysis, data cleaning, "
        "sanity checks, sensitivity analysis, and reproducible calculation. Keep "
        "the experiment design proportional to the selected budget."
    ),
    writeup_guidance=(
        "Write as a domain-neutral research paper. State the research question, "
        "method, evidence, limitations, and reproducibility details without "
        "forcing a field-specific conference framing."
    ),
    review_guidance=(
        "Review the paper as a general scholarly research contribution. Focus on "
        "whether claims are supported by the evidence, whether limitations are "
        "clear, and whether the method is reproducible."
    ),
    default_execution_backend_preference="local_cpu_limited",
    default_budget_profile_preference="small",
)


MACHINE_LEARNING_DOMAIN = DomainPack(
    id="machine_learning",
    display_name="Machine Learning",
    idea_generation_guidance=(
        "It is appropriate to discuss ML papers, benchmarks, baselines, ablations, "
        "datasets, model training, metrics, and publishability at top ML venues "
        "when they are relevant to the topic."
    ),
    experiment_guidance=(
        "For ML experiments, define datasets, baselines, metrics, ablations, "
        "training loops, hyperparameters, and multi-seed evaluation where useful. "
        "Prefer PyTorch for neural network implementations."
    ),
    writeup_guidance=(
        "Use ML paper conventions when the selected idea is an ML contribution: "
        "clear baselines, metrics, ablations, limitations, and reproducibility "
        "details."
    ),
    review_guidance=(
        "Review the paper using ML venue expectations: novelty, empirical "
        "support, baseline strength, reproducibility, clarity, and limitations."
    ),
    default_execution_backend_preference="local_cpu_limited",
    default_budget_profile_preference="small",
)


DOMAIN_PACKS = {
    GENERAL_DOMAIN.id: GENERAL_DOMAIN,
    MACHINE_LEARNING_DOMAIN.id: MACHINE_LEARNING_DOMAIN,
}


def get_domain_pack(domain_id: str) -> DomainPack:
    try:
        return DOMAIN_PACKS[domain_id]
    except KeyError as exc:
        valid = ", ".join(sorted(DOMAIN_PACKS))
        raise ValueError(f"Unknown domain '{domain_id}'. Expected one of: {valid}") from exc


def valid_domain_ids() -> tuple[str, ...]:
    return tuple(sorted(DOMAIN_PACKS))
