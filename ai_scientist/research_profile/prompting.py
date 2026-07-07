"""Prompt composition helpers for generalized runs."""

from __future__ import annotations

import json
import os.path as osp
from typing import Any

from .budgets import budget_prompt_guidance, get_budget_profile
from .domains import get_domain_pack
from .execution_backends import get_execution_backend
from .schema import validate_research_profile


def format_research_profile(profile: dict[str, Any]) -> str:
    return json.dumps(validate_research_profile(profile), indent=2)


def build_ideation_system_prompt(
    profile: dict[str, Any],
    tool_descriptions: str,
    tool_names_str: str,
) -> str:
    profile = validate_research_profile(profile)
    domain_pack = get_domain_pack(profile["domain"]["id"])
    backend = get_execution_backend(profile["execution"]["backend"])
    budget = get_budget_profile(profile["execution"]["budget_profile"])

    return f"""You are an experienced research scientist who proposes high-impact research ideas resembling exciting grant proposals. Feel free to propose novel ideas or experiments; make sure they are novel, feasible, and clearly distinguished from existing literature. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic.

Domain guidance:
{domain_pack.idea_generation_guidance}

Execution and evidence constraints:
- Execution backend: {backend.id} ({backend.display_name}). {backend.guidance}
- Budget profile: {budget.id}. Keep the proposal feasible under this budget.
- Evidence level: {profile["execution"]["evidence_level"]}.
- Allowed claim types: {", ".join(profile["claim_policy"]["allowed"])}.
- Forbidden claim types: {", ".join(profile["claim_policy"]["forbidden"])}.

Research Profile:
```json
{format_research_profile(profile)}
```

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {{"query": "your search query"}}. If ACTION is "FinalizeIdea", provide the idea details as {{"idea": {{ ... }}}} with the IDEA JSON specified below.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }}
}}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research."""


def build_environment_prompt(profile: dict[str, Any]) -> dict[str, str]:
    profile = validate_research_profile(profile)
    if profile["domain"]["id"] == "machine_learning":
        packages = (
            "numpy, pandas, scikit-learn, statsmodels, xgboost, lightGBM, "
            "torch, torchvision, torch-geometric, bayesian-optimization, timm, "
            "albumentations"
        )
        message = (
            "Your solution can use relevant packages such as: "
            f"{packages}. For neural networks, prefer PyTorch over TensorFlow."
        )
    else:
        packages = "numpy, pandas, scipy, statsmodels, matplotlib, seaborn, networkx"
        message = (
            "Your solution can use relevant Python packages such as: "
            f"{packages}. Use additional installed packages only when they are directly useful."
        )
    return {"Installed Packages": message}


def _common_code_guidance(timeout_seconds: int) -> list[str]:
    return [
        "Use executable Python when it helps validate the hypothesis or produce evidence.",
        "Important code structure requirements:",
        "  - Do NOT put execution code inside an if __name__ == \"__main__\" block.",
        "  - All code should be at global scope or in functions called from global scope.",
        "  - The script should execute immediately when run.",
        "The code should start with:",
        "  import os",
        "  working_dir = os.path.join(os.getcwd(), 'working')",
        "  os.makedirs(working_dir, exist_ok=True)",
        "The code should be a single-file python program that is self-contained and can be executed as-is.",
        "No parts of the code should be skipped; do not terminate execution before finishing the script.",
        "Your response should only contain a single code block.",
        f"Be aware of running time; the program must complete within about {timeout_seconds} seconds.",
        "Save structured result artifacts, such as metrics, tables, arrays, JSON, CSV, or compressed files, under the working directory.",
        "Use clear filenames so plotting and writeup stages can understand the generated artifacts.",
    ]


def _ml_code_guidance(
    profile: dict[str, Any],
    *,
    evaluation_metrics: Any = None,
    num_syn_datasets: int = 1,
    k_fold_validation: int = 1,
) -> list[str]:
    lines = [
        "ML implementation guidance:",
        "  - Define datasets, baselines, metrics, and ablations when they are relevant.",
        "  - Track and print validation loss or task-specific metrics at suitable intervals.",
        "  - Track and update these additional metrics when applicable: "
        + str(evaluation_metrics),
        "  - Save plottable data such as metrics, losses, predictions, and ground truth using np.save() or np.savez_compressed().",
        "  - For neural networks, prefer PyTorch.",
    ]
    if profile["execution"]["backend"] == "local_gpu_cuda_limited":
        lines.extend(
            [
                "CUDA guidance:",
                "  - At the start of your code, add device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').",
                "  - Print the selected device.",
                "  - Move models and tensor batches to device using .to(device).",
                "  - Create optimizers after moving the model to device.",
                "  - When using DataLoader, move batch tensors to device inside the training loop.",
            ]
        )
    else:
        lines.append("Use CPU-friendly ML settings unless CUDA is explicitly selected.")

    if num_syn_datasets > 1:
        lines.extend(
            [
                f"Evaluate on at least {num_syn_datasets} synthetic data variants when appropriate.",
                "Report metrics separately and include an aggregate metric.",
            ]
        )
    if k_fold_validation > 1:
        lines.append(
            f"Use {k_fold_validation}-fold cross-validation only if appropriate for the task."
        )
    return lines


def build_experiment_prompt_sections(
    profile: dict[str, Any],
    *,
    timeout_seconds: int,
    evaluation_metrics: Any = None,
    num_syn_datasets: int = 1,
    k_fold_validation: int = 1,
) -> dict[str, Any]:
    profile = validate_research_profile(profile)
    domain_pack = get_domain_pack(profile["domain"]["id"])
    backend = get_execution_backend(profile["execution"]["backend"])
    guidance = _common_code_guidance(timeout_seconds)
    guidance.extend(
        [
            "Domain guidance:",
            domain_pack.experiment_guidance,
            "Execution backend guidance:",
            backend.guidance,
        ]
    )
    guidance.extend(
        budget_prompt_guidance(
            profile["execution"]["budget_profile"],
            domain_id=profile["domain"]["id"],
        )
    )
    guidance.extend(
        [
            "Evidence and claim policy:",
            "  - Allowed claims: " + ", ".join(profile["claim_policy"]["allowed"]),
            "  - Forbidden claims: " + ", ".join(profile["claim_policy"]["forbidden"]),
        ]
    )
    if profile["domain"]["id"] == "machine_learning":
        guidance.extend(
            _ml_code_guidance(
                profile,
                evaluation_metrics=evaluation_metrics,
                num_syn_datasets=num_syn_datasets,
                k_fold_validation=k_fold_validation,
            )
        )
    else:
        guidance.extend(
            [
                "General research guidance:",
                "  - Choose validation methods that fit the question instead of forcing a training workflow.",
                "  - Prefer transparent calculations, simulations, sensitivity checks, or statistical summaries when appropriate.",
                "  - Report uncertainty and limitations directly.",
            ]
        )
    return {"Implementation guideline": guidance}


def build_writeup_guidance(profile: dict[str, Any] | None) -> str:
    if profile is None:
        return ""
    profile = validate_research_profile(profile)
    domain_pack = get_domain_pack(profile["domain"]["id"])
    return f"""Research Profile Guidance:
- Domain: {profile["domain"]["id"]}.
- Evidence level: {profile["execution"]["evidence_level"]}.
- Backend: {profile["execution"]["backend"]}.
- Budget profile: {profile["execution"]["budget_profile"]}.
- Domain writeup guidance: {domain_pack.writeup_guidance}
- Allowed claim types: {", ".join(profile["claim_policy"]["allowed"])}.
- Forbidden claim types: {", ".join(profile["claim_policy"]["forbidden"])}.

The paper must disclose resource-limited evidence when the evidence level is limited_empirical, smoke, or dry_run. Do not claim full validation unless the Research Profile permits it."""


def build_review_system_prompt(profile: dict[str, Any] | None) -> str:
    if profile is None:
        return (
            "You are an AI researcher reviewing a scholarly paper. "
            "Be critical and cautious in your decision."
        )
    profile = validate_research_profile(profile)
    domain_pack = get_domain_pack(profile["domain"]["id"])
    return (
        "You are an AI researcher reviewing a scholarly paper. Be critical and "
        "cautious in your decision. "
        + domain_pack.review_guidance
        + " Check whether the paper obeys this claim policy: allowed claims are "
        + ", ".join(profile["claim_policy"]["allowed"])
        + "; forbidden claims are "
        + ", ".join(profile["claim_policy"]["forbidden"])
        + "."
    )


def load_research_profile_from_run(base_folder: str) -> dict[str, Any] | None:
    config_path = osp.join(base_folder, "bfts_config.yaml")
    if not osp.exists(config_path):
        return None
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return None
    profile = config.get("research_profile")
    if profile is None:
        return None
    return validate_research_profile(profile)
