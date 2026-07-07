import os
import os.path as osp
import shutil
import yaml

from ai_scientist.research_profile.budgets import apply_budget_profile_to_config
from ai_scientist.research_profile.schema import validate_research_profile

BFTS_MODEL_OVERRIDE_SECTIONS = {"code", "feedback", "vlm_feedback"}


def idea_to_markdown(data: dict, output_path: str, load_code: str) -> None:
    """
    Convert a dictionary into a markdown file.

    Args:
        data: Dictionary containing the data to convert
        output_path: Path where the markdown file will be saved
        load_code: Path to a code file to include in the markdown
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            # Convert key to title format and make it a header
            header = key.replace("_", " ").title()
            f.write(f"## {header}\n\n")

            # Handle different value types
            if isinstance(value, (list, tuple)):
                for item in value:
                    f.write(f"- {item}\n")
                f.write("\n")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"### {sub_key}\n")
                    f.write(f"{sub_value}\n\n")
            else:
                f.write(f"{value}\n\n")

        # Add the code to the markdown file
        if load_code:
            # Assert that the code file exists before trying to open it
            assert os.path.exists(load_code), f"Code path at {load_code} must exist if using the 'load_code' flag. This is an optional code prompt that you may choose to include; if not, please do not set 'load_code'."
            f.write(f"## Code To Potentially Use\n\n")
            f.write(f"Use the following code as context for your experiments:\n\n")
            with open(load_code, "r") as code_file:
                code = code_file.read()
                f.write(f"```python\n{code}\n```\n\n")


def edit_bfts_config_file(
    config_path: str,
    idea_dir: str,
    idea_path: str,
    research_profile: dict | None = None,
    bfts_model_overrides: dict[str, str] | None = None,
) -> str:
    """
    Edit the bfts_config.yaml file to point to the idea.md file

    Args:
        config_path: Path to the bfts_config.yaml file
        idea_dir: Directory where the idea.md file is located
        idea_path: Path to the idea.md file
        research_profile: Optional generalized Research Profile to persist
        bfts_model_overrides: Optional agent model overrides for BFTS stages

    Returns:
        Path to the edited bfts_config.yaml file
    """
    run_config_path = osp.join(idea_dir, "bfts_config.yaml")
    shutil.copy(config_path, run_config_path)
    with open(run_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["desc_file"] = idea_path
    config["workspace_dir"] = idea_dir

    # make an empty data directory
    data_dir = osp.join(idea_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    config["data_dir"] = data_dir

    # make an empty log directory
    log_dir = osp.join(idea_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

    if research_profile is not None:
        research_profile = validate_research_profile(research_profile)
        config["research_profile"] = research_profile
        apply_budget_profile_to_config(
            config,
            research_profile["execution"]["budget_profile"],
        )

    if bfts_model_overrides:
        invalid_sections = set(bfts_model_overrides) - BFTS_MODEL_OVERRIDE_SECTIONS
        if invalid_sections:
            raise ValueError(
                "Unknown BFTS model override section(s): "
                f"{', '.join(sorted(invalid_sections))}"
            )
        agent_config = config.setdefault("agent", {})
        for section, model in bfts_model_overrides.items():
            if not model:
                continue
            agent_config.setdefault(section, {})["model"] = model

    with open(run_config_path, "w") as f:
        yaml.dump(config, f)
    return run_config_path
