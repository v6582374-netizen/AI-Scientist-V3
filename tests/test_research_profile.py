import copy
from argparse import Namespace
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

from ai_scientist.research_profile.budgets import apply_budget_profile_to_config
from ai_scientist.research_profile.planner import plan_research_profile
from ai_scientist.research_profile.prompting import (
    build_experiment_prompt_sections,
    build_ideation_system_prompt,
    build_review_system_prompt,
    build_writeup_guidance,
)
from ai_scientist.research_profile.schema import (
    apply_profile_overrides,
    make_idea_envelope,
    validate_idea_envelope,
)


class ResearchProfileTests(unittest.TestCase):
    def test_rejects_legacy_top_level_idea_array(self):
        with self.assertRaisesRegex(ValueError, "schema version 2"):
            validate_idea_envelope([{"Name": "old"}])

    def test_general_topic_has_no_ml_prompt_contamination(self):
        profile = plan_research_profile(
            "Study how neighborhood tree canopy affects sidewalk temperature using field measurements.",
            domain="auto",
            execution_backend="auto",
            budget_profile="auto",
            cuda_available=False,
        )
        prompt = build_ideation_system_prompt(
            profile,
            "- SearchSemanticScholar",
            '"SearchSemanticScholar", "FinalizeIdea"',
        )
        self.assertEqual(profile["domain"]["id"], "general")
        for forbidden in ["top ML", "PyTorch", "CUDA", "DataLoader", "accuracy/loss"]:
            self.assertNotIn(forbidden, prompt)

    def test_ml_topic_preserves_ml_domain_guidance(self):
        profile = plan_research_profile(
            "Train a transformer model and compare accuracy against deep learning baselines.",
            domain="auto",
            execution_backend="local_gpu_cuda_limited",
            budget_profile="medium",
            cuda_available=True,
        )
        prompt_sections = build_experiment_prompt_sections(
            profile,
            timeout_seconds=3600,
        )
        text = "\n".join(str(v) for v in prompt_sections.values())
        self.assertEqual(profile["domain"]["id"], "machine_learning")
        self.assertIn("PyTorch", text)
        self.assertIn("CUDA", text)

    def test_general_experiment_guidance_has_no_ml_defaults(self):
        profile = plan_research_profile(
            "Analyze whether library opening hours affect community participation.",
            domain="general",
            execution_backend="local_cpu_limited",
            budget_profile="small",
            cuda_available=False,
        )
        prompt_sections = build_experiment_prompt_sections(
            profile,
            timeout_seconds=1800,
        )
        text = "\n".join(str(v) for v in prompt_sections.values())
        for forbidden in [
            "torch.device",
            "CUDA",
            "DataLoader",
            "validation loss",
            "epoch",
        ]:
            self.assertNotIn(forbidden, text)
        self.assertIn("single-file python program", text)
        self.assertIn("structured result", text)

    def test_budget_profile_applies_hard_limits(self):
        config = {
            "exec": {"timeout": 3600},
            "agent": {
                "num_workers": 4,
                "steps": 5,
                "stages": {
                    "stage1_max_iters": 20,
                    "stage2_max_iters": 12,
                    "stage3_max_iters": 12,
                    "stage4_max_iters": 18,
                },
                "multi_seed_eval": {"num_seeds": 3},
            },
        }
        apply_budget_profile_to_config(config, "small")
        self.assertEqual(config["agent"]["num_workers"], 1)
        self.assertEqual(config["agent"]["multi_seed_eval"]["num_seeds"], 1)
        self.assertEqual(config["exec"]["timeout"], 1800)
        self.assertEqual(config["agent"]["stages"]["stage1_max_iters"], 4)

    def test_schema_v2_envelope_round_trip_and_overrides(self):
        profile = plan_research_profile(
            "Analyze municipal water usage records.",
            cuda_available=False,
        )
        envelope = make_idea_envelope(
            profile,
            [{"Name": "water_usage", "Title": "Water Usage"}],
        )
        validated = validate_idea_envelope(copy.deepcopy(envelope))
        overridden = apply_profile_overrides(
            validated["research_profile"],
            budget_profile="tiny",
        )
        self.assertEqual(validated["schema_version"], 2)
        self.assertEqual(overridden["execution"]["budget_profile"], "tiny")
        self.assertEqual(set(validated.keys()), {"schema_version", "research_profile", "ideas"})

    def test_ideation_rejects_legacy_reload_before_llm_call(self):
        from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea

        profile = plan_research_profile(
            "Analyze municipal water usage records.",
            cuda_available=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            idea_path = Path(tmpdir) / "ideas.json"
            idea_path.write_text('[{"Name": "legacy"}]', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "schema version 2"):
                generate_temp_free_idea(
                    idea_fname=str(idea_path),
                    client=None,
                    model="unused",
                    workshop_description="unused",
                    research_profile=profile,
                    max_num_generations=0,
                )

    def test_edit_bfts_config_stores_profile_and_applies_budget(self):
        import yaml

        from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file

        profile = plan_research_profile(
            "Analyze municipal water usage records.",
            domain="general",
            execution_backend="local_cpu_limited",
            budget_profile="small",
            cuda_available=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            run_dir = tmp / "run"
            source_dir.mkdir()
            run_dir.mkdir()
            config_path = source_dir / "bfts_config.yaml"
            idea_path = run_dir / "idea.json"
            idea_path.write_text('{"Name": "water_usage"}', encoding="utf-8")
            config_path.write_text(
                """
data_dir: data
desc_file: null
goal: null
eval: null
log_dir: logs
workspace_dir: workspaces
preprocess_data: false
copy_data: true
exp_name: run
exec:
  timeout: 3600
  agent_file_name: runfile.py
  format_tb_ipython: false
generate_report: true
report: {}
experiment:
  num_syn_datasets: 1
debug:
  stage4: false
agent:
  type: parallel
  num_workers: 4
  stages:
    stage1_max_iters: 20
    stage2_max_iters: 12
    stage3_max_iters: 12
    stage4_max_iters: 18
  steps: 5
  k_fold_validation: 1
  multi_seed_eval:
    num_seeds: 3
  expose_prediction: false
  data_preview: false
  code: {}
  feedback: {}
  vlm_feedback: {}
  search: {}
""",
                encoding="utf-8",
            )

            run_config_path = edit_bfts_config_file(
                str(config_path),
                str(run_dir),
                str(idea_path),
                research_profile=profile,
            )
            saved = yaml.safe_load(Path(run_config_path).read_text(encoding="utf-8"))

        self.assertEqual(saved["research_profile"]["domain"]["id"], "general")
        self.assertEqual(saved["agent"]["num_workers"], 1)
        self.assertEqual(saved["agent"]["multi_seed_eval"]["num_seeds"], 1)
        self.assertEqual(saved["exec"]["timeout"], 1800)

    def test_default_bfts_config_does_not_require_bedrock(self):
        import yaml
        from ai_scientist.model_defaults import DEFAULT_MODEL

        config = yaml.safe_load(Path("bfts_config.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["report"]["model"], DEFAULT_MODEL)
        self.assertEqual(config["agent"]["code"]["model"], DEFAULT_MODEL)
        self.assertEqual(config["agent"]["feedback"]["model"], DEFAULT_MODEL)
        self.assertEqual(config["agent"]["vlm_feedback"]["model"], DEFAULT_MODEL)
        self.assertNotIn("anthropic.claude", config["agent"]["code"]["model"])

    def test_edit_bfts_config_overrides_experiment_stage_models(self):
        import yaml

        from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_dir = tmp / "source"
            run_dir = tmp / "run"
            source_dir.mkdir()
            run_dir.mkdir()
            config_path = source_dir / "bfts_config.yaml"
            idea_path = run_dir / "idea.json"
            idea_path.write_text('{"Name": "water_usage"}', encoding="utf-8")
            config_path.write_text(
                """
data_dir: data
desc_file: null
goal: null
eval: null
log_dir: logs
workspace_dir: workspaces
preprocess_data: false
copy_data: true
exp_name: run
exec:
  timeout: 3600
  agent_file_name: runfile.py
  format_tb_ipython: false
generate_report: true
report: {}
experiment:
  num_syn_datasets: 1
debug:
  stage4: false
agent:
  type: parallel
  num_workers: 4
  stages: {}
  steps: 5
  k_fold_validation: 1
  multi_seed_eval:
    num_seeds: 3
  expose_prediction: false
  data_preview: false
  code:
    model: anthropic.claude-3-5-sonnet-20241022-v2:0
  feedback:
    model: gpt-4o-2024-11-20
  vlm_feedback:
    model: gpt-4o-2024-11-20
  search: {}
""",
                encoding="utf-8",
            )

            run_config_path = edit_bfts_config_file(
                str(config_path),
                str(run_dir),
                str(idea_path),
                bfts_model_overrides={
                    "code": "qwen/qwen-plus",
                    "feedback": "qwen/qwen-plus",
                    "vlm_feedback": "qwen/qwen-vl-plus",
                },
            )
            saved = yaml.safe_load(Path(run_config_path).read_text(encoding="utf-8"))

        self.assertEqual(saved["agent"]["code"]["model"], "qwen/qwen-plus")
        self.assertEqual(saved["agent"]["feedback"]["model"], "qwen/qwen-plus")
        self.assertEqual(
            saved["agent"]["vlm_feedback"]["model"], "qwen/qwen-vl-plus"
        )

    def test_launch_builds_bfts_model_overrides_from_args(self):
        from launch_scientist_bfts import build_bfts_model_overrides

        args = Namespace(
            model_bfts_code="qwen/qwen-plus",
            model_bfts_feedback="qwen/qwen-plus",
            model_bfts_vlm_feedback="qwen/qwen-vl-plus",
        )

        self.assertEqual(
            build_bfts_model_overrides(args),
            {
                "code": "qwen/qwen-plus",
                "feedback": "qwen/qwen-plus",
                "vlm_feedback": "qwen/qwen-vl-plus",
            },
        )

    def test_launch_model_defaults_use_shared_default_model(self):
        from unittest.mock import patch

        from ai_scientist.model_defaults import DEFAULT_MODEL
        from launch_scientist_bfts import parse_arguments

        with patch(
            "sys.argv",
            ["launch_scientist_bfts.py", "--load_ideas", "ideas.json"],
        ):
            args = parse_arguments()

        self.assertEqual(args.model_agg_plots, DEFAULT_MODEL)
        self.assertEqual(args.model_writeup, DEFAULT_MODEL)
        self.assertEqual(args.model_citation, DEFAULT_MODEL)
        self.assertEqual(args.model_writeup_small, DEFAULT_MODEL)
        self.assertEqual(args.model_review, DEFAULT_MODEL)
        self.assertEqual(args.model_bfts_code, DEFAULT_MODEL)
        self.assertEqual(args.model_bfts_feedback, DEFAULT_MODEL)
        self.assertEqual(args.model_bfts_vlm_feedback, DEFAULT_MODEL)

    def test_stage_function_defaults_use_shared_default_model(self):
        import inspect

        from ai_scientist.model_defaults import DEFAULT_MODEL
        from ai_scientist import (
            perform_icbinb_writeup,
            perform_plotting,
            perform_writeup,
        )

        self.assertEqual(
            inspect.signature(perform_plotting.aggregate_plots)
            .parameters["model"]
            .default,
            DEFAULT_MODEL,
        )
        self.assertEqual(
            inspect.signature(perform_writeup.perform_writeup)
            .parameters["small_model"]
            .default,
            DEFAULT_MODEL,
        )
        self.assertEqual(
            inspect.signature(perform_writeup.perform_writeup)
            .parameters["big_model"]
            .default,
            DEFAULT_MODEL,
        )
        self.assertEqual(
            inspect.signature(perform_icbinb_writeup.gather_citations)
            .parameters["small_model"]
            .default,
            DEFAULT_MODEL,
        )
        self.assertEqual(
            inspect.signature(perform_icbinb_writeup.perform_writeup)
            .parameters["small_model"]
            .default,
            DEFAULT_MODEL,
        )
        self.assertEqual(
            inspect.signature(perform_icbinb_writeup.perform_writeup)
            .parameters["big_model"]
            .default,
            DEFAULT_MODEL,
        )

    def test_launch_rejects_ml_only_options_for_general_domain(self):
        from launch_scientist_bfts import validate_launch_options

        profile = plan_research_profile(
            "Analyze municipal water usage records.",
            domain="general",
            execution_backend="local_cpu_limited",
            budget_profile="small",
            cuda_available=False,
        )
        args = Namespace(
            load_code=True,
            add_dataset_ref=False,
            writeup_type="normal",
        )
        with self.assertRaisesRegex(ValueError, "ML-oriented"):
            validate_launch_options(args, profile)

    def test_launch_accepts_default_normal_writeup_for_general_domain(self):
        from launch_scientist_bfts import validate_launch_options

        profile = plan_research_profile(
            "Analyze municipal water usage records.",
            domain="general",
            execution_backend="local_cpu_limited",
            budget_profile="small",
            cuda_available=False,
        )
        args = Namespace(
            load_code=False,
            add_dataset_ref=False,
            writeup_type="normal",
        )
        self.assertIsNone(validate_launch_options(args, profile))

    def test_minimal_agent_general_prompt_uses_profile(self):
        from ai_scientist.treesearch.parallel_agent import MinimalAgent

        profile = plan_research_profile(
            "Analyze municipal water usage records.",
            domain="general",
            execution_backend="local_cpu_limited",
            budget_profile="small",
            cuda_available=False,
        )
        cfg = SimpleNamespace(
            research_profile=profile,
            exec=SimpleNamespace(timeout=1800),
            experiment=SimpleNamespace(num_syn_datasets=1),
            agent=SimpleNamespace(k_fold_validation=1),
        )
        agent = MinimalAgent("Analyze municipal water usage records.", cfg)
        text = "\n".join(agent._prompt_impl_guideline["Implementation guideline"])
        env_text = "\n".join(agent._prompt_environment.values())

        self.assertNotIn("CUDA", text)
        self.assertNotIn("torch.device", text)
        self.assertNotIn("PyTorch", env_text)
        self.assertIn("structured result", text)

    def test_writeup_and_review_guidance_respects_claim_policy(self):
        profile = plan_research_profile(
            "Analyze municipal water usage records.",
            domain="general",
            execution_backend="local_cpu_limited",
            budget_profile="small",
            cuda_available=False,
        )
        writeup_guidance = build_writeup_guidance(profile)
        review_prompt = build_review_system_prompt(profile)

        self.assertIn("resource-limited", writeup_guidance)
        self.assertIn("full empirical validation", writeup_guidance)
        self.assertNotIn("prestigious ML venue", review_prompt)
        self.assertIn("scholarly paper", review_prompt)


if __name__ == "__main__":
    unittest.main()
