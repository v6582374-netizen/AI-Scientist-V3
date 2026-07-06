# Generalized AI Scientist Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first generalized AI Scientist vertical slice: schema v2 idea files, Research Profile planning, domain-aware prompts, local budget enforcement, and evidence-aware writeup/review.

**Architecture:** Keep the existing two-step flow and existing artifact layout. Add a small `ai_scientist/research_profile/` package for reusable profile, domain, budget, backend, and prompt helpers; existing entrypoints call those helpers instead of embedding generalized logic directly.

**Tech Stack:** Python stdlib dataclasses/JSON, OmegaConf/YAML-compatible dictionaries, existing `argparse`, existing BFTS tree search, `unittest` for no-new-dependency tests.

---

## File Structure

- Create: `ai_scientist/research_profile/__init__.py` exports public helpers.
- Create: `ai_scientist/research_profile/schema.py` validates Research Profile and schema v2 idea envelopes.
- Create: `ai_scientist/research_profile/domains.py` defines static `general` and `machine_learning` domain packs.
- Create: `ai_scientist/research_profile/budgets.py` defines static `tiny/small/medium/full` budgets and applies hard config limits.
- Create: `ai_scientist/research_profile/execution_backends.py` defines local backend metadata, CUDA detection, and evidence mapping.
- Create: `ai_scientist/research_profile/planner.py` chooses a Research Profile from topic text plus CLI overrides.
- Create: `ai_scientist/research_profile/prompting.py` composes ideation, experiment, writeup, and review guidance.
- Create: `tests/test_research_profile.py` tests schema, planner, prompt composition, and budget application without LLM calls.
- Modify: `ai_scientist/perform_ideation_temp_free.py` requires explicit topic file, creates Research Profile, writes schema v2 envelope.
- Modify: `launch_scientist_bfts.py` requires schema v2 idea file, applies overrides/budget/backend, rejects ML-only options in `general`.
- Modify: `ai_scientist/treesearch/bfts_utils.py` stores profile and budget in per-run `bfts_config.yaml`.
- Modify: `ai_scientist/treesearch/utils/config.py` adds `research_profile` typing to the structured config.
- Modify: `ai_scientist/treesearch/parallel_agent.py` replaces hard-coded ML/CUDA implementation guidance with composed guidance from the Research Profile.
- Modify: `ai_scientist/perform_writeup.py`, `ai_scientist/perform_icbinb_writeup.py`, and `launch_scientist_bfts.py` pass evidence/claim policy into writeup/review.

## Task 1: Research Profile Core

**Files:**
- Create: `ai_scientist/research_profile/__init__.py`
- Create: `ai_scientist/research_profile/schema.py`
- Create: `ai_scientist/research_profile/domains.py`
- Create: `ai_scientist/research_profile/budgets.py`
- Create: `ai_scientist/research_profile/execution_backends.py`
- Create: `ai_scientist/research_profile/planner.py`
- Create: `ai_scientist/research_profile/prompting.py`
- Test: `tests/test_research_profile.py`

- [ ] **Step 1: Write failing schema and planner tests**

```python
import copy
import unittest

from ai_scientist.research_profile.budgets import apply_budget_profile_to_config
from ai_scientist.research_profile.planner import plan_research_profile
from ai_scientist.research_profile.prompting import (
    build_experiment_prompt_sections,
    build_ideation_system_prompt,
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
        prompt = build_ideation_system_prompt(profile, "- SearchSemanticScholar", '"SearchSemanticScholar", "FinalizeIdea"')
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
        prompt_sections = build_experiment_prompt_sections(profile, timeout_seconds=3600)
        text = "\n".join(str(v) for v in prompt_sections.values())
        self.assertEqual(profile["domain"]["id"], "machine_learning")
        self.assertIn("PyTorch", text)
        self.assertIn("CUDA", text)

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
        profile = plan_research_profile("Analyze municipal water usage records.", cuda_available=False)
        envelope = make_idea_envelope(profile, [{"Name": "water_usage", "Title": "Water Usage"}])
        validated = validate_idea_envelope(copy.deepcopy(envelope))
        overridden = apply_profile_overrides(validated["research_profile"], budget_profile="tiny")
        self.assertEqual(validated["schema_version"], 2)
        self.assertEqual(overridden["execution"]["budget_profile"], "tiny")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_research_profile -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'ai_scientist.research_profile'`.

- [ ] **Step 3: Implement core modules**

Implement dataclasses and dictionary helpers with these public functions:

```python
plan_research_profile(topic_text, domain="auto", execution_backend="auto", budget_profile="auto", cuda_available=None) -> dict
validate_research_profile(profile: dict) -> dict
validate_idea_envelope(payload: object) -> dict
make_idea_envelope(profile: dict, ideas: list[dict]) -> dict
apply_profile_overrides(profile: dict, domain="auto", execution_backend="auto", budget_profile="auto") -> dict
apply_budget_profile_to_config(config: dict, budget_profile_id: str) -> dict
build_ideation_system_prompt(profile: dict, tool_descriptions: str, tool_names_str: str) -> str
build_experiment_prompt_sections(profile: dict, timeout_seconds: int, evaluation_metrics=None, num_syn_datasets=1, k_fold_validation=1) -> dict
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest tests.test_research_profile -v`

Expected: PASS all tests in `ResearchProfileTests`.

## Task 2: Ideation Schema v2

**Files:**
- Modify: `ai_scientist/perform_ideation_temp_free.py`
- Test: `tests/test_research_profile.py`

- [ ] **Step 1: Add failing tests for idea envelope output helpers**

Add a test that calls `make_idea_envelope` with a generated idea and asserts top-level keys are `schema_version`, `research_profile`, and `ideas`.

- [ ] **Step 2: Run test and verify RED if helper behavior is missing**

Run: `python -m unittest tests.test_research_profile.ResearchProfileTests.test_schema_v2_envelope_round_trip_and_overrides -v`

- [ ] **Step 3: Modify ideation entrypoint**

Change `--workshop-file` to `required=True`, add `--domain`, `--execution-backend`, and `--budget-profile`, call `plan_research_profile`, pass the profile into `generate_temp_free_idea`, and write:

```json
{
  "schema_version": 2,
  "research_profile": { "...": "..." },
  "ideas": [{ "Name": "...", "Title": "..." }]
}
```

Also load existing output only through `validate_idea_envelope`; reject legacy arrays before any LLM calls.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_research_profile -v`

Expected: PASS.

## Task 3: Launch Validation, Budget, and Backend

**Files:**
- Modify: `launch_scientist_bfts.py`
- Modify: `ai_scientist/treesearch/bfts_utils.py`
- Modify: `ai_scientist/treesearch/utils/config.py`
- Test: `tests/test_research_profile.py`

- [ ] **Step 1: Write failing tests for config mutation**

Add a temp-file test that writes a minimal BFTS config, calls `edit_bfts_config_file(..., research_profile=profile)`, reads YAML, and asserts `research_profile` is present and `small` budget limits are applied.

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_research_profile -v`

Expected: FAIL because `edit_bfts_config_file` does not accept `research_profile`.

- [ ] **Step 3: Implement launch validation and config mutation**

Update launch to:

```python
with open(args.load_ideas, "r") as f:
    envelope = validate_idea_envelope(json.load(f))
research_profile = apply_profile_overrides(
    envelope["research_profile"],
    domain=args.domain,
    execution_backend=args.execution_backend,
    budget_profile=args.budget_profile,
)
ideas = envelope["ideas"]
idea = select_idea(envelope, args.idea_idx)
```

Reject `--load_code`, `--add_dataset_ref`, and `--writeup-type icbinb` when `domain.id == "general"`. Apply CPU backend by setting `CUDA_VISIBLE_DEVICES=""` before experiments. Pass `research_profile` into `edit_bfts_config_file`.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_research_profile -v`

Expected: PASS.

## Task 4: Tree Search Prompt Composition

**Files:**
- Modify: `ai_scientist/treesearch/parallel_agent.py`
- Test: `tests/test_research_profile.py`

- [ ] **Step 1: Add failing prompt contamination tests**

Assert `build_experiment_prompt_sections(general_profile, ...)` has no `torch.device`, `CUDA`, `DataLoader`, `validation loss`, or mandatory `epoch`, and that ML GPU profile includes CUDA/PyTorch guidance.

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_research_profile -v`

- [ ] **Step 3: Replace hard-coded prompt blocks**

Change `MinimalAgent._prompt_environment` and `MinimalAgent._prompt_impl_guideline` to call `build_environment_prompt` and `build_experiment_prompt_sections`. Keep response formats and tree search behavior intact.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_research_profile -v`

Expected: PASS.

## Task 5: Evidence-Aware Writeup and Review

**Files:**
- Modify: `ai_scientist/perform_writeup.py`
- Modify: `ai_scientist/perform_icbinb_writeup.py`
- Modify: `launch_scientist_bfts.py`
- Test: `tests/test_research_profile.py`

- [ ] **Step 1: Add failing tests for claim policy guidance**

Test `build_writeup_guidance(profile)` for `limited_empirical` contains `resource-limited` and forbids full validation claims.

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_research_profile -v`

- [ ] **Step 3: Inject profile guidance into writeup and review**

Read `research_profile` from `base_folder/bfts_config.yaml`, append `build_writeup_guidance(profile)` to writeup system/user prompts, and call review with `reviewer_system_prompt=build_review_system_prompt(profile)`.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_research_profile -v`

Expected: PASS.

## Task 6: Final Verification and Commit

**Files:**
- All modified files

- [ ] **Step 1: Run unit tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Run syntax compilation**

Run: `python -m compileall ai_scientist launch_scientist_bfts.py`

Expected: exit code 0.

- [ ] **Step 3: Inspect diff**

Run: `git diff --stat && git diff --check`

Expected: no whitespace errors; diff only covers planned files.

- [ ] **Step 4: Commit**

```bash
git add ai_scientist/research_profile tests/test_research_profile.py ai_scientist/perform_ideation_temp_free.py launch_scientist_bfts.py ai_scientist/treesearch/bfts_utils.py ai_scientist/treesearch/utils/config.py ai_scientist/treesearch/parallel_agent.py ai_scientist/perform_writeup.py ai_scientist/perform_icbinb_writeup.py docs/superpowers/plans/2026-07-07-generalized-ai-scientist-core.md
git commit -m "Implement generalized research profile core"
```

## Self-Review

- Spec coverage: Domain neutrality, two-step workflow, schema v2, built-in domain packs, static budgets, local execution backends, budget enforcement, prompt composition, artifact placement, and evidence-aware writeup/review are covered.
- First-version exclusions: remote/cloud execution, third-party plugins, human-in-loop, adaptive runtime budgeting, and capability-pack decomposition are not implemented.
- Placeholder scan: no task uses TBD/TODO/fill-in placeholders; implementation functions and file paths are explicit.
- Risk: `perform_icbinb_writeup.py` remains intentionally ML/workshop-specific and is rejected for `domain=general` at launch time.
