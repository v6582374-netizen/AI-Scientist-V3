# Generalized AI Scientist Design Spec

Status: Draft for review  
Date: 2026-07-07  
Related glossary: `CONTEXT.md`  
Related decisions: `docs/adr/0001-*.md` through `docs/adr/0023-*.md`

## Purpose

The current project is valuable because it already combines idea generation, literature search, tree search, executable experiments, plotting, writeup, and review into an automated research loop. The first innovation should preserve that loop while removing the hidden assumption that every research problem is a machine learning problem.

This spec defines a first-version generalization architecture. The goal is not to build a full plugin platform. The goal is to make the system domain-neutral by default, keep the existing machine-learning workflow available as an explicit domain choice, and make local execution resource limits explicit enough for ordinary computers.

## Current-State Findings

The existing project has three important ML-specific assumptions:

1. Idea generation uses an ML-oriented system prompt.
   `ai_scientist/perform_ideation_temp_free.py` tells the model that proposals should lead to papers publishable at top ML conferences.

2. Default examples are deep-learning examples.
   The default workshop and ideas files are `i_cant_believe_its_not_better.*`, which are explicitly about applied deep learning failure modes.

3. Experiment prompt guidance assumes PyTorch/CUDA.
   `ParallelAgent._prompt_impl_guideline` tells generated code to use `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` and contains ML training guidance.

The launch entrypoint itself, `launch_scientist_bfts.py`, is mostly orchestration code. It reads an idea file, writes a per-run config, runs BFTS experiments, aggregates plots, writes a paper, reviews it, and cleans up. The first version should keep this entrypoint rather than replacing it.

## Goals

- Make the Core Research Loop domain-neutral.
- Preserve the existing end-to-end automated research workflow.
- Preserve the machine-learning workflow as a selectable Domain Pack.
- Support non-ML research problems without ML prompt contamination.
- Keep executable code as a core feature for all domains.
- Use local resource-aware execution by default for computational work.
- Make evidence strength explicit and carry it into writeup.
- Avoid old-schema compatibility layers and redundant project structure.
- Keep the first implementation small enough to land safely.

## Non-Goals

- Do not build a third-party plugin system.
- Do not implement remote or cloud GPU execution.
- Do not support many new scientific domains in the first version.
- Do not preserve every old ML-default behavior.
- Do not make literature search, code execution, plotting, writing, or review independently selectable packs in the first version.
- Do not allow the LLM to invent hard resource limit numbers.
- Do not support legacy idea JSON as a first-class input format.
- Do not add human-in-the-loop selection in the first version.

## Architecture

The first-version architecture has five concepts:

```text
Core Research Loop
  Owns the existing automated flow:
  ideation, literature search, tree search, code execution, plotting,
  writeup, review, and artifact persistence.

Research Profile Planner
  Automatically creates a Research Profile during ideation.

Research Profile
  A structured contract for the run:
  domain, execution backend, budget profile, evidence level,
  claim policy, rationale, and risk flags.

Domain Pack
  Static built-in domain guidance.
  First version: general and machine_learning.

Execution Backend + Budget Profile
  Local execution strategy and resource budget.
```

The Core Research Loop must not know ML-specific language. It may ask for domain guidance and execution guidance, but it should not hard-code any field-specific assumptions.

## Domain Packs

First version supports exactly two Domain Packs.

### `general`

The neutral default domain. It must use general research language:

- research question
- hypothesis
- evidence
- method
- limitation
- validation
- simulation
- statistical analysis
- reproducibility

It must not inject:

- top ML conference language
- benchmark / baseline / ablation as mandatory framing
- model training as the default research method
- dataset / dataloader assumptions
- PyTorch, CUDA, MPS, GPU guidance
- accuracy / loss as default metrics

`general` still allows executable code. Code may be used for simulation, statistics, numerical analysis, data processing, sanity checks, or other non-ML validation.

### `machine_learning`

The ML Domain Pack preserves the current ML-oriented workflow. It may include:

- ML conference writing expectations
- datasets, benchmarks, baselines, metrics, ablations
- model training and hyperparameter tuning
- multi-seed evaluation guidance
- PyTorch/CUDA device guidance, when the execution backend supports CUDA
- ML-specific result reporting

All existing PyTorch/CUDA/DataLoader/training-loop prompt guidance must move into this Domain Pack or an ML-specific prompt provider. It must not remain in Core.

### First-Version Domain Pack Interface

Domain Packs are static built-in configuration objects. They are not dynamic plugins.

Suggested shape:

```python
@dataclass
class DomainPack:
    id: str
    display_name: str
    idea_generation_guidance: str
    experiment_guidance: str
    writeup_guidance: str
    default_execution_backend_preference: str
    default_budget_profile_preference: str
```

Domain Packs must not:

- register arbitrary tools
- replace tree search
- override Core execution flow
- dynamically load third-party code
- own artifact layout

## Research Profile Planner

The Research Profile Planner runs during idea generation. It produces one Research Profile for the generated idea file.

### Allowed Inputs

The planner may use:

- user-provided research topic or workshop file
- explicit CLI/config overrides
- available Domain Packs
- local hardware detection
- available API keys or local tool availability

The planner must not use:

- bundled ML example files
- old generated idea JSON
- README defaults
- historical generated ideas
- existing ML sample code as evidence of the user's intended domain

### Default Behavior

Default domain selection is automatic:

```bash
--domain auto
```

The planner chooses between:

```text
general
machine_learning
```

Explicit user values override auto:

```bash
--domain general
--domain machine_learning
```

The first version remains fully automatic. Human-in-the-loop review can be added later, but automatic choices must record their rationale.

## Research Profile

Research Profile is the auditable contract between automatic selection and the Core Research Loop.

Suggested config shape:

```yaml
research_profile:
  schema_version: 1
  domain:
    id: general
    confidence: 0.78
    rationale: "The topic asks for a general empirical investigation and does not require ML model training."
  execution:
    backend: local_cpu_limited
    budget_profile: small
    evidence_level: limited_empirical
    rationale: "The research requires executable analysis, and no CUDA GPU was detected."
  claim_policy:
    allowed:
      - "preliminary empirical evidence"
      - "feasibility observations"
      - "resource-limited analysis"
    forbidden:
      - "full empirical validation"
      - "state-of-the-art performance claim"
      - "strong causal conclusion"
  risk_flags:
    - "domain selection may be wrong"
    - "resource-limited execution may understate or overstate effects"
```

Research Profile is stored inside the existing run config, not in a new artifact structure. The per-run `bfts_config.yaml` and stage `config.yaml` snapshots should carry it.

## Idea File Schema

The generalized pipeline accepts only the new idea schema.

Old top-level array format:

```json
[
  { "Name": "...", "Title": "..." }
]
```

New format:

```json
{
  "schema_version": 2,
  "research_profile": {
    "domain": { "id": "general" },
    "execution": {
      "backend": "local_cpu_limited",
      "budget_profile": "small",
      "evidence_level": "limited_empirical"
    },
    "claim_policy": {
      "allowed": [],
      "forbidden": []
    },
    "risk_flags": []
  },
  "ideas": [
    {
      "Name": "...",
      "Title": "...",
      "Short Hypothesis": "...",
      "Related Work": "...",
      "Abstract": "...",
      "Experiments": "...",
      "Risk Factors and Limitations": "..."
    }
  ]
}
```

Startup validation must reject invalid schema before any LLM calls or experiment execution.

## Workflow

The first version keeps the current two-step workflow.

### Step 1: Ideation

The user explicitly provides a research topic or workshop file. There is no bundled ML default.

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file path/to/topic.md \
  --domain auto \
  --execution-backend auto \
  --budget-profile auto
```

Ideation does:

1. Validate explicit research input.
2. Run Research Profile Planner.
3. Select `general` or `machine_learning`.
4. Build the idea-generation prompt from Core guidance plus Domain Pack guidance.
5. Generate ideas.
6. Write a versioned idea file envelope with Research Profile and ideas.

### Step 2: Experiment and Writeup

The run stage consumes the new idea file:

```bash
python launch_scientist_bfts.py \
  --load_ideas path/to/generated_ideas.json
```

Run stage does:

1. Validate the new idea schema before LLM or experiment calls.
2. Read the Research Profile.
3. Apply explicit CLI overrides, if any.
4. Write Research Profile into the per-run `bfts_config.yaml`.
5. Run the existing BFTS workflow with domain and execution guidance.
6. Pass Evidence Level and claim policy into writeup and review.

## CLI Changes

Both ideation and launch should accept the generalized options where relevant.

Suggested options:

```bash
--domain auto|general|machine_learning
--execution-backend auto|dry_run|smoke|local_cpu_limited|local_gpu_cuda_limited
--budget-profile auto|tiny|small|medium|full
```

The ML example files should not be default values for generalized commands. Generalized ideation requires an explicit `--workshop-file` or equivalent research input.

Existing `--load_code` and `--add_dataset_ref` may remain, but they should be treated as ML-oriented options. If used with `--domain general`, startup validation should either reject them or emit a clear pre-run error, because they inject ML/data-set assumptions.

## Execution Backends

First version supports local execution only.

### `dry_run`

Does not execute generated experimental code. It can support proposal writing, experiment planning, and feasibility reasoning. It cannot support empirical claims.

### `smoke`

Executes minimal code paths to verify basic feasibility. It can support implementation sanity-check claims. It cannot support performance or strong empirical claims.

### `local_cpu_limited`

Executes real local code with CPU-friendly constraints. It is the default backend for computational research when no CUDA GPU is available.

### `local_gpu_cuda_limited`

Executes real local code with CUDA available, still under a Budget Profile. It is not unrestricted full-scale execution unless paired with the `full` Budget Profile.

Remote/cloud execution is out of scope for the first version.

## Budget Profiles

Budget Profiles are static. The LLM may choose a tier, but it must not invent numeric hard limits.

Suggested first-version tiers:

```yaml
budget_profiles:
  tiny:
    num_workers: 1
    num_seeds: 1
    timeout_per_trial_minutes: 10
    max_stage_iterations: 2
    max_epochs_hint: 1
    max_dataset_samples_hint: 1000

  small:
    num_workers: 1
    num_seeds: 1
    timeout_per_trial_minutes: 30
    max_stage_iterations: 4
    max_epochs_hint: 3
    max_dataset_samples_hint: 5000

  medium:
    num_workers: 2
    num_seeds: 2
    timeout_per_trial_minutes: 60
    max_stage_iterations: 8
    max_epochs_hint: 5
    max_dataset_samples_hint: 20000

  full:
    num_workers: "current config or detected hardware"
    num_seeds: 3
    timeout_per_trial_minutes: "current config default"
    max_stage_iterations: "current config default"
    max_epochs_hint: null
    max_dataset_samples_hint: null
```

Budget Profile controls:

- `agent.num_workers`
- `agent.multi_seed_eval.num_seeds`
- `exec.timeout`
- stage iteration caps
- soft prompt guidance for epochs, data size, model size, and experiment scope

The first version does not implement adaptive runtime budget control. A future Adaptive Budget Controller may read CPU, memory, GPU, and process state to adjust budgets dynamically.

## Prompt Composition

Prompts should be composed from:

```text
Core prompt guidance
Domain Pack guidance
Execution Backend guidance
Budget Profile guidance
Evidence Level / claim policy guidance
```

Core prompt guidance may say:

- generate clear runnable code when code helps validate the hypothesis
- save structured result artifacts
- produce logs and summaries
- respect the selected budget
- do not overstate evidence

Core prompt guidance must not say:

- use PyTorch
- use CUDA
- train a model
- use a DataLoader
- evaluate accuracy/loss by default
- target top ML conferences

Machine-learning guidance may say those things only when `domain=machine_learning`.

## Writeup and Review

Writeup and review stages must receive Research Profile, Evidence Level, and claim policy.

Claim policy examples:

```text
dry_run:
  allowed: proposed method, expected validation plan
  forbidden: empirical improvement, measured performance

smoke:
  allowed: implementation feasibility, basic sanity check
  forbidden: performance conclusion, robustness claim

limited_empirical:
  allowed: preliminary evidence, resource-limited experiment
  forbidden: full validation, strong generalization claim

full_empirical:
  allowed: complete empirical conclusion within the actual experiment scope
  forbidden: claims beyond measured evidence
```

Generated papers must disclose limited evidence when appropriate.

## Artifact Placement

Do not create a new artifact tree for Research Profile.

Use existing locations:

```text
experiments/<run>/
  idea.md
  idea.json
  bfts_config.yaml        # includes research_profile
  logs/
    0-run/
      stage_<name>/
        config.yaml       # includes research_profile snapshot
        journal.json
        tree_plot.html
        notes/
```

This matches the existing artifact layout and avoids redundant persistence.

## Module Placement

New logic should not be added directly into `launch_scientist_bfts.py` beyond argument parsing and orchestration.

Suggested modules:

```text
ai_scientist/research_profile/
  __init__.py
  planner.py              # Research Profile Planner
  schema.py               # dataclasses / validation helpers
  domains.py              # general and machine_learning Domain Packs
  budgets.py              # static Budget Profiles
  execution_backends.py   # local backend metadata and config application
  prompting.py            # prompt composition helpers
```

Existing modules should call these helpers rather than duplicating profile logic.

## Validation Rules

Startup validation should happen before any expensive work.

Ideation validation:

- research input must be explicit
- domain value must be valid
- execution backend value must be valid
- budget profile value must be valid

Launch validation:

- idea file must be schema version 2
- idea file must contain `research_profile`
- selected idea index must exist inside `ideas`
- explicit overrides must be valid
- ML-only options should not be used with `domain=general`

Invalid inputs should fail fast before LLM calls, code execution, or citation gathering.

## Acceptance Criteria

### General Topic Smoke Test

Given a clearly non-ML research topic, ideation should:

- select `domain=general`
- produce prompts without ML conference, PyTorch, CUDA, dataset, training, or accuracy/loss defaults
- still generate a research idea with a concrete evidence plan

### ML Preservation Test

Given a clearly ML research topic, ideation should:

- select `domain=machine_learning`
- allow ML terminology and ML experiment guidance
- preserve tree search, code execution, plotting, writeup, review, budget, and multi-seed mechanisms

### Budget Enforcement Test

Given no CUDA and computational research, the run should:

- select or apply `local_cpu_limited`
- apply `tiny` or `small` constraints
- set `num_workers` and `num_seeds` according to Budget Profile
- reduce timeout and stage iteration caps
- produce limited-evidence claim policy

### Writeup Claim Test

Given `evidence_level=limited_empirical`, writeup should:

- describe results as preliminary or resource-limited
- not claim full validation
- not claim state-of-the-art performance without sufficient evidence

### Schema Validation Test

Given legacy top-level-array ideas JSON, launch should:

- fail during startup validation
- not call LLMs
- not run experiments
- explain that schema version 2 is required

## Implementation Slices

1. Add Research Profile schema and static Domain Packs.
2. Add static Budget Profiles and local Execution Backend metadata.
3. Add Research Profile Planner for `general` vs `machine_learning`.
4. Update ideation to require explicit research input and write schema version 2 idea files.
5. Update launch to validate schema version 2 and write Research Profile into run config.
6. Move ML prompt guidance out of Core prompts and into `machine_learning`.
7. Apply Budget Profile to existing config fields before BFTS execution.
8. Pass Evidence Level and claim policy into writeup/review prompts.
9. Add focused tests for general topic, ML topic, budget enforcement, and schema validation.

## Risks

- The Research Profile Planner may misclassify ambiguous topics.
- General prompts may become too vague if stripped of too much guidance.
- Limited CPU execution may produce weak or noisy evidence.
- Moving ML guidance out of Core may break current ML prompts if not covered by tests.
- Schema version changes will require updating examples and README instructions.

## Future Extensions

- Human-in-the-loop profile review.
- Adaptive Budget Controller based on live machine state.
- Limited Apple MPS execution backend for Apple Silicon Macs.
- Additional Domain Packs.
- External Capability Packs.
- Remote/cloud execution.
- One-command end-to-end workflow.
