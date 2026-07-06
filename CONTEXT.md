# AI Scientist Generalization Context

This context defines the language for generalizing AI Scientist from an ML-specific system into a domain-neutral research system with pluggable domain assumptions, capabilities, and execution strategies.

## Language

**Core Research Loop**:
The domain-neutral orchestration layer for research flow: problem intake, planning, tree search, code execution, evidence tracking, result aggregation, and writeup coordination. It must not embed assumptions from any particular research field.
_Avoid_: AI Scientist core, main pipeline, engine

**Domain Pack**:
A pluggable package of field-specific assumptions, vocabulary, prompt guidance, evidence standards, and optional defaults for a research domain or research paradigm. Machine learning is one Domain Pack, not the default worldview of the system.
_Avoid_: domain plugin, field module, template

**Core Capability**:
A general research ability that remains part of the default Core Research Loop in the first generalized design, such as literature search, code execution, plotting, writing, and review. Core Capabilities are not independently selected by the Research Profile Planner in the first implementation.
_Avoid_: capability pack, tool, utility

**Capability Pack**:
A future extension form for making a Core Capability replaceable or externally installable. Capability Packs are not a first-version implementation target for the generalization effort.
_Avoid_: tool plugin, utility plugin

**Execution Backend**:
A pluggable strategy for carrying out computational work at a specific resource level, such as dry run, smoke test, limited local CPU, or limited local CUDA GPU. It determines how evidence is produced, not what domain the research belongs to.
_Avoid_: runner, executor, GPU mode

**Research Profile Planner**:
The selector that derives a research profile from the user's research problem, available Domain Packs, local execution options, and explicit configuration. In the first generalized design it operates automatically, while preserving enough recorded rationale to support a future human-in-the-loop review step.
_Avoid_: selector, router, classifier

**Research Profile**:
The structured plan chosen for a run: selected Domain Pack, Execution Backend, evidence level, allowed claim strength, forbidden claim types, rationales, and risk flags. It is the auditable contract between automatic selection and the Core Research Loop.
_Avoid_: configuration, domain label, run settings

**Evidence Level**:
The declared strength of the evidence produced by a run, such as dry run, smoke test, limited empirical validation, or full empirical validation. It determines what claims the system may make in generated analysis and writing.
_Avoid_: confidence, validation mode, result quality

**Budget Profile**:
A static resource budget tier that maps to hard execution limits and soft experiment-size guidance, such as tiny, small, medium, or full. The Research Profile Planner may choose a Budget Profile, but it does not invent numeric hard limits in the first implementation.
_Avoid_: resource preset, hardware profile, runtime settings
