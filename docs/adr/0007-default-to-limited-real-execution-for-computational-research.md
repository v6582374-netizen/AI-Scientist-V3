# Default to limited real execution for computational research

Automatic execution backend selection should preserve the ability to produce real evidence by default. For computational research without CUDA, the planner should choose limited local CPU execution rather than dry run; for computational research with CUDA, it should choose limited local CUDA execution. Dry run remains appropriate for non-computational research or explicit user selection, while limited execution must constrain workers, seeds, timeouts, model size, dataset size, and claim strength.
