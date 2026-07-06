# Default to automatic domain selection

The generalized entrypoint should default to `--domain auto`, allowing the Research Profile Planner to choose between `general` and `machine_learning` from the user's research problem and allowed inputs. Explicit CLI configuration such as `--domain general` or `--domain machine_learning` must override automatic selection.
