# Avoid over-compatibility with old ML defaults

The generalized architecture should not add complex compatibility branches to preserve every old ML-default behavior. Existing ML workflows remain available through explicit `machine_learning` selection, but the default experience should prioritize the new automatic generalized behavior to reduce complexity and user mental overhead.
