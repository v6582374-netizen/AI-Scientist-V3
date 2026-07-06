# Budget Profile controls parallelism

The system keeps parallel tree search as a Core Research Loop capability, but actual worker count is controlled by the selected Budget Profile and detected hardware. Low-budget runs may execute serially while preserving the tree-search strategy; higher-budget runs may expand and execute multiple nodes in parallel.
