# Move ML experiment guidance out of Core

PyTorch, CUDA, DataLoader, training-loop, benchmark, metric, ablation, and multi-seed experiment guidance must move out of the Core Research Loop and into the `machine_learning` Domain Pack. Core may keep only domain-neutral code execution and artifact requirements, while Execution Backends describe resource limits and available hardware.
