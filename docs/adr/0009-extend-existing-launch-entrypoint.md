# Extend the existing launch entrypoint

The first generalized implementation should keep `launch_scientist_bfts.py` as the user-facing launch entrypoint and add generalized options there, rather than introducing a separate first-class entrypoint. New planning, domain, budget, and execution profile logic should live in dedicated modules so the entrypoint remains orchestration code and does not accumulate the new architecture's implementation details.
