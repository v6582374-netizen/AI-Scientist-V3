# Limit Research Profile Planner inputs

The Research Profile Planner may use the user's research problem, explicit configuration, installed Domain Packs, local Execution Backends, static Budget Profiles, detected machine resources, and available tool credentials when building a research profile. It must not use bundled example ideas, historical generated idea JSON, README defaults, or ML sample files as implicit evidence for the user's research domain, because those artifacts would reintroduce domain bias into automatic selection.
