# Store Research Profile in existing config artifacts

Research Profile should be stored as part of the run configuration, primarily in the per-run `bfts_config.yaml` and the existing stage `config.yaml` snapshots written by `save_run`. The first implementation should avoid adding a new artifact directory or parallel persistence structure for the profile.
