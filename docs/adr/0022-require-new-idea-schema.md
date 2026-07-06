# Require the new idea schema

The generalized pipeline should accept only the new idea schema that includes Research Profile metadata. The first version should not implement a compatibility layer for legacy bundled or user-generated idea JSON files; old examples should be migrated or removed from default flows, and invalid inputs should fail during startup validation before any LLM calls or experiment execution.
