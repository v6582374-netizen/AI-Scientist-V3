# Use gpt-5.5 as the shared default model

All model-bearing stages should use one shared Default Model when the user does not provide an explicit override. The shared default is `gpt-5.5`.

This replaces scattered defaults such as Bedrock Claude, GPT-4o, o1, and o3-mini. The previous BFTS experiment default could route code generation through Amazon Bedrock Claude even when the user had configured only OpenAI-compatible credentials, causing avoidable runtime failures. Stage-specific CLI flags and config overrides remain available for users who explicitly want Claude, Qwen, Ollama, or another OpenAI-compatible provider.
