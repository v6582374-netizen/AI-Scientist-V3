# Use a shared OpenAI-compatible provider resolver

OpenAI-compatible model providers should be resolved through one shared provider resolver used by high-level LLM calls, VLM helper calls, and the BFTS tree-search OpenAI backend. This avoids the current split where ideation/writeup can support a hard-coded provider while tree search or figure review silently falls back to the default OpenAI client; Alibaba Qwen is an initial built-in provider of this resolver rather than a separate calling stack.
