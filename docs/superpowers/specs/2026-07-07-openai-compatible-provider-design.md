# OpenAI-Compatible Provider Design Spec

Status: Accepted
Date: 2026-07-07
Related ADR: `docs/adr/0024-use-shared-openai-compatible-provider-resolver.md`

## Purpose

The project currently has partial OpenAI-compatible support through hard-coded branches in `ai_scientist/llm.py`, but this support is not general and does not consistently apply to the BFTS tree-search backend or the VLM helper path used during writeup/review. The goal is to support arbitrary OpenAI-compatible Chat Completions providers and add first-class Alibaba Qwen support without introducing another model-specific call stack.

## Decisions

- Add a shared provider resolver module used by `ai_scientist/llm.py`, `ai_scientist/vlm.py`, and `ai_scientist/treesearch/backend/backend_openai.py`.
- Support a generic model prefix: `openai-compatible/<model>`.
- Support Alibaba Qwen through a built-in prefix: `qwen/<model>`, for example `qwen/qwen-plus`.
- Treat Qwen as an OpenAI-Compatible Provider, not as a Domain Pack, Execution Backend, or provider-specific pipeline.

## Configuration

Generic provider:

```text
OPENAI_COMPATIBLE_API_KEY
OPENAI_COMPATIBLE_BASE_URL
```

Alibaba Qwen provider:

```text
DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL
```

If `DASHSCOPE_BASE_URL` is not set, default to:

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

Alibaba Cloud also supports region/workspace-specific compatible endpoints. Users who need those endpoints should set `DASHSCOPE_BASE_URL`; the default is only the built-in fallback for the common China/Beijing DashScope compatible endpoint.

The resolver must return both:

- an `openai.OpenAI` client configured with the provider credentials and base URL
- the provider-native model name with the local prefix stripped

## Examples

Ideation with Qwen:

```bash
export DASHSCOPE_API_KEY=...
python ai_scientist/perform_ideation_temp_free.py \
  --model qwen/qwen-plus \
  --workshop-file smoke_inputs/topic.md \
  --domain general \
  --execution-backend local_cpu_limited \
  --budget-profile tiny
```

Ideation with any OpenAI-compatible provider:

```bash
export OPENAI_COMPATIBLE_API_KEY=...
export OPENAI_COMPATIBLE_BASE_URL=https://provider.example.com/v1
python ai_scientist/perform_ideation_temp_free.py \
  --model openai-compatible/model-name \
  --workshop-file smoke_inputs/topic.md
```

BFTS config with Qwen:

```yaml
agent:
  code:
    model: qwen/qwen-plus
  feedback:
    model: qwen/qwen-plus
  vlm_feedback:
    model: qwen/qwen-plus
```

## Acceptance Criteria

- `qwen/qwen-plus` is accepted by user-facing model choices.
- `openai-compatible/<model>` is accepted even though the exact model is not enumerated.
- `create_client("qwen/qwen-plus")` constructs an OpenAI SDK client using DashScope credentials and returns model name `qwen-plus`.
- `create_client("openai-compatible/foo")` constructs an OpenAI SDK client using generic compatible credentials and returns model name `foo`.
- VLM client creation uses the same resolver, so writeup/review figure-description paths can use compatible multimodal models such as `qwen/qwen-vl-plus`.
- BFTS OpenAI backend uses the same resolver, so tree search can call Qwen or a generic provider.
- Existing OpenAI, Anthropic, Ollama, DeepSeek, OpenRouter, and Gemini behavior remains intact.
