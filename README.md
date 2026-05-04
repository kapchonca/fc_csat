# Controlled Banking Dialogue Generator

This project generates controlled synthetic banking support dialogues from
declarative case specs. Tools are not executed; they are labels used for trace
construction, prompting, and validation.

## Local Mock Run

```bash
python3 -m src.runner run-all
```

The default `configs/generation_config.json` uses the deterministic mock
renderer and writes outputs to `outputs/`.

## OpenAI Run

```bash
export OPENAI_API_KEY="..."
python3 -m src.runner run-all \
  --generation-config configs/generation_config.openai.example.json \
  --output-dir outputs_openai
```

The OpenAI config defaults to:

- `renderer`: `openai`
- endpoint: `https://api.openai.com/v1/chat/completions`
- key env var: `OPENAI_API_KEY`
- model: `gpt-5-mini`
- output limit field: `max_completion_tokens`
- reasoning field: `reasoning_effort`

You can override the endpoint with `api_endpoint` or `api_base_url`.

For a small smoke test, cap the number of API calls:

```bash
python3 -m src.runner run-all \
  --generation-config configs/generation_config.openai.example.json \
  --output-dir outputs_openai_smoke \
  --limit-dialogues 2
```

For a full debug pass with one variant per included case:

```bash
python3 -m src.runner run-all \
  --generation-config configs/generation_config.openai.debug.json \
  --output-dir outputs_openai_debug
```

## OpenRouter Run

```bash
export OPENROUTER_API_KEY="..."
python3 -m src.runner run-all \
  --generation-config configs/generation_config.openrouter.example.json \
  --output-dir outputs_openrouter
```

The OpenRouter config defaults to:

- `renderer`: `openrouter`
- endpoint: `https://openrouter.ai/api/v1/chat/completions`
- key env var: `OPENROUTER_API_KEY`
- model: `openai/gpt-5-mini`
- output limit field: `max_completion_tokens`
- reasoning field: `extra_payload.reasoning`
- optional attribution headers: `http_referer`, `app_title`

For a small smoke test:

```bash
python3 -m src.runner run-all \
  --generation-config configs/generation_config.openrouter.example.json \
  --output-dir outputs_openrouter_smoke \
  --limit-dialogues 2
```

For a full debug pass with one variant per included case:

```bash
python3 -m src.runner run-all \
  --generation-config configs/generation_config.openrouter.debug.json \
  --output-dir outputs_openrouter_debug
```

## Outputs

Each run writes:

- `case_specs.json`
- `dialogues.json`
- `debug.jsonl`
- `run_summary.json`

Raw model outputs are always preserved in `debug.jsonl`, including failed
validator results.
