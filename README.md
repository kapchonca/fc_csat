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

You can override the endpoint with `api_endpoint` or `api_base_url`.

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
- optional attribution headers: `http_referer`, `app_title`

## Outputs

Each run writes:

- `case_specs.json`
- `dialogues.json`
- `debug.jsonl`
- `run_summary.json`

Raw model outputs are always preserved in `debug.jsonl`, including failed
validator results.
