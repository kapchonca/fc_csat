from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


REQUIRED_CONFIG_FILES = {
    "tool_catalog": ("tools.json", "tool_catalog.json"),
    "action_graph": ("graph.json", "action_graph_product_support.json"),
    "case_templates": ("cases.json", "case_templates.json"),
    "generation_config": ("generation.json", "generation_config.json"),
}


def read_json(path: str | Path) -> Any:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing config file: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc


def load_configs(
    config_dir: str | Path = "configs",
    generation_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(config_dir)
    configs = {
        key: read_json(
            Path(generation_config_path)
            if key == "generation_config" and generation_config_path is not None
            else _resolve_config_path(root, filenames)
        )
        for key, filenames in REQUIRED_CONFIG_FILES.items()
    }

    validate_tool_catalog(configs["tool_catalog"])
    validate_action_graph(configs["action_graph"])
    validate_case_templates(configs["case_templates"])
    validate_generation_config(configs["generation_config"])
    return configs


def _resolve_config_path(root: Path, filenames: tuple[str, ...]) -> Path:
    for filename in filenames:
        path = root / filename
        if path.exists():
            return path
    return root / filenames[0]


def validate_tool_catalog(tool_catalog: Any) -> None:
    if not isinstance(tool_catalog, list):
        raise ConfigError("tool_catalog.json must contain a JSON array.")

    seen: set[str] = set()
    for index, tool in enumerate(tool_catalog):
        if not isinstance(tool, dict):
            raise ConfigError(f"Tool at index {index} must be an object.")
        for key in ("id", "description", "inputs", "adds", "errors"):
            if key not in tool:
                raise ConfigError(f"Tool at index {index} is missing {key!r}.")
        if not isinstance(tool["id"], str) or not tool["id"]:
            raise ConfigError(f"Tool at index {index} has an invalid id.")
        if tool["id"] in seen:
            raise ConfigError(f"Duplicate tool id: {tool['id']}")
        seen.add(tool["id"])
        for list_key in ("inputs", "adds", "errors"):
            if not isinstance(tool[list_key], list) or not all(
                isinstance(item, str) and item for item in tool[list_key]
            ):
                raise ConfigError(f"Tool {tool['id']} has invalid {list_key}.")


def validate_action_graph(action_graph: Any) -> None:
    if not isinstance(action_graph, dict):
        raise ConfigError("action_graph.json must contain an object.")
    if not isinstance(action_graph.get("nodes"), list):
        raise ConfigError("action_graph.json must contain a nodes array.")

    nodes = action_graph["nodes"]
    if not all(isinstance(node, str) and node for node in nodes):
        raise ConfigError("All action graph nodes must be non-empty strings.")
    if len(nodes) != len(set(nodes)):
        raise ConfigError("Action graph nodes must be unique.")

    for key in (
        "dependency_edges",
        "hard_precondition_edges",
        "recovery_edges",
        "confusion_edges",
        "extra_step_candidates",
    ):
        edges = action_graph.get(key, [])
        if not isinstance(edges, list):
            raise ConfigError(f"action_graph.{key} must be an array.")
        for edge in edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or not all(isinstance(item, str) and item for item in edge)
            ):
                raise ConfigError(f"Invalid edge in action_graph.{key}: {edge!r}")


def validate_case_templates(case_templates: Any) -> None:
    if not isinstance(case_templates, dict):
        raise ConfigError("case_templates.json must contain an object.")
    if not isinstance(case_templates.get("version"), str):
        raise ConfigError("case_templates.json must contain a string version.")
    templates = case_templates.get("templates")
    if not isinstance(templates, list):
        raise ConfigError("case_templates.json must contain a templates array.")

    for index, template in enumerate(templates):
        if not isinstance(template, dict):
            raise ConfigError(f"Template at index {index} must be an object.")
        for key in ("task", "base_trace", "conditions"):
            if key not in template:
                raise ConfigError(f"Template at index {index} is missing {key!r}.")
        if not isinstance(template["task"], str) or not template["task"]:
            raise ConfigError(f"Template at index {index} has an invalid task.")
        for key in ("base_trace", "conditions"):
            if not isinstance(template[key], list) or not all(
                isinstance(item, str) and item for item in template[key]
            ):
                raise ConfigError(f"Template {template['task']} has invalid {key}.")


def validate_generation_config(generation_config: Any) -> None:
    if not isinstance(generation_config, dict):
        raise ConfigError("generation_config.json must contain an object.")

    variants = generation_config.get("variants_per_case")
    if not isinstance(variants, int) or variants < 1:
        raise ConfigError("variants_per_case must be a positive integer.")

    max_parallel_requests = generation_config.get("max_parallel_requests", 1)
    if not isinstance(max_parallel_requests, int) or max_parallel_requests < 1:
        raise ConfigError("max_parallel_requests must be a positive integer.")

    plan_config = generation_config.get("dialogue_plan")
    if plan_config is not None:
        validate_dialogue_plan_config(plan_config)
    if not isinstance(plan_config, dict) or not plan_config.get("enabled", False):
        for key in ("output_message_min", "output_message_max"):
            value = generation_config.get(key)
            if not isinstance(value, int) or value < 1:
                raise ConfigError(f"{key} must be a positive integer.")
        if generation_config["output_message_min"] > generation_config["output_message_max"]:
            raise ConfigError("output_message_min cannot exceed output_message_max.")

    include_cases = generation_config.get("include_cases", [])
    if not isinstance(include_cases, list) or not all(
        isinstance(case_id, str) and case_id for case_id in include_cases
    ):
        raise ConfigError("include_cases must be an array of case ids.")

    validate_semantic_variants(generation_config.get("semantic_variants", []))

    if not isinstance(generation_config.get("model"), str):
        raise ConfigError("model must be a string.")

    pricing = generation_config.get("pricing")
    if pricing is not None:
        validate_pricing_config(pricing)


def validate_dialogue_plan_config(plan_config: Any) -> None:
    if not isinstance(plan_config, dict):
        raise ConfigError("dialogue_plan must be an object.")

    enabled = plan_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("dialogue_plan.enabled must be a boolean.")
    if not enabled:
        return

    message_count = plan_config.get("message_count")
    if not isinstance(message_count, int) or message_count < 1:
        raise ConfigError("dialogue_plan.message_count must be a positive integer.")

    role_pattern = plan_config.get("role_pattern")
    if not isinstance(role_pattern, list) or not all(
        role in {"user", "assistant"} for role in role_pattern
    ):
        raise ConfigError("dialogue_plan.role_pattern must contain user/assistant roles.")
    if len(role_pattern) != message_count:
        raise ConfigError("dialogue_plan.role_pattern length must equal message_count.")


def validate_pricing_config(pricing: Any) -> None:
    if not isinstance(pricing, dict):
        raise ConfigError("pricing must be an object.")
    if "currency" in pricing and not isinstance(pricing["currency"], str):
        raise ConfigError("pricing.currency must be a string.")
    for key in (
        "input_per_1m_tokens",
        "cached_input_per_1m_tokens",
        "output_per_1m_tokens",
    ):
        if key not in pricing:
            continue
        value = pricing[key]
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"pricing.{key} must be a non-negative number.")


def validate_semantic_variants(semantic_variants: Any) -> None:
    if semantic_variants is None:
        return
    if not isinstance(semantic_variants, list):
        raise ConfigError("semantic_variants must be an array.")

    seen: set[str] = set()
    for index, variant in enumerate(semantic_variants):
        if not isinstance(variant, dict):
            raise ConfigError(f"semantic_variants[{index}] must be an object.")
        variant_id = variant.get("id")
        instruction = variant.get("instruction")
        if not isinstance(variant_id, str) or not variant_id:
            raise ConfigError(f"semantic_variants[{index}].id must be a non-empty string.")
        if not variant_id.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(
                f"semantic_variants[{index}].id must use letters, digits, hyphens, or underscores."
            )
        if variant_id in seen:
            raise ConfigError(f"Duplicate semantic variant id: {variant_id}")
        seen.add(variant_id)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ConfigError(
                f"semantic_variants[{index}].instruction must be a non-empty string."
            )
