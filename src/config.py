from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


REQUIRED_CONFIG_FILES = {
    "tool_catalog": "tool_catalog.json",
    "action_graph": "action_graph.json",
    "case_templates": "case_templates.json",
    "generation_config": "generation_config.json",
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


def load_configs(config_dir: str | Path = "configs") -> dict[str, Any]:
    root = Path(config_dir)
    configs = {
        key: read_json(root / filename)
        for key, filename in REQUIRED_CONFIG_FILES.items()
    }

    validate_tool_catalog(configs["tool_catalog"])
    validate_action_graph(configs["action_graph"])
    validate_case_templates(configs["case_templates"])
    validate_generation_config(configs["generation_config"])
    return configs


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

    if not isinstance(generation_config.get("model"), str):
        raise ConfigError("model must be a string.")
