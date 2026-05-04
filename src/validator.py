from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


DEFAULT_VALIDATION_RULES: dict[str, Any] = {
    "forbid_tool_names": True,
}


@dataclass(frozen=True)
class ValidationResult:
    case_id: str
    variant_id: int
    status: str
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "variant_id": self.variant_id,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_dialogue(
    case_spec: dict[str, Any],
    messages: list[Any] | None,
    generation_config: dict[str, Any],
    tool_catalog: list[dict[str, Any]] | None = None,
    variant_id: int = 0,
    parse_errors: list[str] | None = None,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    parse_errors = parse_errors or []
    rules = _validation_rules(generation_config)

    for error in parse_errors:
        _add_error(errors, error)

    if messages is None:
        return _result(case_spec["case_id"], variant_id, errors or ["invalid_json"], warnings)

    _check_roles_and_content(messages, errors)
    _check_message_count(messages, generation_config, errors)

    text = _dialogue_text(messages)
    if rules.get("forbid_tool_names", True) and tool_catalog is not None:
        _check_tool_names(text, tool_catalog, errors)

    return _result(case_spec["case_id"], variant_id, errors, warnings)


def _check_roles_and_content(messages: list[Any], errors: list[str]) -> None:
    for message in messages:
        if not isinstance(message, dict):
            _add_error(errors, "invalid_message")
            continue
        if message.get("role") not in {"user", "assistant"}:
            _add_error(errors, "wrong_role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            _add_error(errors, "empty_message")


def _check_message_count(
    messages: list[Any],
    generation_config: dict[str, Any],
    errors: list[str],
) -> None:
    message_min = generation_config["output_message_min"]
    message_max = generation_config["output_message_max"]
    if not message_min <= len(messages) <= message_max:
        _add_error(errors, "message_count_out_of_range")


def _check_tool_names(
    text: str,
    tool_catalog: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if any(_contains_token(text, tool["id"]) for tool in tool_catalog):
        _add_error(errors, "tool_name_leak")


def _dialogue_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            parts.append(message["content"])
    return "\n".join(parts)


def _contains_token(text: str, token: str) -> bool:
    if not token:
        return False
    text_lower = text.lower()
    token_lower = token.lower()
    if "_" in token_lower:
        return token_lower in text_lower
    pattern = r"(?<![a-z0-9_])" + re.escape(token_lower) + r"(?![a-z0-9_])"
    return re.search(pattern, text_lower) is not None


def _validation_rules(generation_config: dict[str, Any]) -> dict[str, Any]:
    rules = dict(DEFAULT_VALIDATION_RULES)
    overrides = generation_config.get("validation", {})
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(rules.get(key), dict):
            nested = dict(rules[key])
            nested.update(value)
            rules[key] = nested
        else:
            rules[key] = value
    return rules


def _result(
    case_id: str,
    variant_id: int,
    errors: list[str],
    warnings: list[str],
) -> ValidationResult:
    return ValidationResult(
        case_id=case_id,
        variant_id=variant_id,
        status="failed" if errors else "passed",
        errors=errors,
        warnings=warnings,
    )


def _add_error(errors: list[str], code: str) -> None:
    if code not in errors:
        errors.append(code)
