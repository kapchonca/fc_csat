from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


DEFAULT_VALIDATION_RULES: dict[str, Any] = {
    "success_terms": [
        "cancelled",
        "cancellation is confirmed",
        "completed",
        "done",
        "successful",
        "confirmed",
    ],
    "failure_terms": [
        "unable",
        "cannot",
        "can't",
        "couldn't",
        "failed",
        "needs to be escalated",
        "need to escalate",
        "not completed",
    ],
    "internal_forbidden_tokens": [
        "case_id",
        "trace",
        "tool",
        "condition",
        "wrong_parameter",
        "missing_input",
        "skip_step",
        "wrong_order",
        "wrong_tool",
        "task_completed",
        "task_failed",
    ],
    "human_condition_labels": [
        "case id",
        "hidden trace",
        "wrong parameter",
        "missing input",
        "skip step",
        "wrong order",
        "wrong tool",
        "task completed",
        "task failed",
    ],
    "condition_indicators": {
        "correct": [],
        "skip_step": ["without checking", "missed"],
        "extra_step": ["recent transactions", "first"],
        "wrong_order": ["before confirming", "backwards"],
        "wrong_tool": ["recent transactions", "instead of"],
        "wrong_parameter": ["only the amount", "not resolved"],
    },
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
    final_assistant = _final_assistant_message(messages)

    _check_internal_labels(case_spec, text, rules, errors)
    if rules.get("forbid_tool_names", True) and tool_catalog is not None:
        _check_tool_names(text, tool_catalog, errors)
    _check_unexpected_condition_labels(text, rules, errors)
    _check_condition_match(case_spec, text, rules, errors)
    _check_expected_outcome(case_spec, final_assistant, rules, errors)

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


def _check_internal_labels(
    case_spec: dict[str, Any],
    text: str,
    rules: dict[str, Any],
    errors: list[str],
) -> None:
    tokens = list(rules["internal_forbidden_tokens"])
    tokens.extend([case_spec["case_id"], case_spec["condition"]])
    tokens.extend(case_spec.get("labels", []))
    error = case_spec.get("error")
    if error:
        tokens.append(error["type"])
    if any(_contains_token(text, token) for token in tokens):
        _add_error(errors, "internal_label_leak")


def _check_tool_names(
    text: str,
    tool_catalog: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if any(_contains_token(text, tool["id"]) for tool in tool_catalog):
        _add_error(errors, "tool_name_leak")


def _check_unexpected_condition_labels(
    text: str,
    rules: dict[str, Any],
    errors: list[str],
) -> None:
    if any(_contains_token(text, label) for label in rules["human_condition_labels"]):
        _add_error(errors, "unexpected_condition_label")


def _check_condition_match(
    case_spec: dict[str, Any],
    text: str,
    rules: dict[str, Any],
    errors: list[str],
) -> None:
    indicators = rules["condition_indicators"].get(case_spec["condition"], [])
    if indicators and not any(indicator.lower() in text.lower() for indicator in indicators):
        _add_error(errors, "condition_mismatch")


def _check_expected_outcome(
    case_spec: dict[str, Any],
    final_assistant: str,
    rules: dict[str, Any],
    errors: list[str],
) -> None:
    final_text = final_assistant.lower()
    success = any(term.lower() in final_text for term in rules["success_terms"])
    failure = any(term.lower() in final_text for term in rules["failure_terms"])

    if case_spec["expected_outcome"] == "task_completed":
        if not success or failure:
            _add_error(errors, "wrong_outcome")
    elif case_spec["expected_outcome"] == "task_failed":
        if not failure:
            _add_error(errors, "wrong_outcome")
    else:
        _add_error(errors, "wrong_outcome")


def _final_assistant_message(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


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
