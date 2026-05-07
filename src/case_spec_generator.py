from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CaseSpecError(ValueError):
    """Raised when case specs cannot be generated or validated."""


EXPECTED_OUTCOME_COMPLETED = "task_completed"
EXPECTED_OUTCOME_FAILED = "task_failed"


def generate_case_specs(
    tool_catalog: list[dict[str, Any]],
    action_graph: dict[str, Any],
    case_templates: dict[str, Any],
) -> list[dict[str, Any]]:
    tools = _tool_map(tool_catalog)
    graph = _normalise_graph(action_graph)
    specs: list[dict[str, Any]] = []

    for template in case_templates["templates"]:
        base_trace = list(template["base_trace"])
        _validate_base_trace(template["task"], base_trace, tools, graph["nodes"])
        context = _template_context(template, base_trace, tools, graph["nodes"])

        for condition in template["conditions"]:
            spec = _generate_condition_spec(
                task=template["task"],
                base_trace=base_trace,
                condition=condition,
                tools=tools,
                graph=graph,
                context=context,
            )
            specs.append(spec)

    validate_case_specs(specs, tool_catalog, action_graph)
    return specs


def save_case_specs(specs: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(specs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_case_specs(
    specs: list[dict[str, Any]],
    tool_catalog: list[dict[str, Any]],
    action_graph: dict[str, Any],
) -> None:
    tools = _tool_map(tool_catalog)
    graph = _normalise_graph(action_graph)
    ids = [spec.get("case_id") for spec in specs]
    if len(ids) != len(set(ids)):
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        raise CaseSpecError(f"Case ids must be unique: {duplicates}")

    for spec in specs:
        _validate_spec_shape(spec)
        for action in spec["trace"]:
            if _is_error_marker(action):
                error_type, tool_id = _split_error_marker(action)
                if tool_id not in tools:
                    raise CaseSpecError(
                        f"{spec['case_id']} references unknown tool {tool_id!r}."
                    )
                if error_type == "skip_step":
                    continue
                if error_type not in tools[tool_id]["errors"]:
                    raise CaseSpecError(
                        f"{spec['case_id']} uses unsupported error "
                        f"{error_type!r} for {tool_id!r}."
                    )
            elif action not in tools and action not in graph["nodes"]:
                raise CaseSpecError(
                    f"{spec['case_id']} references unknown action {action!r}."
                )

        error = spec.get("error")
        if error and error["type"] == "wrong_tool":
            edge = (error["at"], error["replacement"])
            if edge not in graph["confusion_edges"]:
                raise CaseSpecError(
                    f"{spec['case_id']} has a wrong_tool replacement not present "
                    "in confusion_edges."
                )

def _generate_condition_spec(
    task: str,
    base_trace: list[str],
    condition: str,
    tools: dict[str, dict[str, Any]],
    graph: dict[str, Any],
    context: dict[str, str],
) -> dict[str, Any]:
    if condition == "correct":
        return _spec(
            task=task,
            condition=condition,
            parts=[],
            error=None,
            trace=base_trace,
            expected_outcome=EXPECTED_OUTCOME_COMPLETED,
            labels=["correct", EXPECTED_OUTCOME_COMPLETED],
            context=context,
        )

    if condition == "skip_step":
        skipped = _choose_target_or_skip_action(
            context,
            base_trace,
            graph["hard_precondition_edges"],
        )
        trace = _insert_error_before(base_trace, "skip_step", skipped)
        return _spec(
            task=task,
            condition=condition,
            parts=[skipped],
            error={"type": "skip_step", "at": skipped, "recovered": True},
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_COMPLETED,
            labels=["skip_step", "recovered", EXPECTED_OUTCOME_COMPLETED],
            context=context,
        )

    if condition == "skip_step_not_recovered":
        skipped = _choose_target_or_skip_action(
            context,
            base_trace,
            graph["hard_precondition_edges"],
        )
        trace = _replace_with_error_marker(base_trace, "skip_step", skipped)
        return _spec(
            task=task,
            condition=condition,
            parts=[skipped],
            error={"type": "skip_step", "at": skipped, "recovered": False},
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_FAILED,
            labels=_not_recovered_labels("skip_step", context),
            context=context,
        )

    if condition == "extra_step":
        anchor, extra = _choose_extra_action(
            base_trace,
            tools,
            graph["extra_step_candidates"],
        )
        trace = _insert_after(base_trace, anchor, extra)
        return _spec(
            task=task,
            condition=condition,
            parts=[extra],
            error={"type": "extra_step", "at": extra, "anchor": anchor},
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_COMPLETED,
            labels=["extra_step", EXPECTED_OUTCOME_COMPLETED],
            context=context,
        )

    if condition == "wrong_order":
        before, after = _choose_order_pair(
            context,
            base_trace,
            graph["hard_precondition_edges"],
        )
        trace = list(base_trace)
        before_index = trace.index(before)
        after_index = trace.index(after)
        trace[before_index], trace[after_index] = trace[after_index], trace[before_index]
        trace = _insert_after(trace, before, after)
        return _spec(
            task=task,
            condition=condition,
            parts=[before, after],
            error={
                "type": "wrong_order",
                "at": after,
                "required_before": before,
                "recovered": True,
            },
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_COMPLETED,
            labels=["wrong_order", "recovered", EXPECTED_OUTCOME_COMPLETED],
            context=context,
        )

    if condition == "wrong_order_not_recovered":
        before, after = _choose_order_pair(
            context,
            base_trace,
            graph["hard_precondition_edges"],
        )
        trace = list(base_trace)
        before_index = trace.index(before)
        after_index = trace.index(after)
        trace[before_index], trace[after_index] = trace[after_index], trace[before_index]
        return _spec(
            task=task,
            condition=condition,
            parts=[before, after],
            error={
                "type": "wrong_order",
                "at": after,
                "required_before": before,
                "recovered": False,
            },
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_FAILED,
            labels=_not_recovered_labels("wrong_order", context),
            context=context,
        )

    if condition == "wrong_tool":
        expected, replacement = _choose_confused_action(
            base_trace,
            graph["confusion_edges"],
            tools,
            graph["nodes"],
            preferred_action=context.get("target_action"),
        )
        trace = _insert_before(base_trace, expected, replacement)
        return _spec(
            task=task,
            condition=condition,
            parts=[expected],
            error={
                "type": "wrong_tool",
                "at": expected,
                "replacement": replacement,
                "recovered": True,
            },
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_COMPLETED,
            labels=["wrong_tool", "recovered", EXPECTED_OUTCOME_COMPLETED],
            context=context,
        )

    if condition == "wrong_tool_not_recovered":
        expected, replacement = _choose_confused_action(
            base_trace,
            graph["confusion_edges"],
            tools,
            graph["nodes"],
            preferred_action=context.get("target_action"),
        )
        trace = _replace_action(base_trace, expected, replacement)
        return _spec(
            task=task,
            condition=condition,
            parts=[expected],
            error={
                "type": "wrong_tool",
                "at": expected,
                "replacement": replacement,
                "recovered": False,
            },
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_FAILED,
            labels=_not_recovered_labels("wrong_tool", context),
            context=context,
        )

    if condition == "wrong_parameter":
        tool_id = _choose_parameter_error_tool(
            context,
            "wrong_parameter",
            base_trace,
            tools,
        )
        trace = _insert_error_before(base_trace, "wrong_parameter", tool_id)
        return _spec(
            task=task,
            condition=condition,
            parts=[tool_id],
            error={"type": "wrong_parameter", "at": tool_id, "recovered": True},
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_COMPLETED,
            labels=["wrong_parameter", "recovered", EXPECTED_OUTCOME_COMPLETED],
            context=context,
        )

    if condition == "wrong_parameter_not_recovered":
        tool_id = _choose_parameter_error_tool(
            context,
            "wrong_parameter",
            base_trace,
            tools,
        )
        trace = _wrong_parameter_not_recovered_trace(base_trace, tool_id, context)
        return _spec(
            task=task,
            condition=condition,
            parts=[tool_id],
            error={"type": "wrong_parameter", "at": tool_id, "recovered": False},
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_FAILED,
            labels=_not_recovered_labels("wrong_parameter", context),
            context=context,
        )

    raise CaseSpecError(f"Unsupported condition: {condition}")


def _spec(
    task: str,
    condition: str,
    parts: list[str],
    error: dict[str, Any] | None,
    trace: list[str],
    expected_outcome: str,
    labels: list[str],
    context: dict[str, str],
) -> dict[str, Any]:
    suffix = "_".join([condition, *parts]) if parts else condition
    spec = {
        "case_id": f"{task}_{suffix}",
        "task": task,
        "condition": condition,
        "error": error,
        "trace": list(trace),
        "expected_outcome": expected_outcome,
        "labels": labels,
    }
    for key in ("user_goal", "target_action", "downstream_action"):
        if context.get(key):
            spec[key] = context[key]
    return spec


def _choose_skip_action(
    base_trace: list[str],
    hard_precondition_edges: set[tuple[str, str]],
) -> str:
    candidates = [action for action in base_trace[1:-1] if action != "done"]
    for required, dependent in hard_precondition_edges:
        if required in candidates and dependent in base_trace:
            return required
    if not candidates:
        raise CaseSpecError("skip_step requires an intermediate action.")
    return candidates[0]


def _choose_target_or_skip_action(
    context: dict[str, str],
    base_trace: list[str],
    hard_precondition_edges: list[tuple[str, str]],
) -> str:
    target_action = context.get("target_action")
    if target_action and target_action in base_trace[1:-1]:
        return target_action
    return _choose_skip_action(base_trace, hard_precondition_edges)


def _choose_extra_action(
    base_trace: list[str],
    tools: dict[str, dict[str, Any]],
    extra_step_candidates: list[tuple[str, str]],
) -> tuple[str, str]:
    for anchor in base_trace:
        for candidate_anchor, extra in extra_step_candidates:
            if candidate_anchor == anchor and extra not in base_trace and extra in tools:
                return anchor, extra
    raise CaseSpecError(
        "extra_step requires an extra_step_candidates edge anchored in the base trace."
    )


def _insert_after(base_trace: list[str], anchor: str, action: str) -> list[str]:
    trace: list[str] = []
    inserted = False
    for item in base_trace:
        trace.append(item)
        if item == anchor and not inserted:
            trace.append(action)
            inserted = True
    if not inserted:
        raise CaseSpecError(f"Cannot insert {action!r}; anchor {anchor!r} is absent.")
    return trace


def _insert_before(base_trace: list[str], anchor: str, action: str) -> list[str]:
    trace: list[str] = []
    inserted = False
    for item in base_trace:
        if item == anchor and not inserted:
            trace.append(action)
            inserted = True
        trace.append(item)
    if not inserted:
        raise CaseSpecError(f"Cannot insert {action!r}; anchor {anchor!r} is absent.")
    return trace


def _replace_action(base_trace: list[str], expected: str, replacement: str) -> list[str]:
    trace = list(base_trace)
    try:
        index = trace.index(expected)
    except ValueError as exc:
        raise CaseSpecError(
            f"Cannot replace {expected!r}; it is absent from the base trace."
        ) from exc
    trace[index] = replacement
    return trace


def _choose_hard_precondition(
    base_trace: list[str],
    hard_precondition_edges: set[tuple[str, str]],
) -> tuple[str, str]:
    for before, after in hard_precondition_edges:
        if before in base_trace and after in base_trace:
            return before, after
    raise CaseSpecError("wrong_order requires a hard_precondition_edge in the base trace.")


def _choose_order_pair(
    context: dict[str, str],
    base_trace: list[str],
    hard_precondition_edges: list[tuple[str, str]],
) -> tuple[str, str]:
    target_action = context.get("target_action")
    downstream_action = context.get("downstream_action")
    preferred = (target_action, downstream_action)
    if (
        target_action
        and downstream_action
        and target_action in base_trace
        and downstream_action in base_trace
        and preferred in hard_precondition_edges
    ):
        return preferred
    return _choose_hard_precondition(base_trace, hard_precondition_edges)


def _choose_confused_action(
    base_trace: list[str],
    confusion_edges: set[tuple[str, str]],
    tools: dict[str, dict[str, Any]],
    graph_nodes: list[str],
    preferred_action: str | None = None,
) -> tuple[str, str]:
    graph_node_set = set(graph_nodes)
    if preferred_action:
        for expected, replacement in confusion_edges:
            if (
                expected == preferred_action
                and expected in base_trace
                and (replacement in tools or replacement in graph_node_set)
            ):
                return expected, replacement
    for expected, replacement in confusion_edges:
        if expected in base_trace and (replacement in tools or replacement in graph_node_set):
            return expected, replacement
    raise CaseSpecError("wrong_tool requires a confusion_edge from a base trace action.")


def _choose_parameter_error_tool(
    context: dict[str, str],
    error_type: str,
    base_trace: list[str],
    tools: dict[str, dict[str, Any]],
) -> str:
    target_action = context.get("target_action")
    if (
        target_action
        and target_action in base_trace
        and target_action in tools
        and error_type in tools[target_action]["errors"]
    ):
        return target_action
    return _choose_error_tool(error_type, base_trace, tools)


def _choose_error_tool(
    error_type: str,
    base_trace: list[str],
    tools: dict[str, dict[str, Any]],
) -> str:
    for action in base_trace:
        if action == "done" or action not in tools:
            continue
        if error_type not in tools[action]["errors"]:
            continue
        return action
    raise CaseSpecError(f"No tool in base trace supports {error_type}.")


def _insert_terminal_error(
    base_trace: list[str],
    error_type: str,
    tool_id: str,
) -> list[str]:
    trace: list[str] = []
    for action in base_trace:
        if action == tool_id:
            trace.append(f"{error_type}@{tool_id}")
            break
        trace.append(action)
    trace.append("done")
    return trace


def _wrong_parameter_not_recovered_trace(
    base_trace: list[str],
    tool_id: str,
    context: dict[str, str],
) -> list[str]:
    downstream_action = context.get("downstream_action")
    if not downstream_action or downstream_action not in base_trace:
        return _insert_terminal_error(base_trace, "wrong_parameter", tool_id)

    trace: list[str] = []
    for action in base_trace:
        if action == tool_id:
            trace.append(f"wrong_parameter@{tool_id}")
            continue
        if action == downstream_action:
            trace.append(action)
            break
        trace.append(action)
    trace.append("done")
    return trace


def _replace_with_error_marker(
    base_trace: list[str],
    error_type: str,
    tool_id: str,
) -> list[str]:
    trace: list[str] = []
    replaced = False
    for action in base_trace:
        if action == tool_id and not replaced:
            trace.append(f"{error_type}@{tool_id}")
            replaced = True
        else:
            trace.append(action)
    if not replaced:
        raise CaseSpecError(f"Cannot replace {tool_id!r}; it is absent from the base trace.")
    return trace


def _insert_error_before(
    base_trace: list[str],
    error_type: str,
    tool_id: str,
) -> list[str]:
    trace: list[str] = []
    inserted = False
    for action in base_trace:
        if action == tool_id and not inserted:
            trace.append(f"{error_type}@{tool_id}")
            inserted = True
        trace.append(action)
    if not inserted:
        raise CaseSpecError(f"Cannot insert {error_type!r}; tool {tool_id!r} is absent.")
    return trace


def _validate_base_trace(
    task: str,
    base_trace: list[str],
    tools: dict[str, dict[str, Any]],
    graph_nodes: list[str],
) -> None:
    known = set(tools) | set(graph_nodes)
    if not base_trace:
        raise CaseSpecError(f"Template {task} has an empty base_trace.")
    for action in base_trace:
        if action not in known:
            raise CaseSpecError(f"Template {task} references unknown action {action!r}.")


def _template_context(
    template: dict[str, Any],
    base_trace: list[str],
    tools: dict[str, dict[str, Any]],
    graph_nodes: list[str],
) -> dict[str, str]:
    known = set(tools) | set(graph_nodes)
    context: dict[str, str] = {}
    user_goal = template.get("user_goal")
    if isinstance(user_goal, str) and user_goal.strip():
        context["user_goal"] = user_goal.strip()

    for key in ("target_action", "downstream_action"):
        action = template.get(key)
        if action is None:
            continue
        if not isinstance(action, str) or not action:
            raise CaseSpecError(f"Template {template['task']} has invalid {key}.")
        if action not in known:
            raise CaseSpecError(
                f"Template {template['task']} {key} references unknown action {action!r}."
            )
        if action not in base_trace:
            raise CaseSpecError(
                f"Template {template['task']} {key} must be present in base_trace."
            )
        context[key] = action
    return context


def _not_recovered_labels(error_type: str, context: dict[str, str]) -> list[str]:
    labels = [error_type, "not_recovered", "target_answer_unresolved"]
    if context.get("downstream_action"):
        labels.append("downstream_action_taken")
    return labels


def _validate_spec_shape(spec: dict[str, Any]) -> None:
    required = {
        "case_id",
        "task",
        "condition",
        "error",
        "trace",
        "expected_outcome",
        "labels",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise CaseSpecError(f"Case spec is missing fields: {missing}")
    if not isinstance(spec["trace"], list) or not spec["trace"]:
        raise CaseSpecError(f"{spec.get('case_id')} has an invalid trace.")
    if spec["expected_outcome"] not in {EXPECTED_OUTCOME_COMPLETED, EXPECTED_OUTCOME_FAILED}:
        raise CaseSpecError(f"{spec['case_id']} has an invalid expected_outcome.")
    if not isinstance(spec["labels"], list):
        raise CaseSpecError(f"{spec['case_id']} has invalid labels.")


def _normalise_graph(action_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": list(action_graph.get("nodes", [])),
        "dependency_edges": _edge_list(action_graph.get("dependency_edges", [])),
        "hard_precondition_edges": _edge_list(
            action_graph.get("hard_precondition_edges", [])
        ),
        "recovery_edges": _edge_list(action_graph.get("recovery_edges", [])),
        "confusion_edges": _edge_list(action_graph.get("confusion_edges", [])),
        "extra_step_candidates": _edge_list(
            action_graph.get("extra_step_candidates", [])
        ),
    }


def _tool_map(tool_catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {tool["id"]: tool for tool in tool_catalog}


def _edge_list(edges: list[list[str]]) -> list[tuple[str, str]]:
    return [(edge[0], edge[1]) for edge in edges]


def _is_error_marker(action: str) -> bool:
    return "@" in action


def _split_error_marker(action: str) -> tuple[str, str]:
    error_type, tool_id = action.split("@", 1)
    return error_type, tool_id
