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

        for condition in template["conditions"]:
            spec = _generate_condition_spec(
                task=template["task"],
                base_trace=base_trace,
                condition=condition,
                tools=tools,
                graph=graph,
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
        )

    if condition == "skip_step":
        skipped = _choose_skip_action(base_trace, graph["hard_precondition_edges"])
        trace = [action for action in base_trace if action != skipped]
        return _spec(
            task=task,
            condition=condition,
            parts=[skipped],
            error={"type": "skip_step", "at": skipped},
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_FAILED,
            labels=["skip_step", EXPECTED_OUTCOME_FAILED],
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
        )

    if condition == "wrong_order":
        before, after = _choose_hard_precondition(base_trace, graph["hard_precondition_edges"])
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
            },
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_FAILED,
            labels=["wrong_order", EXPECTED_OUTCOME_FAILED],
        )

    if condition == "wrong_tool":
        expected, replacement = _choose_confused_action(
            base_trace,
            graph["confusion_edges"],
            tools,
            graph["nodes"],
        )
        trace = [replacement if action == expected else action for action in base_trace]
        return _spec(
            task=task,
            condition=condition,
            parts=[expected],
            error={
                "type": "wrong_tool",
                "at": expected,
                "replacement": replacement,
            },
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_FAILED,
            labels=["wrong_tool", EXPECTED_OUTCOME_FAILED],
        )

    if condition == "wrong_parameter":
        tool_id = _choose_error_tool(
            "wrong_parameter",
            base_trace,
            tools,
        )
        trace = _insert_terminal_error(base_trace, "wrong_parameter", tool_id)
        return _spec(
            task=task,
            condition=condition,
            parts=[tool_id],
            error={"type": "wrong_parameter", "at": tool_id},
            trace=trace,
            expected_outcome=EXPECTED_OUTCOME_FAILED,
            labels=["wrong_parameter", EXPECTED_OUTCOME_FAILED],
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
) -> dict[str, Any]:
    suffix = "_".join([condition, *parts]) if parts else condition
    return {
        "case_id": f"{task}_{suffix}",
        "task": task,
        "condition": condition,
        "error": error,
        "trace": list(trace),
        "expected_outcome": expected_outcome,
        "labels": labels,
    }


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


def _choose_hard_precondition(
    base_trace: list[str],
    hard_precondition_edges: set[tuple[str, str]],
) -> tuple[str, str]:
    for before, after in hard_precondition_edges:
        if before in base_trace and after in base_trace:
            return before, after
    raise CaseSpecError("wrong_order requires a hard_precondition_edge in the base trace.")


def _choose_confused_action(
    base_trace: list[str],
    confusion_edges: set[tuple[str, str]],
    tools: dict[str, dict[str, Any]],
    graph_nodes: list[str],
) -> tuple[str, str]:
    graph_node_set = set(graph_nodes)
    for expected, replacement in confusion_edges:
        if expected in base_trace and (replacement in tools or replacement in graph_node_set):
            return expected, replacement
    raise CaseSpecError("wrong_tool requires a confusion_edge from a base trace action.")


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
