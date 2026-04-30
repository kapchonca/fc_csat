from __future__ import annotations

import json
from typing import Any


CONDITION_RENDERING_INSTRUCTIONS = {
    "correct": "The assistant completes the customer request smoothly and safely.",
    "skip_step": (
        "The assistant moves forward without an expected check. The customer should "
        "be able to notice that an important check was missed."
    ),
    "extra_step": (
        "The assistant performs one unnecessary additional step before completing "
        "the request."
    ),
    "wrong_order": (
        "The assistant performs actions in an unsafe or illogical order, such as "
        "looking up sensitive information before confirming the customer."
    ),
    "wrong_tool": (
        "The assistant performs a related but incorrect action, such as showing "
        "recent transactions instead of identifying the specific pending payment."
    ),
    "wrong_parameter_recovered": (
        "The assistant initially uses insufficient or incorrect details, then asks "
        "for clarification or corrects course. The request succeeds."
    ),
    "wrong_parameter_not_recovered": (
        "The assistant uses insufficient or incorrect details and does not properly "
        "correct course. The request fails."
    ),
    "tool_failure_recovered": (
        "The assistant encounters a temporary system issue, retries or resolves it, "
        "and the request succeeds."
    ),
    "tool_failure_not_recovered": (
        "The assistant encounters a system issue and fails to complete the request."
    ),
}


def build_prompt(
    case_spec: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> str:
    tools = {tool["id"]: tool for tool in tool_catalog}
    relevant_tools = _relevant_tools(case_spec, tools)
    message_min = generation_config["output_message_min"]
    message_max = generation_config["output_message_max"]
    forbidden_tool_ids = ", ".join(sorted(tool["id"] for tool in relevant_tools))

    return "\n".join(
        [
            "You are a natural-language renderer for a controlled survey experiment.",
            "Setting: retail banking support chat between one customer and one assistant.",
            "The interaction structure is predefined. Do not invent new labels, outcomes, or hidden actions.",
            "",
            "Hidden experiment metadata. Use it only to render the dialogue; never reveal it.",
            f"Case id: {case_spec['case_id']}",
            f"User goal: {_humanize_task(case_spec['task'])}",
            f"Condition: {case_spec['condition']}",
            f"Expected outcome: {case_spec['expected_outcome']}",
            f"Hidden action trace: {json.dumps(case_spec['trace'])}",
            f"Experimental labels: {json.dumps(case_spec['labels'])}",
            "",
            "Relevant declarative action labels. These are not real API calls.",
            _format_relevant_tools(relevant_tools),
            "",
            "How the condition should be visible:",
            CONDITION_RENDERING_INSTRUCTIONS[case_spec["condition"]],
            "",
            "Dialogue requirements:",
            f"- Write approximately {message_min} to {message_max} total messages.",
            "- Alternate naturally between customer and assistant.",
            "- Use realistic but fictional banking details such as merchant names, amounts, dates, and account endings.",
            "- Make the expected final outcome clear in the final assistant message.",
            "- The assistant can describe customer-facing checks, delays, or issues in plain language.",
            "",
            "Forbidden behavior:",
            "- Do not mention internal tool ids, trace, case id, condition names, labels, or error labels.",
            "- Do not use words such as trace, tool, condition, wrong_parameter, missing_input, tool_failure, skip_step, wrong_order, wrong_tool, recovered, not_recovered, task_completed, or task_failed.",
            f"- Do not mention exact action ids: {forbidden_tool_ids}.",
            "- Do not add markdown, comments, explanations, headings, or text outside the JSON.",
            "",
            "Output format:",
            "Return only a valid JSON array in this exact shape:",
            '[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]',
        ]
    )


def _relevant_tools(
    case_spec: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    action_ids: list[str] = []
    for action in case_spec["trace"]:
        action_id = action.split("@", 1)[-1]
        if action_id in tools and action_id not in action_ids:
            action_ids.append(action_id)

    error = case_spec.get("error")
    if error and error.get("replacement") in tools:
        replacement = error["replacement"]
        if replacement not in action_ids:
            action_ids.append(replacement)
    return [tools[action_id] for action_id in action_ids]


def _format_relevant_tools(relevant_tools: list[dict[str, Any]]) -> str:
    return "\n".join(
        "- {id}: {description} Inputs: {inputs}. Adds: {adds}. Possible errors: {errors}.".format(
            id=tool["id"],
            description=tool["description"],
            inputs=", ".join(tool["inputs"]),
            adds=", ".join(tool["adds"]),
            errors=", ".join(tool["errors"]),
        )
        for tool in relevant_tools
    )


def _humanize_task(task: str) -> str:
    if task == "cancel_payment":
        return "cancel a pending payment before it is processed"
    return task.replace("_", " ")
