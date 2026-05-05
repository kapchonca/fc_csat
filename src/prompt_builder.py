from __future__ import annotations

import json
from typing import Any


CONDITION_RENDERING_INSTRUCTIONS = {
    "correct": "The assistant completes the customer request smoothly and safely.",
    "skip_step": (
        "The assistant moves forward without an expected check. The customer should "
        "be able to notice that an important check was missed. The assistant then "
        "corrects the omission and completes the request."
    ),
    "extra_step": (
        "The assistant performs one unnecessary additional step before completing "
        "the request."
    ),
    "wrong_order": (
        "The assistant performs actions in an unsafe or illogical order. The customer "
        "should be able to notice the order problem, after which the assistant corrects "
        "course and completes the request."
    ),
    "wrong_tool": (
        "The assistant performs a related but incorrect action. The customer should be "
        "able to notice the mismatch, after which the assistant uses the correct action "
        "and completes the request."
    ),
    "wrong_parameter": (
        "The assistant initially uses insufficient or incorrect details. The assistant "
        "then obtains or applies the correct details and completes the request."
    ),
}


def build_prompt(
    case_spec: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> str:
    return build_dialogue_prompt(
        case_spec=case_spec,
        tool_catalog=tool_catalog,
        generation_config=generation_config,
        dialogue_plan=None,
    )


def build_dialogue_plan_prompt(
    case_spec: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> str:
    tools = {tool["id"]: tool for tool in tool_catalog}
    relevant_tools = _relevant_tools(case_spec, tools)
    plan_config = _dialogue_plan_config(generation_config)
    message_count = plan_config["message_count"]
    role_pattern = plan_config["role_pattern"]
    forbidden_tool_ids = ", ".join(sorted(tool["id"] for tool in relevant_tools))

    return "\n".join(
        [
            "Prompt type: dialogue_plan",
            "You are planning a controlled retail banking support dialogue for a survey experiment.",
            "Create only a high-level message plan. Do not write the final dialogue yet.",
            "",
            "Hidden experiment metadata. Use it only to plan the dialogue; never reveal it.",
            f"Case id: {case_spec['case_id']}",
            f"User goal: {_humanize_task(case_spec['task'])}",
            f"Condition: {case_spec['condition']}",
            f"Expected outcome: {case_spec['expected_outcome']}",
            f"Hidden action trace: {json.dumps(case_spec['trace'])}",
            f"Error metadata: {json.dumps(case_spec.get('error'))}",
            f"Experimental labels: {json.dumps(case_spec['labels'])}",
            "",
            "Relevant declarative action labels. These are not real API calls.",
            _format_relevant_tools(relevant_tools),
            "",
            "Planning requirements:",
            f"- Return exactly {message_count} plan items.",
            f"- Use this exact role pattern: {json.dumps(role_pattern)}.",
            "- Each item must describe the purpose of that message in natural language.",
            "- The plan must make the condition visible to a survey respondent.",
            "- The plan must make the expected final outcome clear.",
            "- Refer to banking actions only in customer-facing plain language.",
            "",
            "Condition-specific planning instruction:",
            _condition_planning_instruction(case_spec, tools),
            "",
            "Forbidden behavior:",
            "- Do not mention internal tool ids, trace, case id, condition names, labels, or error labels.",
            f"- Do not mention exact action ids: {forbidden_tool_ids}.",
            "- Do not add markdown, comments, explanations, headings, or text outside the JSON.",
            "",
            "Output format:",
            "Return only a valid JSON array in this exact shape:",
            '[{"role":"user","purpose":"..."},{"role":"assistant","purpose":"..."}]',
        ]
    )


def build_dialogue_prompt(
    case_spec: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
    generation_config: dict[str, Any],
    dialogue_plan: list[dict[str, Any]] | None = None,
) -> str:
    tools = {tool["id"]: tool for tool in tool_catalog}
    relevant_tools = _relevant_tools(case_spec, tools)
    forbidden_tool_ids = ", ".join(sorted(tool["id"] for tool in relevant_tools))
    semantic_variant = _semantic_variant(generation_config)
    if dialogue_plan is None:
        message_requirement = (
            f"- Write approximately {generation_config['output_message_min']} to "
            f"{generation_config['output_message_max']} total messages."
        )
        plan_lines = []
    else:
        message_requirement = f"- Write exactly {len(dialogue_plan)} total messages."
        plan_lines = [
            "",
            "Validated dialogue plan. Follow it exactly for roles and message purposes:",
            json.dumps(dialogue_plan, ensure_ascii=False, indent=2),
        ]

    return "\n".join(
        [
            "Prompt type: dialogue_render",
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
            f"Error metadata: {json.dumps(case_spec.get('error'))}",
            f"Experimental labels: {json.dumps(case_spec['labels'])}",
            f"Semantic variant id: {semantic_variant['id']}",
            *plan_lines,
            "",
            "Relevant declarative action labels. These are not real API calls.",
            _format_relevant_tools(relevant_tools),
            "",
            "How the condition should be visible:",
            CONDITION_RENDERING_INSTRUCTIONS[case_spec["condition"]],
            "",
            "Dialogue requirements:",
            message_requirement,
            "- Follow the role order from the validated plan when a plan is provided.",
            "- Use realistic but fictional banking details such as merchant names, amounts, dates, and account endings.",
            "- Make the expected final outcome clear in the final assistant message.",
            "- The assistant can describe customer-facing checks, delays, or issues in plain language.",
            "- Preserve the semantic style from the validated plan.",
            f"- Semantic rendering instruction: {semantic_variant['instruction']}",
            "",
            "Forbidden behavior:",
            "- Do not mention internal tool ids, trace, case id, condition names, labels, or error labels.",
            "- Do not mention the semantic variant id or say that a style variant is being used.",
            "- Do not use words such as trace, tool, condition, wrong_parameter, missing_input, skip_step, wrong_order, wrong_tool, task_completed, or task_failed.",
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


def _condition_planning_instruction(
    case_spec: dict[str, Any],
    tools: dict[str, dict[str, Any]],
) -> str:
    condition = case_spec["condition"]
    error = case_spec.get("error") or {}
    if condition == "correct":
        return "Plan a smooth exchange where the assistant handles the request in the proper order."
    if condition == "skip_step":
        skipped = _action_description(error.get("at"), tools)
        return (
            "Plan an exchange where the assistant moves forward without this expected "
            f"customer-facing check: {skipped}. The customer should notice the omission, "
            "then the assistant should correct the omitted check and complete the request."
        )
    if condition == "extra_step":
        extra = _action_description(error.get("at"), tools)
        anchor = _action_description(error.get("anchor"), tools)
        return (
            "Plan an exchange where the assistant performs an unnecessary additional "
            f"step after {anchor}: {extra}. The request can still be completed."
        )
    if condition == "wrong_order":
        performed = _action_description(error.get("at"), tools)
        required = _action_description(error.get("required_before"), tools)
        return (
            "Plan an exchange where the assistant does this too early: "
            f"{performed}. It should have happened only after: {required}. "
            "The customer should be able to object to the order, then the assistant "
            "should correct the sequence and complete the request."
        )
    if condition == "wrong_tool":
        expected = _action_description(error.get("at"), tools)
        replacement = _action_description(error.get("replacement"), tools)
        return (
            "Plan an exchange where the assistant performs a related but incorrect "
            f"action: {replacement}. The expected action was: {expected}. The assistant "
            "should then correct course and complete the request."
        )
    if condition == "wrong_parameter":
        action = _action_description(error.get("at"), tools)
        return (
            "Plan an exchange where the assistant tries to proceed with insufficient "
            f"or incorrect details for this action: {action}. The assistant should then "
            "resolve the details and complete the request."
        )
    return CONDITION_RENDERING_INSTRUCTIONS[condition]


def _action_description(
    action_id: str | None,
    tools: dict[str, dict[str, Any]],
) -> str:
    if action_id and action_id in tools:
        return tools[action_id]["description"]
    if action_id:
        return action_id.replace("_", " ")
    return "the required banking support step"


def _dialogue_plan_config(generation_config: dict[str, Any]) -> dict[str, Any]:
    return generation_config["dialogue_plan"]


def _semantic_variant(generation_config: dict[str, Any]) -> dict[str, str]:
    variant = generation_config.get("semantic_variant")
    if isinstance(variant, dict):
        return {
            "id": str(variant.get("id", "default")),
            "instruction": str(
                variant.get(
                    "instruction",
                    "Use neutral wording without changing the scenario.",
                )
            ),
        }
    return {
        "id": "default",
        "instruction": "Use neutral wording without changing the scenario.",
    }


def _humanize_task(task: str) -> str:
    if task == "cancel_payment":
        return "cancel a pending payment before it is processed"
    return task.replace("_", " ")
